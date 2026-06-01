# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Build-once / cached basix tabulation shim used by the on-the-fly custom
kernel.

The generated C kernel references plain C symbols (``qugar_register_element_*``
and ``qugar_tabulate_*``) that this module's tiny ``extern "C"`` C++ wrapper
exposes. The wrapper itself links ``libbasix``. It is JIT-compiled the first
time a kernel needs it and cached on disk; downstream kernel JIT only has to
link the resulting shared library.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SHIM_SRC = _HERE / "qugar_basix_shim.cpp"
LIB_NAME = "qugar_basix_shim"
# Use the *platform* shared-library suffix (not Python's extension suffix from
# sysconfig, which is ``.so`` even on macOS): macOS's linker resolves ``-lname``
# only against ``libname.dylib`` (or .tbd), not ``.so``.
if sys.platform == "darwin":
    _SHLIB_SUFFIX = ".dylib"
elif sys.platform == "win32":
    _SHLIB_SUFFIX = ".dll"
else:
    _SHLIB_SUFFIX = ".so"
_LIB_FILENAME = f"lib{LIB_NAME}{_SHLIB_SUFFIX}"

_LOCK = threading.Lock()


def _cache_dir() -> Path:
    base = os.environ.get("QUGAR_SHIM_CACHE")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "qugar" / "shim"


def _find_cxx() -> Path:
    # Honour an explicit user/CI override first.
    cxx_env = os.environ.get("CXX")
    if cxx_env:
        p = Path(cxx_env)
        if p.exists():
            return p
        resolved = shutil.which(cxx_env)
        if resolved:
            return Path(resolved)
    # Prefer a compiler shipped in the active environment: it must match the
    # C++ ABI of the conda-forge libbasix the shim links (notably basix's
    # std::floating_point-constrained create_element mangling), which a system
    # compiler is not guaranteed to do. conda's compilers land in the env bin
    # either plain (clang++/g++) or toolchain-prefixed (x86_64-conda-linux-gnu-c++,
    # arm64-apple-darwin*-clang++). Only fall back to the system PATH if the env
    # ships none -- that risks an ABI mismatch, hence it is the last resort.
    bindir = Path(sys.prefix) / "bin"
    prefixed = sorted(
        p.name
        for pat in ("*-c++", "*-g++", "*-clang++")
        for p in bindir.glob(pat)
    )
    names = ["clang++", *sorted(p.name for p in bindir.glob("clang++-*")),
             "g++", "c++", *prefixed]
    for name in names:
        candidate = bindir / name
        if candidate.exists():
            return candidate
    for name in ["clang++", "g++", "c++"]:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    raise FileNotFoundError(
        f"no C++ compiler found in {bindir} or on PATH; set $CXX to override"
    )


def _find_basix_dirs() -> tuple[Path, Path]:
    """Locate basix's C++ header and library directories.

    The shim ``#include``s ``<basix/finite-element.h>`` and links
    ``libbasix``; both must come from the *same* basix install, otherwise an
    ABI / name-mangling mismatch surfaces only at kernel-load time as
    ``undefined symbol: basix::FiniteElement<...>::tabulate``.

    ``sys.prefix`` happens to hold both under a conda-forge install, which is
    why that assumption worked there. But in the official FEniCS Docker image
    the Python venv (``sys.prefix`` = ``/dolfinx-env``) and the C++ basix
    install (``/usr/local``) are *different* prefixes, so the old code linked
    a stale/mismatched libbasix. We instead probe a priority-ordered list of
    candidate prefixes and return the include/lib dirs of the first one that
    holds *both* the header and the library.

    Returns:
        ``(include_dir, lib_dir)`` of a self-consistent basix install.
    """
    header_rel = Path("include") / "basix" / "finite-element.h"

    candidates: list[Path] = []
    # CMAKE_PREFIX_PATH is the canonical way an install advertises its
    # location; the FEniCS Docker image points it at the C++ prefix.
    for entry in os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix))
    candidates += [
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path("/usr/local"),
        Path("/usr"),
    ]

    seen: set[Path] = set()
    for prefix in candidates:
        if prefix in seen:
            continue
        seen.add(prefix)
        if not (prefix / header_rel).exists():
            continue
        for libdir_name in ("lib", "lib64"):
            libdir = prefix / libdir_name
            if any(libdir.glob("libbasix.*")):
                return prefix / "include", libdir

    raise FileNotFoundError(
        "could not locate a basix install providing both "
        "<basix/finite-element.h> and libbasix; looked under "
        f"{[str(p) for p in seen]}. Set CMAKE_PREFIX_PATH to the basix prefix."
    )


@contextlib.contextmanager
def _file_lock(path: Path):
    """POSIX advisory exclusive file lock; no-op on platforms without fcntl.

    Serializes builds across processes (e.g. MPI ranks racing on the same
    cache directory) so only one build runs at a time. Idle waiters block
    until the holder releases.
    """
    try:
        import fcntl
    except ImportError:  # Windows -- skip; conda FEniCSx is POSIX in practice.
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def ensure_built() -> tuple[Path, str]:
    """Build the shim if missing or out of date.

    MPI/multi-process safe: cross-process serialized via a POSIX file
    lock on the cache directory (waiters block until the writer
    releases), in-process serialized via a threading lock. Double-
    checked locking inside the critical section avoids a redundant
    rebuild when another rank/process built it while we were waiting.

    Returns:
        (library_dir, library_name) suitable for passing to cffi's
        ``library_dirs`` and ``libraries``.
    """
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / _LIB_FILENAME
    # Treat a change to either the shim source OR this builder (which
    # controls the compile flags) as cache-invalidating.
    builder_mtime = Path(__file__).stat().st_mtime
    src_mtime = max(SHIM_SRC.stat().st_mtime, builder_mtime)

    # Fast path: cache hit without taking any lock.
    if lib.exists() and lib.stat().st_mtime >= src_mtime:
        return cache, LIB_NAME

    with _LOCK, _file_lock(cache / ".build.lock"):
        # Double-check under the lock — another rank/process/thread may
        # have built it while we were waiting.
        if lib.exists() and lib.stat().st_mtime >= src_mtime:
            return cache, LIB_NAME

        inc_dir, lib_dir = _find_basix_dirs()
        # Library flags MUST come *after* the source on the command line.
        # GNU ld defaults to --as-needed, which records a DT_NEEDED for
        # -lbasix only if it satisfies an as-yet-unresolved symbol at the
        # point the linker reaches it. With -lbasix before the source, no
        # basix symbol is pending yet, so the dependency is silently dropped
        # and the shim ends up with unresolved
        # basix::FiniteElement<...>::tabulate symbols that only fail at
        # kernel-load time. macOS's ld does not prune this way, which is why
        # the bug surfaced solely on the Linux docs build.
        cmd = [
            str(_find_cxx()),
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{inc_dir}",
            str(SHIM_SRC),
            f"-L{lib_dir}",
            "-lbasix",
            f"-Wl,-rpath,{lib_dir}",
            "-o",
            str(lib),
        ]
        # Align the shim's macOS deployment target with what cffi will use
        # when it links the kernel (otherwise ld warns "built for newer
        # version X" because conda clang's default target is much newer).
        if sys.platform == "darwin":
            target = (os.environ.get("MACOSX_DEPLOYMENT_TARGET")
                      or sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET"))
            if target:
                cmd.insert(cmd.index("-o"), f"-mmacosx-version-min={target}")
        subprocess.run(cmd, check=True)
        return cache, LIB_NAME


__all__ = ["ensure_built", "LIB_NAME"]
