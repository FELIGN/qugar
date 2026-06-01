# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Workarounds for upstream dolfinx bugs.

This module patches dolfinx in-place to fix bugs that affect qugar's
custom-coefficient flow. Each patch is gated on the dolfinx version so
it becomes a no-op as soon as the upstream fix is released, and the
whole module can be deleted when qugar's minimum supported dolfinx
version includes every fix.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx
import dolfinx.cpp
import dolfinx.fem
import dolfinx.fem.assemble as _assemble_module


def _patch_assemble_scalar() -> None:
    """Fix dolfinx 0.10.0 ``assemble_scalar`` swapping ``coeffs`` for
    ``constants``.

    In dolfinx 0.10.0 (``python/dolfinx/fem/assemble.py``, line 163),
    ``assemble_scalar`` reads::

        coeffs = pack_coefficients(M) if coeffs is None else constants

    so any user-supplied ``coeffs`` is silently replaced by
    ``constants``. qugar relies on passing custom coefficients to the
    assembler, so this breaks every custom-form assembly.

    Fixed upstream in dolfinx PR #4186 (commit ``77265120f6``,
    2026-04-27) but not yet present in any released v0.10.x tag
    (checked up to v0.10.0.post5). The companion ``assemble_vector`` /
    ``assemble_matrix`` functions in the same file were inspected and
    do *not* have this bug.

    Remove this patch once qugar requires a dolfinx version that
    contains the fix.
    """
    if not dolfinx.__version__.startswith("0.10."):
        return
    if getattr(_assemble_module.assemble_scalar, "_qugar_patched", False):
        return

    _cpp_assemble_scalar = dolfinx.cpp.fem.assemble_scalar
    _original = _assemble_module.assemble_scalar

    def assemble_scalar(M, constants=None, coeffs=None):
        # Resolve ``pack_constants`` / ``pack_coefficients`` from the module
        # at call time (not capture time) so that qugar's later
        # ``pack_coefficients`` patch — which dispatches ``CustomForm`` to
        # its custom packer (see qugar.dolfinx._assembly_patches) — is
        # honoured here too, making ``assemble_scalar(custom_form)`` correct
        # without an explicit ``coeffs=`` argument.
        constants = _assemble_module.pack_constants(M) if constants is None else constants
        coeffs = _assemble_module.pack_coefficients(M) if coeffs is None else coeffs
        return _cpp_assemble_scalar(M._cpp_object, constants, coeffs)

    assemble_scalar.__doc__ = _original.__doc__
    assemble_scalar.__name__ = _original.__name__
    assemble_scalar._qugar_patched = True  # type: ignore[attr-defined]

    _assemble_module.assemble_scalar = assemble_scalar
    dolfinx.fem.assemble_scalar = assemble_scalar


def apply_patches() -> None:
    """Apply every dolfinx patch needed by qugar."""
    _patch_assemble_scalar()
