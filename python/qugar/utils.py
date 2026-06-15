# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

import importlib.util
from typing import Tuple, cast

DOLFINX_VERSION_MIN = "0.11.0"
"""str: Minimum DOLFINx compatible version.
The imported DOLFINx version must be greater or equal than this version.
"""

DOLFINX_VERSION_MAX = "0.12.0"
"""str: Maximum DOLFINx non-compatible version.
The imported DOLFINx version must be smaller than this version.
"""

BASIX_VERSION_MIN = "0.11.0"
"""str: Minimum Basix compatible version.
The imported Basix version must be greater or equal than this version.
"""

BASIX_VERSION_MAX = "0.12.0"
"""str: Maximum Basix non-compatible version.
The imported Basix version must be smaller than this version.
"""

FFCX_VERSION_MIN = "0.11.0"
"""str: Minimum FFCx compatible version.
The imported FFCx version must be greater or equal than this version.
"""

FFCX_VERSION_MAX = "0.12.0"
"""str: Maximum FFCx non-compatible version.
The imported FFCx version must be smaller than this version.
"""

UFL_VERSION_MIN = "2026.1.0"
"""str: Minimum UFL compatible version.
The imported UFL version must be greater or equal than this version.
"""

UFL_VERSION_MAX = "2026.2.0"
"""str: Maximum UFL non-compatible version.
The imported UFL version must be smaller than this version.
"""


def _extract_version(version_str: str) -> Tuple[int, int, int]:
    """Extracts the major, minor, and revision numbers of a string with
    with format "X.Y.Z".

    Args:
        version_str (str): Version sring with format "X.Y.Z".

    Returns:
        Tuple[int,int,int]: Major, minor, and revision numbers (i.e.,
        X, Y, and Z).
    """
    vals = tuple(int(val) for val in version_str.split("."))
    vals = cast(Tuple[int, int, int], vals)
    assert len(vals) == 3
    return vals


def _compare_version_lt(version_lhs: str, version_rhs: str) -> bool:
    """Checks if the version_lhs is smaller than version_rhs.

    Args:
        version_lhs (str): Left-hand-side version to compare.
        version_rhs (str): Right-hand-side version to compare.

    Returns:
        bool: ``True`` if version_lhs is smaller than version_rhs,
        ``False`` otherwise.
    """
    lhs = _extract_version(version_lhs)
    rhs = _extract_version(version_rhs)

    if lhs[0] != rhs[0]:
        return lhs[0] < rhs[0]
    elif lhs[1] != rhs[1]:
        return lhs[1] < rhs[1]
    else:
        return lhs[2] < rhs[2]


def _compare_version_le(version_lhs: str, version_rhs: str) -> bool:
    """Checks if the version_lhs is smaller or equal than version_rhs.

    Args:
        version_lhs (str): Left-hand-side version to compare.
        version_rhs (str): Right-hand-side version to compare.

    Returns:
        bool: ``True`` if version_lhs is smaller or equal than version_rhs,
        ``False`` otherwise.
    """
    lhs = _extract_version(version_lhs)
    rhs = _extract_version(version_rhs)

    if lhs[0] > rhs[0]:
        return False
    elif lhs[1] > rhs[1]:
        return False
    else:
        return lhs[2] <= rhs[2]


def check_version_in_range(version, lower, upper) -> bool:
    """
    Checks if a version is within a specified range.

    Args:
        version (str): The version to check.
        lower (str): The lower bound of the version range (inclusive).
        upper (str): The upper bound of the version range (exclusive).

    Returns:
        bool: True if the version is within the range, False otherwise.
    """
    return _compare_version_le(lower, version) and _compare_version_lt(version, upper)


def check_DOLFINx() -> bool:
    """
    Checks if the installed version of DOLFINx is within the required range.

    This function attempts to import the DOLFINx library and compare its version
    against the minimum and maximum required versions. If the library is not
    installed or the version is out of the required range, it returns False.

    Returns:
        bool: True if DOLFINx is installed and its version is within the required range,
              False otherwise.
    """
    try:
        from dolfinx.cpp import __version__ as version

        return check_version_in_range(version, DOLFINX_VERSION_MIN, DOLFINX_VERSION_MAX)
    except ImportError:
        return False


def check_PETSc() -> bool:
    """
    Checks if petsc4py is installed.

    Returns:
        bool: Whether PETSc (petsc4py) is found.
    """
    try:
        from petsc4py import PETSc  # noqa: F401

        return True
    except ImportError:
        return False


def check_Basix() -> bool:
    """
    Checks if the installed version of Basix is within the required range.

    This function attempts to import the Basix library and compare its version
    against the minimum and maximum required versions. If the library is not
    installed or the version is out of the required range, it returns False.

    Returns:
        bool: True if Basix is installed and its version is within the required range,
              False otherwise.
    """
    try:
        from basix import __version__ as version  # type: ignore

        return check_version_in_range(version, BASIX_VERSION_MIN, BASIX_VERSION_MAX)
    except ImportError:
        return False


def check_FFCx() -> bool:
    """
    Checks if the installed version of FFCx is within the required range.

    This function attempts to import the FFCx library and compare its version
    against the minimum and maximum required versions. If the library is not
    installed or the version is out of the required range, it returns False.

    Returns:
        bool: True if FFCx is installed and its version is within the required range,
              False otherwise.
    """
    try:
        from ffcx import __version__ as version

        return check_version_in_range(version, FFCX_VERSION_MIN, FFCX_VERSION_MAX)
    except ImportError:
        return False


def check_UFL() -> bool:
    """
    Checks if the installed version of UFL is within the required range.

    This function attempts to import the UFL library and compare its version
    against the minimum and maximum required versions. If the library is not
    installed or the version is out of the required range, it returns False.

    Returns:
        bool: True if UFL is installed and its version is within the required range,
              False otherwise.
    """
    try:
        from ufl import __version__ as version

        return check_version_in_range(version, UFL_VERSION_MIN, UFL_VERSION_MAX)
    except ImportError:
        return False


"""Whether FEniCSx packages were available."""
has_FEniCSx: bool = check_DOLFINx() and check_Basix() and check_FFCx() and check_UFL()

"""Whether PETSc packages were available."""
has_PETSc: bool = check_PETSc()

"""Whether PyVista packages were available."""
has_PyVista: bool = importlib.util.find_spec("pyvista") is not None


__all__ = ["has_FEniCSx", "has_PyVista"]
