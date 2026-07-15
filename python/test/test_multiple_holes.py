# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for domains with several circular holes described through the intersection of the
negative regions of several Bezier polynomials (Algoim multi-polynomial quadrature)."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from collections.abc import Sequence

import numpy as np
import pytest
from test_volume_primitives import volume_test_and_area_test  # type: ignore
from utils import dtypes  # type: ignore

import qugar.impl
from qugar.impl import ImplicitFunc


def create_holes_domain(
    radii: Sequence[float],
    centers: Sequence[Sequence[float]],
    dtype: type[np.float32 | np.float64],
) -> ImplicitFunc:
    """Creates a unit-square domain with circular holes.

    Each hole is a negated disk (negative outside the disk), and the domain is the intersection
    of their negative regions, i.e., the region outside every disk.

    Args:
        radii (Sequence[float]): Radii of the holes.
        centers (Sequence[Sequence[float]]): Centers of the holes.
        dtype (type[np.float32 | np.float64]): Scalar type used to build the geometry.

    Returns:
        ImplicitFunc: Implicit function describing the domain with holes.
    """
    holes = [
        qugar.impl.create_negative(
            qugar.impl.create_disk(radius=dtype(r), center=np.asarray(c, dtype=dtype), use_bzr=True)
        )
        for r, c in zip(radii, centers)
    ]
    return qugar.impl.create_functions_intersection(holes)


# Disjoint circular holes fully contained in the unit square, with 1, 2 and 3 holes.
holes_cases = [
    ([0.3], [[0.5, 0.5]]),
    ([0.2, 0.15], [[0.3, 0.3], [0.7, 0.65]]),
    ([0.15, 0.12, 0.1], [[0.25, 0.25], [0.7, 0.3], [0.5, 0.75]]),
]


@pytest.mark.parametrize("radii,centers", holes_cases)
@pytest.mark.parametrize("n_cells", [16])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
def test_multiple_holes(
    radii: Sequence[float],
    centers: Sequence[Sequence[float]],
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
):
    """Tests the volume and unfitted-boundary length of a unit square with several holes.

    For disjoint holes fully contained in the square, the area is ``1 - sum(pi r_i^2)`` and the
    unfitted-boundary length is ``sum(2 pi r_i)`` (the circle circumferences).

    Args:
        radii (Sequence[float]): Radii of the holes.
        centers (Sequence[Sequence[float]]): Centers of the holes.
        n_cells (int): Number of cells per direction in the Cartesian mesh.
        n_quad_pts (int): Number of quadrature points per direction for the cut cells.
        dtype (type): Scalar type used for the computations.
    """
    func = create_holes_domain(radii, centers, dtype)
    assert func.dim == 2

    exact_volume = dtype(1.0 - np.pi * np.sum(np.square(radii)))
    exact_area = dtype(2.0 * np.pi * np.sum(radii))

    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, exact_area, dtype)


def test_intersection_requires_bezier():
    """Non-Bezier operands are rejected by the intersection operator."""
    disk = qugar.impl.create_disk(radius=0.3, center=np.array([0.5, 0.5]), use_bzr=False)
    with pytest.raises(ValueError):
        qugar.impl.create_functions_intersection([qugar.impl.create_negative(disk)])


def test_intersection_requires_matching_dim():
    """Mixing dimensions is rejected."""
    disk = qugar.impl.create_disk(radius=0.3, center=np.array([0.5, 0.5]), use_bzr=True)
    sphere = qugar.impl.create_sphere(radius=0.3, center=np.array([0.5, 0.5, 0.5]), use_bzr=True)
    with pytest.raises(AssertionError):
        qugar.impl.create_functions_intersection(
            [qugar.impl.create_negative(disk), qugar.impl.create_negative(sphere)]
        )
