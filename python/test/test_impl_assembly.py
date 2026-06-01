# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for matrix assembly on *real* implicit-domain meshes (not
the reproducibility-friendly mock mesh).

The mock unfitted mesh used by ``test_matrix`` / ``test_vector`` /
``test_scalar`` is a clever construct: on cut cells it repeats the
*standard* quadrature several times and rescales the weights inversely
so that the custom assembly result is provably equal to the standard
one. That symmetry catches a lot of bookkeeping bugs, but it cannot
catch bugs that depend on the *real* custom-quadrature point
coordinates (e.g. an indexing mistake that lands on the wrong
quadrature point but still sums to the same value when the points are
symmetric repeats of one another).

This module assembles forms on actual implicit cut domains -- a disk
in 2D and a sphere in 3D -- using qugar-generated quadrature, and
checks invariants the cut geometry must satisfy.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import check_vals, dtypes  # type: ignore

import qugar.impl
from qugar.dolfinx import dsu, form_custom
from qugar.mesh import create_unfitted_impl_Cartesian_mesh


def _make_disk_mesh(n_cells, dtype):
    """Build a 2D unfitted Cartesian mesh containing a disk of radius
    0.4 centered at (0.5, 0.5)."""
    impl = qugar.impl.create_disk(
        radius=dtype(0.4), center=np.array([0.5, 0.5], dtype=dtype)
    )
    n = [n_cells, n_cells]
    return create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD,
        impl,
        n,
        np.zeros(2, dtype),
        np.ones(2, dtype),
        exclude_empty_cells=True,
    )


def _make_sphere_mesh(n_cells, dtype):
    """Build a 3D unfitted Cartesian mesh containing a sphere of
    radius 0.4 centered at (0.5, 0.5, 0.5)."""
    impl = qugar.impl.create_sphere(
        radius=dtype(0.4), center=np.array([0.5, 0.5, 0.5], dtype=dtype)
    )
    n = [n_cells, n_cells, n_cells]
    return create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD,
        impl,
        n,
        np.zeros(3, dtype),
        np.ones(3, dtype),
        exclude_empty_cells=True,
    )


def _exact_volume(dim, radius):
    """Closed-form volume / area of a ``dim``-dimensional ball."""
    if dim == 2:
        return float(np.pi) * radius**2
    elif dim == 3:
        return 4.0 / 3.0 * float(np.pi) * radius**3
    raise ValueError(dim)


def _exact_surface(dim, radius):
    """Closed-form surface area / perimeter of a ``dim``-dimensional
    sphere boundary."""
    if dim == 2:
        return 2.0 * float(np.pi) * radius
    elif dim == 3:
        return 4.0 * float(np.pi) * radius**2
    raise ValueError(dim)


@pytest.mark.parametrize("dtype", dtypes)
def test_volume_via_constant_integral(dtype):
    """Assembling ``1 * dx`` on the cut implicit domain must recover
    the analytic area (2D disk)."""
    unf = _make_disk_mesh(16, dtype)
    one = dolfinx.fem.Constant(unf, dtype(1.0))
    form = form_custom(one * ufl.dx(domain=unf, degree=4), dtype=dtype)
    coeffs = form.pack_coefficients()
    vol = dolfinx.fem.assemble_scalar(form, coeffs=coeffs)
    check_vals(np.array([vol]), np.array([_exact_volume(2, 0.4)]), rtol=2.0e-3, dtype=dtype)


@pytest.mark.parametrize("dtype", dtypes)
def test_sphere_volume(dtype):
    """Assembling ``1 * dx`` on the cut implicit sphere domain must
    recover the analytic volume (3D)."""
    unf = _make_sphere_mesh(16, dtype)
    one = dolfinx.fem.Constant(unf, dtype(1.0))
    form = form_custom(one * ufl.dx(domain=unf, degree=4), dtype=dtype)
    coeffs = form.pack_coefficients()
    vol = dolfinx.fem.assemble_scalar(form, coeffs=coeffs)
    check_vals(np.array([vol]), np.array([_exact_volume(3, 0.4)]), rtol=2.0e-3, dtype=dtype)


@pytest.mark.parametrize("dtype", dtypes)
def test_disk_unfitted_surface(dtype):
    """Assembling ``1 * dsu`` on the cut implicit domain must
    recover the analytic perimeter (2D disk)."""
    unf = _make_disk_mesh(16, dtype)
    one = dolfinx.fem.Constant(unf, dtype(1.0))
    form = form_custom(
        one * dsu(domain=unf, degree=4), dtype=dtype
    )
    coeffs = form.pack_coefficients()
    perim = dolfinx.fem.assemble_scalar(form, coeffs=coeffs)
    check_vals(
        np.array([perim]), np.array([_exact_surface(2, 0.4)]), rtol=2.0e-3, dtype=dtype
    )


@pytest.mark.parametrize("dtype", dtypes)
def test_mass_matrix_symmetric(dtype):
    """The custom-quadrature mass matrix on an implicit-domain mesh
    must be symmetric. This is a basic correctness invariant that
    catches index / transposition mistakes in the qugar assembly
    pipeline that a one-point quadrature couldn't detect."""
    unf = _make_disk_mesh(8, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf, degree=5)
    form = form_custom(a, dtype=dtype)
    coeffs = form.pack_coefficients()
    M = dolfinx.fem.assemble_matrix(form, coeffs=coeffs)
    M.scatter_reverse()
    M_dense = M.to_scipy().toarray()
    asym = np.abs(M_dense - M_dense.T).max()
    # The symmetry tolerance scales with the matrix entries; allow a
    # generous absolute floor for float32.
    atol = 1.0e-5 if dtype is np.float32 else 1.0e-11
    assert asym < atol, f"Mass matrix asymmetric: max |M - M.T| = {asym:.3e}"


@pytest.mark.parametrize("dtype", dtypes)
def test_stiffness_matrix_symmetric(dtype):
    """Same as ``test_mass_matrix_symmetric`` for the stiffness
    matrix. Exercises ``grad-grad`` rather than ``u*v``."""
    unf = _make_disk_mesh(8, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf, degree=5)
    form = form_custom(a, dtype=dtype)
    coeffs = form.pack_coefficients()
    K = dolfinx.fem.assemble_matrix(form, coeffs=coeffs)
    K.scatter_reverse()
    K_dense = K.to_scipy().toarray()
    asym = np.abs(K_dense - K_dense.T).max()
    atol = 1.0e-4 if dtype is np.float32 else 1.0e-10
    assert asym < atol, f"Stiffness matrix asymmetric: max |K - K.T| = {asym:.3e}"


@pytest.mark.parametrize("dtype", dtypes)
def test_mass_row_sum_equals_volume(dtype):
    """For a partition-of-unity FE space, ``sum_i sum_j M_ij``
    equals the integral of ``1`` over the domain, i.e. the area /
    volume. This relates the matrix entries to a geometric quantity
    we can compare to the analytic answer."""
    unf = _make_disk_mesh(16, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf, degree=4)
    form = form_custom(a, dtype=dtype)
    coeffs = form.pack_coefficients()
    M = dolfinx.fem.assemble_matrix(form, coeffs=coeffs)
    M.scatter_reverse()
    total = M.to_scipy().toarray().sum()
    check_vals(np.array([total]), np.array([_exact_volume(2, 0.4)]), rtol=2.0e-3, dtype=dtype)
