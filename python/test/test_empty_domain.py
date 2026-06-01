# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Edge-case tests for degenerate implicit domains.

Two degenerate configurations to stress:

* **Empty intersection** -- the implicit function returns the empty
  set inside the bounding box (e.g. a disk entirely outside the unit
  square). All mesh cells are empty; ``dx`` integrals must evaluate
  to exactly zero without raising.
* **Full intersection** -- the implicit function covers the whole
  bounding box (e.g. a disk of radius >> bounding box diameter). All
  cells are full; ``dx`` integrals must equal the bounding box
  volume, and unfitted-boundary integrals must be zero (there is no
  unfitted boundary inside the bounding box).

These configurations exercise null-array / empty-loop paths in
qugar's coefficient packing and quadrature aggregation that the
normal cut-mesh tests cannot reach.
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


def _build_mesh(impl, n_cells, dtype):
    return create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD,
        impl,
        [n_cells, n_cells],
        np.zeros(2, dtype),
        np.ones(2, dtype),
        # IMPORTANT: do NOT exclude empty cells. The point of the
        # test is to exercise the path where they exist.
        exclude_empty_cells=False,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", dtypes)
def test_empty_domain_volume_is_zero(dtype):
    """A disk of radius 0.05 centered well outside the unit square
    yields an empty cut domain. The volume integral must be exactly
    zero."""
    impl = qugar.impl.create_disk(
        radius=dtype(0.05), center=np.array([-5.0, -5.0], dtype=dtype)
    )
    unf = _build_mesh(impl, 8, dtype)

    one = dolfinx.fem.Constant(unf, dtype(1.0))
    form = form_custom(one * ufl.dx(domain=unf, degree=2), dtype=dtype)
    vol = dolfinx.fem.assemble_scalar(form, coeffs=form.pack_coefficients())
    assert np.isclose(vol, 0.0, atol=1.0e-12), (
        f"Empty domain volume integral must be 0, got {vol!r}"
    )


@pytest.mark.parametrize("dtype", dtypes)
def test_empty_domain_unfitted_surface_is_zero(dtype):
    """No unfitted boundary inside the bounding box -- the
    ``dsu`` integral must be exactly zero."""
    impl = qugar.impl.create_disk(
        radius=dtype(0.05), center=np.array([-5.0, -5.0], dtype=dtype)
    )
    unf = _build_mesh(impl, 8, dtype)

    one = dolfinx.fem.Constant(unf, dtype(1.0))
    form = form_custom(one * dsu(domain=unf, degree=2), dtype=dtype)
    perim = dolfinx.fem.assemble_scalar(form, coeffs=form.pack_coefficients())
    assert np.isclose(perim, 0.0, atol=1.0e-12), (
        f"Empty domain unfitted-boundary integral must be 0, got {perim!r}"
    )


@pytest.mark.parametrize("dtype", dtypes)
def test_full_domain_volume_equals_box(dtype):
    """A disk whose radius covers the whole [0, 1]^2 yields a full
    domain (no cut cells). The volume integral must equal the
    bounding-box area (1.0)."""
    impl = qugar.impl.create_disk(
        radius=dtype(10.0), center=np.array([0.5, 0.5], dtype=dtype)
    )
    unf = _build_mesh(impl, 8, dtype)

    one = dolfinx.fem.Constant(unf, dtype(1.0))
    form = form_custom(one * ufl.dx(domain=unf, degree=2), dtype=dtype)
    vol = dolfinx.fem.assemble_scalar(form, coeffs=form.pack_coefficients())
    rtol = 1.0e-5 if dtype is np.float32 else 1.0e-12
    check_vals(np.array([vol]), np.array([1.0]), rtol=rtol, dtype=dtype)


@pytest.mark.parametrize("dtype", dtypes)
def test_full_domain_unfitted_surface_is_zero(dtype):
    """The implicit boundary lies entirely outside the bounding box
    so there is no unfitted boundary inside it -- the
    ``dsu`` integral must be zero."""
    impl = qugar.impl.create_disk(
        radius=dtype(10.0), center=np.array([0.5, 0.5], dtype=dtype)
    )
    unf = _build_mesh(impl, 8, dtype)

    one = dolfinx.fem.Constant(unf, dtype(1.0))
    form = form_custom(one * dsu(domain=unf, degree=2), dtype=dtype)
    perim = dolfinx.fem.assemble_scalar(form, coeffs=form.pack_coefficients())
    assert np.isclose(perim, 0.0, atol=1.0e-12), (
        f"Full domain unfitted-boundary integral must be 0, got {perim!r}"
    )


@pytest.mark.parametrize("dtype", dtypes)
def test_empty_domain_mass_matrix_assembles(dtype):
    """The mass matrix on an empty cut domain must assemble cleanly
    (no exceptions, no NaNs / Infs). All entries are zero."""
    impl = qugar.impl.create_disk(
        radius=dtype(0.05), center=np.array([-5.0, -5.0], dtype=dtype)
    )
    unf = _build_mesh(impl, 8, dtype)

    V = dolfinx.fem.functionspace(unf, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx(domain=unf, degree=2)
    form = form_custom(a, dtype=dtype)
    M = dolfinx.fem.assemble_matrix(form, coeffs=form.pack_coefficients())
    M.scatter_reverse()
    M_dense = M.to_scipy().toarray()
    assert np.all(np.isfinite(M_dense)), "Empty domain mass matrix has NaN/Inf"
    assert np.all(M_dense == 0.0), "Empty domain mass matrix must be all zeros"
