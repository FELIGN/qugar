# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Mixed-element form assembly on a real implicit-domain mesh.

``test_elements.py`` covers mixed elements (Taylor-Hood, Lagrange +
DG, Nedelec + Lagrange) on the mock unfitted mesh, where the custom
quadrature is by construction equivalent to the standard quadrature.
``test_impl_assembly.py`` covers real implicit-domain assembly with
scalar Lagrange spaces. The composition -- mixed elements on a real
cut domain -- is otherwise untested.

The forms below assemble a block-mass matrix on a Taylor-Hood-ish
mixed space over a disk-shaped cut domain and check:

* the matrix is symmetric;
* its row-sum equals the cut domain measure (partition-of-unity
  invariant on the diagonal blocks);
* the velocity block and pressure block are decoupled (no
  off-diagonal entries between them in the chosen form).
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import check_vals, dtypes  # type: ignore

import qugar.impl
from qugar.dolfinx import form_custom
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

_RADIUS = 0.4
_DISK_AREA = float(np.pi) * _RADIUS**2


def _build_disk_mesh(n_cells, dtype):
    impl = qugar.impl.create_disk(
        radius=dtype(_RADIUS), center=np.array([0.5, 0.5], dtype=dtype)
    )
    return create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD,
        impl,
        [n_cells, n_cells],
        np.zeros(2, dtype),
        np.ones(2, dtype),
        exclude_empty_cells=True,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", dtypes)
def test_taylor_hood_block_mass_matrix(dtype):
    """Assemble a *decoupled* block-mass matrix on a Taylor-Hood
    mixed space (P2-vector velocity + P1-scalar pressure) over the
    implicit disk:

        a((u, p), (v, q)) = inner(u, v) dx + p * q dx

    The matrix must be symmetric and positive-definite, with the
    velocity and pressure blocks fully decoupled."""
    unf = _build_disk_mesh(16, dtype)
    el_v = basix.ufl.element("Lagrange", unf.basix_cell(), 2, shape=(2,), dtype=dtype)
    el_q = basix.ufl.element("Lagrange", unf.basix_cell(), 1, dtype=dtype)
    el_mixed = basix.ufl.mixed_element([el_v, el_q])
    W = dolfinx.fem.functionspace(unf, el_mixed)

    up, vq = ufl.TrialFunction(W), ufl.TestFunction(W)
    u, p = ufl.split(up)
    v, q = ufl.split(vq)
    a = (ufl.inner(u, v) + p * q) * ufl.dx(domain=unf, degree=5)

    form = form_custom(a, dtype=dtype)
    M = dolfinx.fem.assemble_matrix(form, coeffs=form.pack_coefficients())
    M.scatter_reverse()
    M_dense = M.to_scipy().toarray()

    # Symmetry.
    asym = np.abs(M_dense - M_dense.T).max()
    atol_sym = 1.0e-5 if dtype is np.float32 else 1.0e-11
    assert asym < atol_sym, f"Mass matrix asymmetric: max |M - M.T| = {asym:.3e}"

    # Velocity-pressure block decoupling: the velocity DOFs and
    # pressure DOFs share no non-zero entries. Get the DOF maps for
    # each sub-space.
    V_sub, V_map = W.sub(0).collapse()
    Q_sub, Q_map = W.sub(1).collapse()
    # DOLFINx 0.11 ``FunctionSpace.collapse`` returns the dof map as a list of
    # arrays (one per cell type); these meshes have a single cell type.
    V_dofs = np.asarray(V_map[0], dtype=np.int64)
    Q_dofs = np.asarray(Q_map[0], dtype=np.int64)
    cross_block = np.abs(M_dense[np.ix_(V_dofs, Q_dofs)]).max()
    atol_cross = 1.0e-6 if dtype is np.float32 else 1.0e-12
    assert cross_block < atol_cross, (
        f"Velocity-pressure block coupling is non-zero: {cross_block:.3e}"
    )

    # The pressure-pressure block is a scalar mass matrix on the
    # cut disk: sum of all entries = disk area.
    Q_block_sum = M_dense[np.ix_(Q_dofs, Q_dofs)].sum()
    check_vals(np.array([Q_block_sum]), np.array([_DISK_AREA]), rtol=2.0e-3, dtype=dtype)


@pytest.mark.parametrize("dtype", dtypes)
def test_mixed_lagrange_dg_on_impl_domain(dtype):
    """Same invariants as ``test_taylor_hood_block_mass_matrix`` but
    with a continuous-vector + discontinuous-scalar mixed space
    (P2-vector + DG P0 scalar)."""
    unf = _build_disk_mesh(12, dtype)
    el_v = basix.ufl.element("Lagrange", unf.basix_cell(), 2, shape=(2,), dtype=dtype)
    el_q = basix.ufl.element(
        "Lagrange", unf.basix_cell(), 0, discontinuous=True, dtype=dtype
    )
    el_mixed = basix.ufl.mixed_element([el_v, el_q])
    W = dolfinx.fem.functionspace(unf, el_mixed)

    up, vq = ufl.TrialFunction(W), ufl.TestFunction(W)
    u, p = ufl.split(up)
    v, q = ufl.split(vq)
    a = (ufl.inner(u, v) + p * q) * ufl.dx(domain=unf, degree=5)

    form = form_custom(a, dtype=dtype)
    M = dolfinx.fem.assemble_matrix(form, coeffs=form.pack_coefficients())
    M.scatter_reverse()
    M_dense = M.to_scipy().toarray()

    # Symmetry.
    asym = np.abs(M_dense - M_dense.T).max()
    atol_sym = 1.0e-5 if dtype is np.float32 else 1.0e-11
    assert asym < atol_sym

    # Block decoupling.
    _, V_map = W.sub(0).collapse()
    _, Q_map = W.sub(1).collapse()
    # DOLFINx 0.11 ``FunctionSpace.collapse`` returns the dof map as a list of
    # arrays (one per cell type); these meshes have a single cell type.
    V_dofs = np.asarray(V_map[0], dtype=np.int64)
    Q_dofs = np.asarray(Q_map[0], dtype=np.int64)
    cross_block = np.abs(M_dense[np.ix_(V_dofs, Q_dofs)]).max()
    atol_cross = 1.0e-6 if dtype is np.float32 else 1.0e-12
    assert cross_block < atol_cross

    # The pressure block (DG P0) is diagonal with entries equal to
    # the area of each cut cell. Its trace equals the total disk area.
    Q_block_diag = np.diag(M_dense[np.ix_(Q_dofs, Q_dofs)])
    check_vals(np.array([Q_block_diag.sum()]), np.array([_DISK_AREA]), rtol=2.0e-3, dtype=dtype)
