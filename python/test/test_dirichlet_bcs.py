# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""End-to-end tests for the Dirichlet-BC + ``apply_lifting`` path of the
stock ``dolfinx.fem.petsc.LinearProblem`` on an unfitted mesh.

The standard ``test_matrix`` / ``test_vector`` suites assemble
matrices and vectors without any boundary conditions. The actual
Newton / time-stepping workflows live or die on ``apply_lifting``
correctly subtracting the BC contribution from the right-hand side,
and on the matrix being correctly zeroed (rows / columns + diagonal)
on constrained DOFs.

This module solves a small Poisson problem with non-homogeneous
Dirichlet BCs on the mock unfitted mesh through the stock
``dolfinx.fem.petsc.LinearProblem`` (which routes through qugar's
transparent-assembly patches because the mesh is unfitted). The mock's
custom quadrature is constructed to be equivalent to the standard
quadrature, so for a manufactured quadratic solution on a degree-2
space the discrete solution reproduces the exact solution; we use that
as the regression assertion. This covers the full
``assemble_matrix(bcs=...) + apply_lifting + set_bc`` pipeline inside
``LinearProblem.solve``.
"""

from qugar.utils import has_FEniCSx, has_PETSc

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")
if not has_PETSc:
    import pytest as _pytest

    _pytest.skip("petsc4py installation not found", allow_module_level=True)


from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx.fem.petsc import LinearProblem
from utils import check_vals, create_mock_unfitted_mesh, dtypes  # type: ignore

import qugar.dolfinx  # noqa: F401  (import applies the transparent-assembly patches)

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3

_PETSC_DTYPES = [d for d in dtypes if np.dtype(d) == np.dtype(ScalarType)]


def _build_poisson_problem(unf, dtype):
    """Build the Poisson UFL forms ``(a, L)`` over the mock unfitted
    mesh, plus a Dirichlet BC fixing ``u = g`` on the whole boundary."""
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Manufactured solution u_ex = 1 + x^2 + 2y^2, so -laplace(u_ex) = -6.
    # The weak form is  a(u, v) = inner(grad u, grad v)*dx  and
    # L(v) = f*v*dx with f = -laplace(u_ex) = -6. With u_ex in the degree-2
    # space and exact (here: equivalent) quadrature, the discrete solution
    # reproduces u_ex exactly.
    f = dolfinx.fem.Constant(unf, dtype(-6.0))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)
    L = f * v * ufl.dx(domain=unf)

    g = dolfinx.fem.Function(V, dtype=dtype)
    g.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore

    tdim = unf.topology.dim
    fdim = tdim - 1
    unf.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(unf.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(g, boundary_dofs)

    return V, a, L, bc


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_poisson_dirichlet(dim, simplex_cell, dtype):
    """Solve Poisson with non-homogeneous Dirichlet BCs through the stock
    ``dolfinx.fem.petsc.LinearProblem`` on the unfitted mesh and compare
    against the manufactured exact solution.

    On the mock unfitted mesh the custom quadrature is by construction
    equivalent to the standard quadrature, and the manufactured solution
    is a quadratic that lives in the degree-2 space, so the discrete
    solution reproduces it up to numerical precision. This covers the
    full ``assemble_matrix(bcs=...) + apply_lifting + set_bc`` pipeline
    inside ``LinearProblem.solve``.
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V, a, L, bc = _build_poisson_problem(unf, dtype)

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix=f"qugar_dirichlet_{dim}{int(simplex_cell)}_",
    )
    problem.solve()
    uh = problem.u

    # The manufactured solution u_ex (= bc.g, interpolated on the whole
    # space) is exactly reproduced by the degree-2 discrete solution.
    check_vals(uh.x.array, bc.g.x.array, dtype=dtype)


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_poisson_dirichlet_bc_honored(dim, simplex_cell, dtype):
    """The solution returned by ``LinearProblem`` must match the BC
    value on constrained DOFs to roundoff precision (catches
    regressions where ``set_bc`` is skipped or partially applied).
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V, a, L, bc = _build_poisson_problem(unf, dtype)

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"qugar_dirichlet_bc_{dim}{int(simplex_cell)}_",
    )
    problem.solve()
    uh = problem.u

    # On the constrained DOFs the solution equals the BC values.
    bc_dofs = bc.dof_indices()[0]
    bc_vals = bc.g.x.array[bc_dofs]
    check_vals(uh.x.array[bc_dofs], bc_vals, dtype=dtype)
