# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""End-to-end tests for the stock ``dolfinx.fem.petsc.LinearProblem`` on
unfitted meshes.

qugar no longer ships its own ``LinearProblem``; importing
``qugar.dolfinx`` patches DOLFINx so the *stock*
``dolfinx.fem.petsc.LinearProblem`` assembles unfitted forms with qugar's
runtime quadrature transparently. These tests exercise that path inside
the pytest suite so regressions show up as test failures rather than as
silently broken demos.

The mock unfitted mesh -- which on cut cells repeats the standard
quadrature several times and rescales the weights inversely -- gives
an L^2 projection equal (within numerical tolerance) to the one
computed on a fitted mesh with standard quadrature. We use that
equivalence as the regression assertion.
"""

from qugar.utils import has_FEniCSx, has_PETSc

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")
if not has_PETSc:
    pytest_collectstart = "petsc4py installation not found"
    import pytest as _pytest

    _pytest.skip(pytest_collectstart, allow_module_level=True)


from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from dolfinx.fem.petsc import LinearProblem
from utils import (  # type: ignore
    check_vals,
    create_mock_unfitted_mesh,
    dtypes,
)

import qugar.dolfinx  # noqa: F401  (import applies the transparent-assembly patches)

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3

_PETSC_DTYPES = [d for d in dtypes if np.dtype(d) == np.dtype(ScalarType)]


# ---------------------------------------------------------------------------
# L^2 projection via LinearProblem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_L2_projection(dim, simplex_cell, dtype):
    """L^2 projection of a polynomial onto a Lagrange space.

    Because the mock unfitted mesh's custom quadrature is a (re-scaled)
    re-use of the standard quadrature on cut cells, the projection
    matrix and RHS are identical to the standard ones, and the solution
    must equal the standard-assembled projection. We compare against
    a direct standard-DOLFINx solve.
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    target = dolfinx.fem.Function(V, dtype=dtype)
    target.interpolate(lambda x: 1 + 0.5 * x[0] + x[1] ** 2)  # type: ignore

    a = u * v * ufl.dx(domain=unf)
    L = target * v * ufl.dx(domain=unf)

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = LinearProblem(
        a,
        L,
        petsc_options=petsc_options,
        petsc_options_prefix=f"qugar_l2_{dim}{int(simplex_cell)}_",
    )
    problem.solve()
    uh_custom = problem.u

    # An L^2 projection of a polynomial onto a polynomial space of equal
    # or higher degree reproduces the target exactly up to numerics.
    check_vals(uh_custom.x.array, target.x.array, dtype=dtype)


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_resolve(dim, simplex_cell, dtype):
    """Solve twice with the same matrix but different RHS targets. The
    second solve goes through ``LinearProblem.solve`` again, which
    re-packs coefficients each time -- verifies the RHS-only path."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    target = dolfinx.fem.Function(V, dtype=dtype)
    target.interpolate(lambda x: 1 + x[0])  # type: ignore

    a = u * v * ufl.dx(domain=unf)
    L = target * v * ufl.dx(domain=unf)

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = LinearProblem(
        a,
        L,
        petsc_options=petsc_options,
        petsc_options_prefix=f"qugar_resolve_{dim}{int(simplex_cell)}_",
    )
    problem.solve()
    uh1 = np.copy(problem.u.x.array)

    # Same problem instance, but the source has changed.
    target.interpolate(lambda x: 2 - x[1])  # type: ignore
    problem.solve()
    uh2 = problem.u.x.array

    # The two solutions must differ (the RHS changed)...
    assert not np.allclose(uh1, uh2), (
        "Second solve produced the same result as the first despite the "
        "RHS being mutated -- the LinearProblem solve path is not "
        "re-packing coefficients."
    )

    # ...and the second solve should reproduce the new target.
    target_new = target.x.array
    check_vals(uh2, target_new, dtype=dtype)
