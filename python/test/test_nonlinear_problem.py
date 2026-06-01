# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""End-to-end tests for the stock ``dolfinx.fem.petsc.NonlinearProblem``
(PETSc SNES) on an unfitted mesh.

qugar no longer ships its own ``NonlinearProblem``; importing
``qugar.dolfinx`` patches DOLFINx so the *stock* SNES-based
``dolfinx.fem.petsc.NonlinearProblem`` assembles the unfitted residual
and Jacobian with qugar's runtime quadrature transparently. This module
covers that nonlinear-solve path in pytest with a small scalar nonlinear
problem so regressions in qugar's coefficient packing / re-assemble path
show up as test failures.

The chosen problem is a simple algebraic nonlinearity solved through
the FE machinery rather than a real PDE: the residual is

    F(u, v) = (u^3 - f) * v * dx,    Jacobian J(u; du, v) = 3 * u^2 * du * v * dx

so the unique solution is ``u_h = (f)^(1/3)`` (interpolated). With a
mock unfitted mesh the custom quadrature is by construction
equivalent to the standard one, so the solution must match the
analytic cube root at the DOFs.
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
import numpy as np
import pytest
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from utils import check_vals, create_mock_unfitted_mesh, dtypes  # type: ignore

import qugar.dolfinx  # noqa: F401  (import applies the transparent-assembly patches)

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3

_PETSC_DTYPES = [d for d in dtypes if np.dtype(d) == np.dtype(ScalarType)]

# SNES + LU options for a Newton line-search solve with tight tolerances.
_SNES_OPTIONS = {
    "snes_type": "newtonls",
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-10,
    "snes_max_it": 30,
    "ksp_type": "preonly",
    "pc_type": "lu",
}


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_cuberoot_nonlinear(dim, simplex_cell, dtype):
    """Solve ``u^3 = f`` pointwise via a Newton iteration on the
    variational residual ``(u^3 - f) v dx = 0``.

    With ``f`` interpolating ``2 + x[0]`` (always positive in [0, 1]^d)
    the unique solution is ``u = (2 + x[0])^(1/3)``. We verify the
    Newton iterate converges to that interpolant.
    """
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))

    # Right-hand side.
    f = dolfinx.fem.Function(V, dtype=dtype)
    f.interpolate(lambda x: 2.0 + x[0])  # type: ignore

    # Unknown, initialized close enough to the solution for Newton
    # to converge cleanly.
    u = dolfinx.fem.Function(V, dtype=dtype)
    u.interpolate(lambda x: np.full_like(x[0], 1.5, dtype=dtype))  # type: ignore

    v = ufl.TestFunction(V)
    F_form = (u**3 - f) * v * ufl.dx(domain=unf)

    problem = NonlinearProblem(
        F_form,
        u,
        petsc_options=_SNES_OPTIONS,
        petsc_options_prefix=f"qugar_cuberoot_{dim}{int(simplex_cell)}_",
    )
    problem.solve()
    assert problem.solver.getConvergedReason() > 0, "SNES solver did not converge"
    u.x.scatter_forward()

    # Reference: interpolant of (2 + x[0])^(1/3).
    u_ref = dolfinx.fem.Function(V, dtype=dtype)
    u_ref.interpolate(lambda x: (2.0 + x[0]) ** (1.0 / 3.0))  # type: ignore

    # Loose tolerance: the FE space cannot represent (2+x)^(1/3)
    # exactly, but the Newton iteration's L^2 projection onto V must
    # match dolfinx's L^2 projection of the same function to a
    # comparable precision. We compare DOF values against the
    # cube-root interpolant with a degree-driven tolerance.
    rtol = 1.0e-3
    atol = 1.0e-3
    check_vals(u.x.array, u_ref.x.array, rtol=rtol, atol=atol, dtype=dtype)


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
@pytest.mark.parametrize("simplex_cell", [True, False])
def test_nonlinear_resolve(simplex_cell, dtype):
    """Run two consecutive SNES solves on the same ``NonlinearProblem``
    object, mutating the right-hand-side coefficient between solves.

    This exercises qugar's coefficient re-packing path: each SNES
    residual/Jacobian assembly re-packs the custom coefficients, so the
    second solve must reflect the mutated ``f`` and not silently reuse the
    previous values.
    """
    unf = create_mock_unfitted_mesh(2, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    f = dolfinx.fem.Function(V, dtype=dtype)
    u = dolfinx.fem.Function(V, dtype=dtype)
    v = ufl.TestFunction(V)
    F_form = (u**3 - f) * v * ufl.dx(domain=unf)

    problem = NonlinearProblem(
        F_form,
        u,
        petsc_options=_SNES_OPTIONS,
        petsc_options_prefix=f"qugar_cuberoot_resolve_{int(simplex_cell)}_",
    )

    # Solve 1: f = 2 + x[0].
    f.interpolate(lambda x: 2.0 + x[0])  # type: ignore
    u.interpolate(lambda x: np.full_like(x[0], 1.5, dtype=dtype))  # type: ignore
    problem.solve()
    u.x.scatter_forward()
    u1 = np.copy(u.x.array)

    # Solve 2: f = 8 + 0*x[0] (uniform; solution should be uniform 2).
    f.interpolate(lambda x: np.full_like(x[0], 8.0, dtype=dtype))  # type: ignore
    u.interpolate(lambda x: np.full_like(x[0], 1.5, dtype=dtype))  # type: ignore
    problem.solve()
    u.x.scatter_forward()
    u2 = u.x.array

    # The two solves must give different solutions...
    assert not np.allclose(u1, u2), (
        "Second Newton solve produced the same result as the first despite "
        "the RHS Function being mutated; coefficient update path is broken."
    )
    # ...and the second must converge to the uniform cube root of 8.
    check_vals(u2, np.full_like(u2, 2.0, dtype=dtype), rtol=1.0e-4, atol=1.0e-4, dtype=dtype)
