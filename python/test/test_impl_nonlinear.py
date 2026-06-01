# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Nonlinear solve end-to-end on a real implicit-domain mesh.

``test_nonlinear_problem.py`` exercises the SNES-solve path on the
mock unfitted mesh (where the custom quadrature is by construction
equivalent to the standard one); ``test_impl_assembly.py`` /
``test_impl_poisson.py`` exercise real implicit-domain assembly with
linear forms. The composition -- a nonlinear solve on a real cut
domain via the stock ``dolfinx.fem.petsc.NonlinearProblem`` -- is not
tested elsewhere.
"""

from qugar.utils import has_FEniCSx, has_PETSc

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")
if not has_PETSc:
    import pytest as _pytest

    _pytest.skip("petsc4py installation not found", allow_module_level=True)


from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import dolfinx
import dolfinx.fem
import numpy as np
import pytest
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from utils import check_vals, dtypes  # type: ignore

import qugar.dolfinx  # noqa: F401  (import applies the transparent-assembly patches)
import qugar.impl
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

_PETSC_DTYPES = [d for d in dtypes if np.dtype(d) == np.dtype(ScalarType)]


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
def test_cuberoot_on_implicit_disk(dtype):
    """Solve ``(u^3 - f) v dx = 0`` (pointwise: ``u = f^{1/3}``) on
    a real implicit disk via qugar's ``NonlinearProblem`` + the
    DOLFINx Newton solver. With ``f = 8`` (constant) the unique
    solution is ``u = 2`` (constant); a Lagrange FE space represents
    constants exactly, so the test asserts the FE solution is the
    constant 2 on the cut domain to roundoff."""
    impl = qugar.impl.create_disk(
        radius=dtype(0.4), center=np.array([0.5, 0.5], dtype=dtype)
    )
    unf = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD,
        impl,
        [12, 12],
        np.zeros(2, dtype),
        np.ones(2, dtype),
        exclude_empty_cells=True,
        dtype=dtype,
    )

    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))

    u = dolfinx.fem.Function(V, dtype=dtype)
    # Initial guess far enough from the solution that Newton has to
    # iterate but close enough to converge in a few steps.
    u.interpolate(lambda x: np.full_like(x[0], 1.5, dtype=dtype))  # type: ignore

    f = dolfinx.fem.Constant(unf, dtype(8.0))
    v = ufl.TestFunction(V)
    F_form = (u**3 - f) * v * ufl.dx(domain=unf, degree=6)

    problem = NonlinearProblem(
        F_form,
        u,
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-10,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="qugar_impl_cuberoot_",
    )
    problem.solve()
    assert problem.solver.getConvergedReason() > 0, "SNES did not converge"
    u.x.scatter_forward()

    # Inactive DOFs (those not connected to any cell with custom or
    # standard quadrature on the cut domain) keep their initial
    # value, so restrict the assertion to the active DOFs that the
    # Newton system actually solved for. A DOF is active iff it has
    # a non-zero column in the assembled Jacobian.
    J_form = ufl.derivative(F_form, u, ufl.TrialFunction(V))
    from qugar.dolfinx import form_custom  # noqa: PLC0415

    J = form_custom(J_form, dtype=dtype)
    A = dolfinx.fem.assemble_matrix(J, coeffs=J.pack_coefficients())
    A.scatter_reverse()
    A_dense = A.to_scipy().toarray()
    active = np.where(np.abs(A_dense).sum(axis=0) > 1.0e-12)[0]
    assert len(active) > 0, "No active DOFs found"

    expected = np.full(len(active), 2.0, dtype=dtype)
    check_vals(u.x.array[active], expected, rtol=1.0e-6, atol=1.0e-6, dtype=dtype)
