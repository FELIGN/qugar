# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""End-to-end Poisson solve on a real implicit-domain mesh, combining
the four qugar-specific code paths most relevant to real workflows:

* ``qugar.dolfinx.LinearProblem`` (solver + apply_lifting + set_bc)
* ``qugar.mesh.create_unfitted_impl_Cartesian_mesh`` (real cut domain
  rather than the mock unfitted mesh)
* ``qugar.dolfinx.dsu`` (integration on the unfitted
  boundary)
* ``qugar.dolfinx.dsu_normal`` (the physical normal at the
  unfitted boundary)

A refactor that breaks any one of these combined paths should fail
this test even if the unit tests for each piece in isolation still
pass.

The chosen problem follows ``demo_poisson.py``:

    -Laplace(u)   = f    in Omega
              u   = 0    on Gamma_D (exterior mesh boundary)
    grad(u) . n   = g    on Gamma_N (the cut implicit boundary)

with manufactured ``u_ex = sin(pi x) sin(pi y)`` (which vanishes on
[0, 1]^2 so the Dirichlet condition is exact). The cut domain is
"plate with hole", i.e. [0, 1]^2 minus a disk at the origin.
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
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import dtypes  # type: ignore

import qugar.impl
from qugar.dolfinx import LinearProblem, dsu, dsu_normal
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

_PETSC_DTYPES = [d for d in dtypes if np.dtype(d) == np.dtype(ScalarType)]


def _build_plate_with_hole_mesh(n_cells, dtype):
    """Plate-with-hole cut domain: [0, 1]^2 minus a disk of radius
    0.3 centered at the origin."""
    impl = qugar.impl.create_disk(
        radius=dtype(0.3), center=np.array([0.0, 0.0], dtype=dtype)
    )
    impl = qugar.impl.create_negative(impl)
    return create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD,
        impl,
        [n_cells, n_cells],
        np.zeros(2, dtype),
        np.ones(2, dtype),
        exclude_empty_cells=True,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", _PETSC_DTYPES)
def test_poisson_neumann_plus_strong_dirichlet(dtype):
    """Solve the plate-with-hole Poisson problem from
    ``demo_poisson.py`` and verify the FE solution recovers the
    manufactured ``u_ex`` in L^2 norm with a moderate mesh."""
    unf_mesh = _build_plate_with_hole_mesh(16, dtype)
    degree = 2
    V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(unf_mesh)
    uex = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    grad_uex = ufl.grad(uex)
    f = -ufl.div(grad_uex)

    # Strong Dirichlet u = 0 on the exterior mesh boundary
    # (the part of [0, 1]^2 boundary that does not coincide with the
    # cut implicit boundary). ``u_ex`` vanishes there, so the exact
    # Dirichlet value is zero.
    tdim = unf_mesh.topology.dim
    fdim = tdim - 1
    unf_mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(unf_mesh.topology)
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dtype(0.0), bc_dofs, V)

    # Neumann data g = grad(u_ex) . n on the unfitted boundary.
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n_unf = dsu_normal(unf_mesh)
    n_quad = degree + 2
    qd = 2 * n_quad + 1
    ds_unf = dsu(domain=unf_mesh, degree=qd)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx(degree=qd)
    L = (
        f * v * ufl.dx(degree=qd)
        + ufl.dot(grad_uex, n_unf) * v * ds_unf
    )

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem.solve()
    uh = problem.u

    # L^2 error of (uh - u_ex) on the cut domain. ``form_custom`` is
    # qugar's compiled-form factory.
    from qugar.dolfinx import form_custom  # noqa: PLC0415

    err_form = form_custom(
        ufl.inner(uh - uex, uh - uex) * ufl.dx(degree=qd, domain=unf_mesh),
        dtype=dtype,
    )
    L2_sq = dolfinx.fem.assemble_scalar(err_form, coeffs=err_form.pack_coefficients())
    L2 = float(np.sqrt(unf_mesh.comm.allreduce(L2_sq, op=MPI.SUM)))

    # The combination of degree-2 Lagrange on a 16x16 Cartesian mesh
    # with a curved unfitted boundary gives L^2 errors around 1e-2.
    # The point of this assertion is to catch gross regressions
    # (e.g., wrong assembly producing O(1) errors), not to certify
    # an asymptotic rate.
    assert L2 < 2.0e-2, f"L^2 error too large: {L2:.3e}"
