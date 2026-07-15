# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # Poisson problem on a domain with several circular holes

#
# This demo is implemented in {download}`demo_multiple_holes.py` and it illustrates:
#
# - How to describe a 2D domain with more than one circular hole by combining several implicit
#   functions through
#   {py:func}`create_functions_intersection<qugar.impl.create_functions_intersection>`.
# - How to solve a Poisson problem on that unfitted domain (Dirichlet on the outer boundary,
#   Neumann on the holes).
# - How to visualize the solution with PyVista.

# A single circular hole is obtained by taking the
# {py:func}`negative<qugar.impl.create_negative>` of a
# {py:func}`disk<qugar.impl.create_disk>`: the resulting function is negative *outside* the disk,
# so its (negative) domain is the plate minus the disk. To carve *several* disjoint holes we
# intersect the negative regions of several such functions with
# {py:func}`create_functions_intersection<qugar.impl.create_functions_intersection>`: the active
# domain is the region where *all* the functions are simultaneously negative (outside *every*
# disk). Internally this relies on Algoim's multi-polynomial quadrature, so all the disks must be
# described through (Bezier) polynomials (`use_bzr=True`).
#
# ```{note}
# The holes must be pairwise disjoint. Overlapping disks would require a (non-polynomial) union
# operation, which is outside the scope of this polynomial-only construction.
# ```

# ## Modules import

# +
from qugar.utils import has_FEniCSx, has_PETSc, has_PyVista

if not has_FEniCSx:
    raise ValueError("FEniCSx installation is required.")

if not has_PETSc:
    raise ValueError("petsc4py installation is required.")

if not has_PyVista:
    raise ValueError("PyVista installation is required.")
# -

# +
from mpi4py import MPI

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import pyvista as pv
import ufl
from dolfinx import default_scalar_type as dtype

import qugar.impl
import qugar.mesh
import qugar.reparam
from qugar.dolfinx import LinearProblem, ds_bdry_unf, form_custom, mapped_normal
from qugar.impl import create_disk, create_functions_intersection, create_negative

# -

# ## Geometry definition
#
# We define a plate $[0,1]^2$ with three disjoint circular holes, each given by its center and
# radius. Each hole is a negated disk, and the domain is the intersection of their negative
# regions.

# +
radii = [0.15, 0.12, 0.1]
centers = [
    np.array([0.25, 0.25], dtype=dtype),
    np.array([0.7, 0.3], dtype=dtype),
    np.array([0.5, 0.75], dtype=dtype),
]

holes = [
    create_negative(create_disk(radius=r, center=c, use_bzr=True))
    for r, c in zip(radii, centers)
]

impl_func = create_functions_intersection(holes)
# -

# ## Unfitted mesh
#
# We embed the domain in an unfitted Cartesian mesh over $[0,1]^2$. Empty cells (fully inside a
# hole) are excluded from the mesh.

# +
n_cells = 32
unf_mesh = qugar.mesh.create_unfitted_impl_Cartesian_mesh(
    MPI.COMM_WORLD, impl_func, n_cells, exclude_empty_cells=True, dtype=dtype
)
# -

# ## Problem definition
#
# We use a manufactured solution $u_{\text{ex}}(x,y) = \sin(\pi x)\,\sin(\pi y)$, which vanishes
# on the outer boundary of the square (so the Dirichlet condition there is simply $u = 0$). The
# source term is $f = -\Delta u_{\text{ex}}$ and the Neumann datum on the holes is
# $g = \nabla u_{\text{ex}} \cdot n$.

# +
x = ufl.SpatialCoordinate(unf_mesh)
uex = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
f = -ufl.div(ufl.grad(uex))
# -

# We define a Lagrange finite element space and impose $u = 0$ strongly on the outer boundary of
# the square (the hole boundaries are immersed and are handled weakly through the Neumann term).

# +
degree = 2
V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

facets = dolfinx.mesh.locate_entities_boundary(
    unf_mesh,
    dim=(unf_mesh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0)
    | np.isclose(x[0], 1.0)
    | np.isclose(x[1], 0.0)
    | np.isclose(x[1], 1.0),
)
dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bc = dolfinx.fem.dirichletbc(value=dtype(0), dofs=dofs, V=V)
# -

# ## Variational forms
#
# The bilinear and linear forms are the standard Poisson ones. The linear form includes the
# Neumann contribution on the unfitted (hole) boundary, integrated with the
# {py:func}`ds_bdry_unf<qugar.dolfinx.ds_bdry_unf>` measure and the runtime
# {py:func}`mapped_normal<qugar.dolfinx.mapped_normal>`.

# +
n_quad_pts = degree + 1
quad_degree = 2 * n_quad_pts + 1

ds_unf = ds_bdry_unf(domain=unf_mesh, degree=quad_degree)
g = ufl.dot(ufl.grad(uex), mapped_normal(unf_mesh))

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx(degree=quad_degree)
L = f * v * ufl.dx(degree=quad_degree) + g * v * ds_unf
# -

# ## Solve
#
# We solve the linear system with a direct solver (Cholesky) and a Jacobi diagonal scaling to
# mitigate the ill-conditioning typical of unfitted discretizations.

# +
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "ksp_diagonal_scale": True,
}

problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)
problem.solve()
uh = problem.u
# -

# As a sanity check we compute the $L^2$ error against the manufactured solution over the holed
# domain.

# +
error_form = form_custom((uh - uex) ** 2 * ufl.dx(degree=quad_degree), dtype=dtype)
error_local = dolfinx.fem.assemble_scalar(error_form, coeffs=error_form.pack_coefficients())
l2_error = np.sqrt(unf_mesh.comm.allreduce(error_local, op=MPI.SUM))
if unf_mesh.comm.rank == 0:
    print(f"L2 error: {l2_error:.3e}")
# -

# ## Visualization
#
# We reparameterize the holed domain, interpolate the discrete solution onto the
# reparameterization mesh, and plot it with PyVista. Setting `pyvista.OFF_SCREEN = True` (or the
# `PYVISTA_OFF_SCREEN` environment variable) saves a screenshot instead of opening a window.

# +
rep_degree = 3
reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=rep_degree, levelset=False)
rep_mesh = reparam.create_mesh()

Vrep = dolfinx.fem.functionspace(rep_mesh, ("Lagrange", rep_degree))
interp_data = qugar.reparam.create_interpolation_data(Vrep, V)
uh_rep = dolfinx.fem.Function(Vrep, dtype=dtype)
uh_rep.interpolate_nonmatching(uh, *interp_data)
uh_rep.name = "u_h"

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(Vrep)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["u_h"] = uh_rep.x.array.real
grid.set_active_scalars("u_h")

pl = pv.Plotter(window_size=[900, 900])
pl.add_mesh(grid, cmap="viridis", show_edges=False, scalar_bar_args={"title": "u_h"})
pl.view_xy()
pl.camera.zoom(1.3)

if pv.OFF_SCREEN:
    pl.screenshot("demo_multiple_holes_poisson.png")
else:
    pl.show()
# -
