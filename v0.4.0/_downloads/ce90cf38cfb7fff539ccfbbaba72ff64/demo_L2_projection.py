# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # $L^2$ projection
#
# This demo is implemented in {download}`demo_L2_projection.py`. It
# illustrates:
#
# - How to perform a $L^2$ projection using QUGaR and FEniCSx.
# - How to export the generated result to VTK files.
#
# ## Problem definition
#
# Let us consider the domain $\Omega \subset \mathbb{R}^n$,
# immersed in a mesh $\mathcal{T}(\Omega)$ (see
# [Divergence theorem demo](demo_div_thm.md) first for further details
# about the immersed setting), and the function
# $f:\Omega\to\mathbb{R}$ defined as $f(x,y,z) = \sin(x)\cos(y)\sin(z)$.
# Let us also consider a finite element space $V$ defined over the mesh
# $\mathcal{T}(\Omega)$, then, computing the $L^2$ projection
# of $f$ onto $V$ is equivalent to solving the following variational
# problem:
# $
# \begin{align}
#   \text{Find } u\in V \text{ such that } \forall v\in V:
#   \int_{\Omega} u v \text{d} x = \int_{\Omega} f v \text{d} x
# \end{align}
# $
#
# ## Implementation
#
# ### Modules import
# First we add the needed modules and functions:

# +

from qugar.utils import has_FEniCSx, has_PETSc

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

if not has_PETSc:
    raise ValueError("petsc4py installation not found is required.")

from pathlib import Path

from mpi4py import MPI

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.io
import numpy as np
import ufl
from dolfinx import default_real_type as dtype
from dolfinx.fem.petsc import LinearProblem

import qugar
import qugar.impl

# Importing qugar.dolfinx (here via form_custom) patches DOLFINx so the
# stock LinearProblem above assembles unfitted forms transparently.
from qugar.dolfinx import form_custom
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

# -


# ### Geometry and mesh

# We define the geometry to be considered: in this case a
# {py:class}`Schoen gyroid<qugar.impl.create_Schoen>` in 3D defined
# through an implicit function.
# The domain can be easily changed by just choosing a different implicit
# function among the available ones (see
# [Implicit functions demo](demo_impl_funcs.md) for some examples).

# +
impl_func = qugar.impl.create_Schoen([1.0, 1.0, 1.0])
# -

# We create an {py:class}`unfitted Cartesian mesh<qugar.mesh.UnfittedCartMesh>`
# (corresponding to $\mathcal{T}$) in which we embed the domain $\Omega$.
# This is a Cartesian mesh corresponding to the domain $[0,1]^3$ and
# with `n_cells` cells per direction.

# +
n_cells = 8

unf_mesh = create_unfitted_impl_Cartesian_mesh(
    MPI.COMM_WORLD, impl_func, n_cells, exclude_empty_cells=True, dtype=dtype
)

# -

# The option `exclude_empty_cells` (set to `True`) prevents the
# generation of empty cells in the mesh (those denoted as
# $\mathcal{T}_{\text{empty}}$ in [Divergence theorem demo](demo_div_thm.md)).
# This is useful to eliminate inactive degrees of freedom in the
# problem.

# ### Spaces and functions

# We create the function $\mathbf{f}$.

# +
x = ufl.SpatialCoordinate(unf_mesh)
f = ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.sin(x[2])
# -

# and the finite element space $V$ over the unfitted mesh
# (corresponding to the mesh $\mathcal{T}$) needed to define the test
# and trial functions $u$ and $v$.
# The finite element space is defined as a Lagrange space of the given
# `degree` (2 in this case).

# +
degree = 2
V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
# -

# ### Linear forms

# We define the variational problem to be solved, which is
# equivalent to the $L^2$ projection of $f$ onto $V$, and generate
# the corresponding bilinear and linear forms `a` and `L` using
# QUGaR's custom form functions.
# The number of quadrature points per cell (or integration cell
# in the case of cut cells) is set to `degree + 1` to
# prevent DOLFINx from using a higher-order quadrature rule
# (because of the trigonometric right-hand side expression).

# +
n_quad_pts = degree + 1
quad_degree = 2 * n_quad_pts + 1
F = (u - f) * v * ufl.dx(degree=quad_degree)
# -

# ### Linear system solution

# We solve the associated linear system
# $A\mathbf{u} = \mathbf{b}$, where $\mathbf{u}$ is the
# solution of the problem. The solution is stored in a
# a finite element function `uh` defined over the same finite
# element space $V$ as the trial functions.
#
# In this case we use a direct solver (Cholesky) to solve the
# linear system. However, due to the potentially ill-conditioning
# of the matrix, we use a (symmetric) Jacobi preconditioner.
# It is known that Jacobi preconditioners are not very effective
# for Lagrange elements, but still help.

# +
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "ksp_diagonal_scale": True,  # Jacobi
    # "ksp_diagonal_scale_fix": True, # transformsa back A an b after Jacobi
}

problem = LinearProblem(
    ufl.lhs(F),
    ufl.rhs(F),
    petsc_options=petsc_options,
    petsc_options_prefix="demo_l2_projection_",
)
problem.solve()

uh = problem.u
# -


# ### Error calculation

# The $L^2$ error of the projection is computed.
# In this case we slightly increase the number of quadrature points
# used to compute the error for achieving a higher accuracy.

# +
n_quad_pts = 2 * degree + 2
quad_degree = 2 * n_quad_pts + 1
error_form: qugar.dolfinx.CustomForm = form_custom(
    (uh - f) ** 2 * ufl.dx(degree=quad_degree), dtype=dtype
)
error_L2 = np.sqrt(
    unf_mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(error_form, coeffs=error_form.pack_coefficients()),
        op=MPI.SUM,
    )
)
if unf_mesh.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")
# -

# ### Visualization

# Finally, we visualize the obtained solution by generating
# an auxiliar mesh that reparameterizes the unfitted domain.
# This mesh approximates the domain using an arbitrary order fitted
# tessellation made of discontinuous (and possibly degenerated)
# Lagrange elements.
#
# An extra mesh corresponding to the wirebasket
# (the line intersections of the domain $\Omega$ with the Cartesian mesh
# $\mathcal{T}$ iso-planes) is also generated to improve the solution's visualization.

# +
rep_degree = 3
reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=rep_degree, levelset=False)
rep_mesh = reparam.create_mesh()
rep_mesh_wb = reparam.create_mesh(wirebasket=True)
# -

# In order to visualize the computed solution, we interpolate it into the
# generated meshes. For doing so we create new spaces associated to the
# meshes and interpolate the solution into them (using the
# `interpolate_nonmatching` DOLFINx's function). Note that even if the
# element types of the new spaces `Vrep` and `Vrep_wb` are continuous
# (`CG`), the underlying meshes are discontinuous, so they will be the
# generated functions.

# +
Vrep = dolfinx.fem.functionspace(rep_mesh, ("CG", rep_degree))
interp_data = qugar.reparam.create_interpolation_data(Vrep, V)
uh_rep = dolfinx.fem.Function(Vrep, dtype=dtype)
uh_rep.interpolate_nonmatching(uh, *interp_data)

Vrep_wb = dolfinx.fem.functionspace(rep_mesh_wb, ("CG", rep_degree))
interp_data_wb = qugar.reparam.create_interpolation_data(Vrep_wb, V)
uh_rep_wb = dolfinx.fem.Function(Vrep_wb, dtype=dtype)
uh_rep_wb.interpolate_nonmatching(uh, *interp_data_wb)
# -


# Both meshes are then exported to VTK files and can be visualized using
# [ParaView](https://www.paraview.org/). In the case in which
# `rep_degree > 1`, the parameter `Nonlinear Subdivision Level` value
# (under the advanced properties menu in ParaView) can be adjusted to
# generate higher-quality visualizations.

# +
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "demo_L2_projection"

with dolfinx.io.VTKFile(rep_mesh.comm, filename.with_suffix(".pvd"), "w") as vtk:
    vtk.write_function(uh_rep)
    vtk.write_function(uh_rep_wb)
# -


# Other DOLFINx file writers (as
# [VTXWriter](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html#dolfinx.io.VTXWriter)
# or [XDMFFile](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html#dolfinx.io.XDMFFile))
# could be also used, however, be aware that between `VTKFFile` and
# `XDMFFile`, the former is the [recommended option for high-degrees](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html#dolfinx.io.VTKFile).
