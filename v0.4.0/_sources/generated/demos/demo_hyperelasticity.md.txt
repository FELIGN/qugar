---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
  main_language: python
---

# Hyperelasticity problem

This demo is implemented in {download}`demo_hyperelasticity.py`. It
illustrates:

- How to solve a nonlinear hyperelasticity problem using QUGaR and FEniCSx.
- How to define and apply time-dependent Dirichlet boundary conditions.
- How to use a Newton solver for nonlinear problems.
- How to visualize the deformation of an unfitted domain over time.

## Problem definition

We consider large deformations of a hyperelastic material. The domain
$\Omega$ is a Fischer-Koch S surface embedded within a unit cube
$[0,1]^3$. The material behavior is described by a compressible
neo-Hookean model.
The goal is to find the displacement field $\mathbf{u}$ that satisfies
the equilibrium equations under applied boundary conditions.

The variational problem is derived from the principle of virtual work.
We seek $\mathbf{u} \in V$ such that:

$$
\int_{\Omega} \mathbf{P}(\mathbf{u}) : \nabla \mathbf{v} \, {\rm d} x = 0
\quad \forall \ \mathbf{v} \in V_0,
$$

where $\mathbf{P}$ is the first Piola-Kirchhoff stress tensor,
$\mathbf{v}$ is a test function from the space $V_0$ (functions
vanishing on the Dirichlet boundary), and $V$ is the function space
for the displacement.

The first Piola-Kirchhoff stress tensor $\mathbf{P}$ is related to the
strain energy density function $\psi$ by:

$$
\mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}
$$

where $\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}$ is the deformation
gradient.

For a compressible neo-Hookean material, the strain energy density
function $\psi$ is given by:

$$
\psi(\mathbf{F}) = \frac{\mu}{2} (I_C - 3) - \mu \ln(J) + \frac{\lambda}{2} (\ln(J))^2
$$

where $\mu$ and $\lambda$ are the Lamé parameters,
$I_C = \text{tr}(\mathbf{C})$ is the first invariant of the right
Cauchy-Green deformation tensor
$\mathbf{C} = \mathbf{F}^T \mathbf{F}$, and $J = \det(\mathbf{F})$ is
the determinant of the deformation gradient.

## Implementation

### Modules import
First, we import the necessary modules.

```python

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem as fem
import dolfinx.log as log
import dolfinx.mesh as mesh_dlf
import dolfinx.plot as plot_dlf
import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type as dtype
from dolfinx.fem.petsc import NonlinearProblem

import qugar
import qugar.dolfinx  # noqa: F401  (import patches DOLFINx for transparent unfitted assembly)
import qugar.impl
from qugar.mesh import create_unfitted_impl_Cartesian_mesh
from qugar.utils import has_FEniCSx, has_PETSc

# Check for required dependencies
if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

if not has_PETSc:
    raise ValueError("petsc4py installation not found is required.")
```

### Geometry and Mesh
We define the Fischer-Koch S geometry using an implicit function and
create an unfitted Cartesian mesh.

```python
# Define the implicit function for the Fischer-Koch S surface
impl_func = qugar.impl.create_Fischer_Koch_S([1.0, 1.0, 1.0])

# Set mesh parameters
n_cells = 8  # Number of cells per direction in the background grid
degree = 2  # Polynomial degree for the finite element space

# Create the unfitted mesh embedding the implicit geometry
unf_mesh = create_unfitted_impl_Cartesian_mesh(
    MPI.COMM_WORLD, impl_func, n_cells, exclude_empty_cells=True, dtype=dtype
)

# Get mesh dimension and facet dimension
dim = unf_mesh.topology.dim
fdim = dim - 1

# Define the vector function space V using Lagrange elements
V = fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
```

### Boundary Conditions
We define Dirichlet boundary conditions. The bottom face (z=0) is
fixed, and the top face (z=1) has a time-dependent vertical
displacement applied incrementally.

```python
# Locate facets on the bottom boundary (z=0)
bottom_facets = mesh_dlf.locate_entities_boundary(
    unf_mesh, fdim, marker=lambda x: np.isclose(x[dim - 1], 0.0)
)
# Locate facets on the top boundary (z=1)
top_facets = mesh_dlf.locate_entities_boundary(
    unf_mesh, fdim, marker=lambda x: np.isclose(x[dim - 1], 1.0)
)

# Mark bottom facets with 1 and top facets with 2
marked_facets = np.hstack([bottom_facets, top_facets])
marked_values = np.hstack([np.full_like(bottom_facets, 1), np.full_like(top_facets, 2)])
# Sort facets and create meshtags
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh_dlf.meshtags(
    unf_mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
)

# Define the boundary condition value for the bottom face (fixed)
u_bc_bottom = np.array((0,) * dim, dtype=dtype)
# Define a constant for the time-dependent boundary condition on the top face
u_bc_top = fem.Constant(unf_mesh, np.array((0,) * dim, dtype=dtype))

# Locate degrees of freedom (DOFs) corresponding to the bottom and top facets
bottom_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
top_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(2))

# Create Dirichlet boundary conditions
bc_bottom = fem.dirichletbc(u_bc_bottom, bottom_dofs, V)
bc_top = fem.dirichletbc(u_bc_top, top_dofs, V)
bcs = [bc_bottom, bc_top]  # List of boundary conditions
```

### Variational Formulation
We define the kinematic quantities, material parameters, strain energy density,
and the variational form for the hyperelasticity problem.

```python
# Define test and trial functions
v = ufl.TestFunction(V)  # Test function
u = fem.Function(V)  # Function for the displacement solution

# Define kinematic quantities
# Identity tensor
I = ufl.variable(ufl.Identity(dim))
# Deformation gradient: F = I + grad(u)
F = ufl.variable(I + ufl.grad(u))
# Right Cauchy-Green tensor: C = F^T * F
C = ufl.variable(F.T * F)
# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))  # First invariant of C
J = ufl.variable(ufl.det(F))  # Determinant of F (volume ratio)

# Define material parameters (Elasticity parameters)
E = dtype(1.0e4)  # Young's modulus
nu = dtype(0.3)  # Poisson's ratio
# Lamé parameters derived from E and nu
mu = fem.Constant(unf_mesh, E / (2 * (1 + nu)))
lmbda = fem.Constant(unf_mesh, E * nu / ((1 + nu) * (1 - 2 * nu)))

# Define the stored strain energy density function (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2

# Compute the first Piola-Kirchhoff stress tensor P = diff(psi, F)
P = ufl.diff(psi, F)

# Define integration measures with appropriate quadrature degree
n_quad_pts = degree + 1
quad_degree = 2 * n_quad_pts - 1
ds = ufl.ds(domain=unf_mesh, subdomain_data=facet_tag, degree=quad_degree)  # Boundary integral
dx = ufl.dx(domain=unf_mesh, degree=quad_degree)  # Domain integral

# Define the variational form F (Residual of the equilibrium equation)
# F = inner(grad(v), P) * dx = 0
F_form = ufl.inner(ufl.grad(v), P) * dx
```

### Nonlinear Solver Setup
We set up the stock DOLFINx ``NonlinearProblem`` (PETSc SNES); because
``unf_mesh`` is unfitted, qugar's assembly patches make it assemble the
residual and Jacobian with the runtime quadrature transparently.

```python
# Newton line-search via SNES with a direct (LU) linear solve.
petsc_options = {
    "snes_type": "newtonls",
    "snes_atol": 1e-8,  # Absolute tolerance
    "snes_rtol": 1e-8,  # Relative tolerance
    "ksp_type": "preonly",
    "pc_type": "lu",
}
sys = PETSc.Sys()  # type: ignore
# Select preferred factorization package (MUMPS or SuperLU_DIST)
use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
if sys.hasExternalPackage("mumps") and not use_superlu:
    petsc_options["pc_factor_mat_solver_type"] = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    petsc_options["pc_factor_mat_solver_type"] = "superlu_dist"

# Create the nonlinear problem
problem = NonlinearProblem(
    F_form,
    u,
    bcs=bcs,
    petsc_options=petsc_options,
    petsc_options_prefix="demo_hyperelasticity_",
)
```

### Visualization Setup
We set up PyVista for plotting and potentially creating a GIF of the deformation.
Reparameterization meshes are created for smoother visualization.

```python
# Initialize PyVista plotter, enabling off-screen rendering if needed.
# ``pyvista.start_xvfb`` is only useful on Linux for headless rendering,
# and recent pyvista releases drop it from the top-level namespace, so
# call it only when present.
if hasattr(pyvista, "start_xvfb"):
    pyvista.start_xvfb()
plotter = pyvista.Plotter()

# Attempt to set up GIF creation
try:
    plotter.open_gif("demo_hyperelasticity_deformation.gif", fps=3)
    create_GIF = True
except ImportError:
    print("imageio not installed, cannot create GIF.")
    create_GIF = False

# Create reparameterization meshes for visualization
rep_degree = 3  # Degree for the reparameterized mesh function space
reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=rep_degree, levelset=False)
rep_mesh = reparam.create_mesh()  # Standard reparameterized mesh
rep_mesh_wb = reparam.create_mesh(wirebasket=True)  # Wirebasket mesh for edges

# Create function spaces on the reparameterized meshes
Vr = fem.functionspace(rep_mesh, ("CG", rep_degree, (dim,)))  # Vector space on rep_mesh
Vr_wb = fem.functionspace(rep_mesh_wb, ("CG", rep_degree, (dim,)))  # Vector space on rep_mesh_wb

# Create interpolation data for transferring solution from original mesh to reparameterized meshes
interp_data = qugar.reparam.create_interpolation_data(Vr, V)
interp_data_wb = qugar.reparam.create_interpolation_data(Vr_wb, V)

# Create PyVista unstructured grids from the reparameterized function spaces
function_grid = pyvista.UnstructuredGrid(*plot_dlf.vtk_mesh(Vr))
function_grid_wb = pyvista.UnstructuredGrid(*plot_dlf.vtk_mesh(Vr_wb))

# Get number of points for array reshaping
n_points_rep = function_grid.n_points
n_points_rep_wb = function_grid_wb.n_points

# Create functions on reparameterized spaces to hold interpolated displacement
u_rep = fem.Function(Vr)
u_rep.interpolate_nonmatching(u, *interp_data)  # Interpolate initial state (zero displacement)

u_rep_wb = fem.Function(Vr_wb)
u_rep_wb.interpolate_nonmatching(u, *interp_data_wb)  # Interpolate initial state

# Add displacement vector data ("u") to the PyVista grids
values = np.zeros((n_points_rep, 3))
values[:, :dim] = u_rep.x.array.reshape(n_points_rep, dim)
function_grid["u"] = values
function_grid.set_active_vectors("u")

values_wb = np.zeros((n_points_rep_wb, 3))
values_wb[:, :dim] = u_rep_wb.x.array.reshape(n_points_rep_wb, dim)
function_grid_wb["u"] = values_wb

# Warp the visualization mesh by the initial displacement vector (zero)
warped = function_grid.warp_by_vector("u", factor=1)
warped.set_active_vectors("u")

warped_wb = function_grid_wb.warp_by_vector("u", factor=1)

# Add the initial meshes to the plotter
plotter.add_mesh(warped_wb, show_edges=True, lighting=False, clim=[0, 0.1])  # Wirebasket mesh
plotter.add_mesh(warped, show_edges=False, lighting=False, clim=[0, 0.1])  # Solid mesh

# Compute magnitude of displacement for visualization coloring
Vs = fem.functionspace(rep_mesh, ("Lagrange", rep_degree))  # Scalar space for magnitude
magnitude = fem.Function(Vs)
# Define expression for displacement magnitude: sqrt(u_x^2 + u_y^2 + u_z^2)
us_expr = ufl.sqrt(sum([u_rep[i] ** 2 for i in range(dim)]))
us = fem.Expression(us_expr, Vs.element.interpolation_points)
magnitude.interpolate(us)  # Interpolate initial magnitude (zero)
warped["mag"] = magnitude.x.array  # Add magnitude data to warped grid
```

### Time Stepping and Solving
We apply the displacement incrementally over several steps, solving
the nonlinear problem at each step. The visualization is updated at each step.

```python
# Set log level to INFO to see solver progress
log.set_log_level(log.LogLevel.INFO)

# Define total displacement and number of steps
disp = -0.25  # Total vertical displacement at the top
n_steps = 10  # Number of load steps

# Loop through time steps
for n in range(1, n_steps + 1):
    # Update the boundary condition value for the current step
    u_bc_top.value[dim - 1] = n * disp / n_steps

    print(f"--- Time step {n} ---")
    # Solve the nonlinear problem for the current displacement increment.
    # ``problem.solve()`` updates ``u`` in place and uses it as the initial
    # guess for the next load step.
    try:
        problem.solve()
        assert problem.solver.getConvergedReason() > 0
        u.x.scatter_forward()  # Update the solution vector across processes
        print(f"Converged in {problem.solver.getIterationNumber()} iterations.")
    except Exception as e:
        print(f"SNES solver did not converge. Exception: {e}")
        break  # Stop simulation if solver fails

    # Print current step information
    print(f"Displacement u_z = {u_bc_top.value[dim - 1]:.3f}")

    # Update visualization data
    # Interpolate solution to reparameterized mesh
    u_rep.interpolate_nonmatching(u, *interp_data)
    function_grid["u"][:, :dim] = u_rep.x.array.reshape(n_points_rep, dim)

    # Update displacement magnitude
    magnitude.interpolate(us)  # Re-interpolate magnitude based on updated u_rep
    warped.point_data["mag"][:] = magnitude.x.array
    warped.set_active_scalars("mag")  # Set magnitude as the active scalar for coloring

    # Warp the main visualization grid by the current displacement
    warped_n = function_grid.warp_by_vector(factor=1)
    warped.points[:, :] = warped_n.points  # Update point coordinates

    # Update wirebasket mesh visualization
    u_rep_wb.interpolate_nonmatching(u, *interp_data_wb)
    function_grid_wb["u"][:, :dim] = u_rep_wb.x.array.reshape(n_points_rep_wb, dim)
    warped_n_wb = function_grid_wb.warp_by_vector(factor=1)
    warped_wb.points[:, :] = warped_n_wb.points  # Update wirebasket point coordinates

    # Update plotter scalar bar range and write frame to GIF if enabled
    plotter.update_scalar_bar_range([0, np.max(magnitude.x.array)])
    if create_GIF:
        plotter.write_frame()

# Save the final plot as an image
plotter.save_graphic("demo_hyperelasticity_final.pdf")

# Display the final plot window
plotter.show()

# Close the plotter and GIF file
plotter.close()
```

```{figure} assets/demo_hyperelasticity_final.pdf
:name: fig-demo-hyperelasticity-deformation
Final deformation
```
