# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # Creation of unfitted implicit domains

#
# This demo is implemented in {download}`demo_impl_funcs.py` and it
# illustrates:
#
# - The library of available implicit functions for defining domains.
# - How to create an unfitted mesh described by an implicit function.
# - How to visualize an unfitted domain in PyVista by creating a reparameterization.

# This demo requires the [FEniCSx](https://fenicsproject.org) and [PyVista](https://pyvista.org)
# capabilities of QUGaR.

# ## Geometry definition
#
# First we check that FEniCSx and PyVista installations are available.

# +
import qugar.utils

if not qugar.utils.has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

if not qugar.utils.has_PyVista:
    raise ValueError("PyVista installation not found is required.")
# -

# Then the modules and functions to be used are imported:

# +

from mpi4py import MPI

import numpy as np
import pyvista as pv

import qugar
import qugar.cpp
import qugar.impl
import qugar.mesh
import qugar.plot
import qugar.reparam

# -

# We have to choose a function for implicitly defining the domain
# among the ones available.

# ### Primitives
# QUGaR provides a basic collection of 2D and 3D primitive geometries for defining
# implicit domains, namely:

# - A {py:class}`2D disk<qugar.impl.create_disk>` defined by its center and radius.

func_disk = qugar.impl.create_disk(radius=0.8, center=np.array([0.51, 0.45]), use_bzr=True)

# - A {py:class}`3D sphere<qugar.impl.create_sphere>` defined by its center and radius.

func_sphere = qugar.impl.create_sphere(radius=0.8, center=np.array([0.5, 0.45, 0.35]), use_bzr=True)

# - A {py:class}`2D half-space<qugar.impl.create_line>` defined as the domain
# on one side of an infinite line.

func_line = qugar.impl.create_line(
    origin=np.array([0.5, 0.5]), normal=np.array([1.0, 0.2]), use_bzr=True
)

# - A {py:class}`3D half-space<qugar.impl.create_plane>` defined as the domain
# on one side of an infinite plane.

func_plane = qugar.impl.create_plane(
    origin=np.array([0.3, 0.47, 0.27]), normal=np.array([1.0, 0.3, -1.0]), use_bzr=True
)

# - A {py:class}`3D cylinder<qugar.impl.create_cylinder>` defined by a point on the axis,
# the direction of the revolution axis, and the radius.

func_cylinder = qugar.impl.create_cylinder(
    radius=0.4, origin=np.array([0.55, 0.45, 0.47]), axis=np.array([1.0, 0.9, -0.95]), use_bzr=True
)


# - A {py:class}`2D annulus<qugar.impl.create_annulus>` defined by its center and inner and outer
# radii.

func_annulus = qugar.impl.create_annulus(
    inner_radius=0.2, outer_radius=0.75, center=np.array([0.55, 0.47]), use_bzr=True
)

# - A {py:class}`2D ellipse<qugar.impl.create_ellipse>` defined by a 2D reference system
# (described through an origin and $x$-axis) and the length of the semi-axes along the
# reference system directions.

ref_system = qugar.cpp.create_ref_system(origin=np.array([0.5, 0.5]), axis=np.array([1.0, 1.0]))
func_ellipse = qugar.impl.create_ellipse(
    semi_axes=np.array([0.7, 0.5]), ref_system=ref_system, use_bzr=True
)

# - A {py:class}`3D ellipsoid<qugar.impl.create_ellipsoid>` defined by a 3D reference system
# (described through an origin and $x$- and $y$-axes) and the length of the semi-axes along the
# reference system directions.

ref_system = qugar.cpp.create_ref_system(
    origin=np.array([0.5, 0.5, 0.5]),
    axis_x=np.array([1.0, 1.0, 1.0]),
    axis_y=np.array([-1.0, 1.0, 1.0]),
)
func_ellipsoid = qugar.impl.create_ellipsoid(
    semi_axes=np.array([0.7, 0.45, 0.52]), ref_system=ref_system, use_bzr=True
)

# - A {py:class}`3D torus<qugar.impl.create_torus>` defined by its major and minor radii,
# its center, and the direction of the revolution axis for the major radius.

func_torus = qugar.impl.create_torus(
    major_radius=0.77,
    minor_radius=0.35,
    center=np.array([0.55, 0.47, 0.51]),
    axis=np.array([1.0, 0.9, 0.8]),
    use_bzr=True,
)

# - A 2D or 3D {py:class}`constant function<qugar.impl.create_constant>` defined by
# a constant value and a dimension.

func_constant = qugar.impl.create_constant(value=-0.5, dim=3, use_bzr=True)

# - A 2D or 3D {py:class}`dim-linear function<qugar.impl.create_dim_linear>` that defines
# a bi- or tri-linear function through its values in the 4 (respec. 8) vertices
# of the unit square (resp. cube)

func_dimlinear = qugar.impl.create_dim_linear(coefs=[-0.5, 0.5, 0.5, -0.7])

# Internally, (almost all) these functions can be represented through (Bezier) polynomials
# or analytical $C^1(\mathbb{R}^d)$ functions by propertly setting the argument `use_bzr`
# to `True` or `False`, respectively.
# This will setup the type of algorithm to be used in the generation of quadratures and
# reparameterizations.
# Check [Algoim library](https://algoim.github.io) for more details.


# ### Triply periodic minimal surface (TPMS)

# QUGaR also defines a library of triply periodic minimal surfaces (TPMS) that can be used
# to generate more complex implicit domains. The available TPMSs are:
# - [Schoen gyroid](https://en.wikipedia.org/wiki/Gyroid), defined as:
#
# $$
# \begin{align}
#   \phi(x,y,z) = \sin(\alpha x) \cos(\beta y) + \sin(\beta y) \cos(\gamma z)
# + \sin(\gamma z) \cos(\alpha x)
# \end{align}
# $$

func_schoen = qugar.impl.create_Schoen(periods=[2, 2, 2])

# - [Schoen F-RD](https://minimalsurfaces.blog/home/repository/triply-periodic/schoen-f-rd/),
# defined as:
#
# $$
# \begin{align}
#   \phi(x,y,z) &=   4 \cos(\alpha x) \cos(\beta y) \cos(\gamma z)
#         - \cos(2\alpha x) \cos(2\beta y)\\ &- \cos(2\beta y) \cos(2\gamma z)
#         - \cos(\gamma z) \cos(2\alpha x)
# \end{align}
# $$

func_schoeniwp = qugar.impl.create_Schoen_IWP(periods=[2, 2, 2])

# - [Schoen I-WP](https://minimalsurfaces.blog/home/repository/triply-periodic/schoen-i-wp/),
# defined as:
#
# $$
# \begin{align}
#   \phi(x,y,z) &= 2 \left(\cos(\alpha x) \cos(\beta y) + \cos(\beta y) \cos(\gamma z)
#              + \cos(\gamma z) \cos(\alpha x)\right)\\
#              &- \cos(2\alpha  x) - \cos(2\beta  y) - \cos(2\gamma z))
# \end{align}
# $$

func_schoenfrd = qugar.impl.create_Schoen_FRD(periods=[2, 2, 2])

# - [Fischer-Koch S](http://kenbrakke.com/evolver/examples/periodic/periodic.html#fishers),
# defined as:
#
# $$
# \begin{align}
#   \phi(x,y,z) &= \cos(2\alpha x) \sin(\beta y) \cos(\gamma z)
#         + \cos(\alpha x) \cos(2\beta y) \sin(\gamma z)\\
#         &+ \sin(\alpha x) \cos(\beta y) \cos(2\gamma z)
# \end{align}
# $$

func_fischerkochs = qugar.impl.create_Fischer_Koch_S(periods=[2, 2, 2])

# - [Schwarz D (Diamond)](https://en.wikipedia.org/wiki/Schwarz_minimal_surface#Schwarz_D_(%22Diamond%22)),
# defined as:
#
# $$
# \begin{align}
#   \phi(x,y,z) = \cos(\alpha x) \cos(\beta y) \cos(\gamma z)
# - \sin(\alpha x) \sin(\beta y) \sin(\gamma z)
# \end{align}
# $$

func_schwarzd = qugar.impl.create_Schwarz_Diamond(periods=[2, 2, 2])

# - [Schwarz P (Primitive)](https://en.wikipedia.org/wiki/Schwarz_minimal_surface#Schwarz_P_(%22Primitive%22)),
# defined as:
#
# $$
# \begin{align}
#   \phi(x,y,z) = \cos(\alpha x) + \cos(\beta y) + \cos(\gamma z)
# \end{align}
# $$

func_schwarzp = qugar.impl.create_Schwarz_Primitive(periods=[2, 2, 2])

#
# where $\alpha=2\pi m$, $\beta=2\pi n$, and $\gamma=2\pi p$, and $m$, $n$, and $p$
# the periods along each parametric direction. Given one of the functions above
# $\phi:\mathbb{R}^3\to\mathbb{R}$, the implicit domain is defined as
# $\Omega = \{\mathbf{x}\in\mathbb{R}^3\,|\,\phi(\mathbf{x})\leq 0\}$.
#
# 2D versions can be generated by passing only 2 periods instead of 3.
# In that case, the mathematical expressionss above hold, but the coordinate $z$ is assumed to be
# $z=0$.
#
# Note that TPMSs can not be represented through Bezier polynomials.

# ### Other functions
# In addition to the primitives and TPMSs, QUGaR provides other implicit functions that
# operate as modifiers of already existing functions, as:
# - {py:class}`Function negative<qugar.impl.create_negative>`: changes the sign of a given function.
# - {py:class}`Functions addition<qugar.impl.create_functions_addition>`:
# adds two given functions.
# - {py:class}`Functions subtraction<qugar.impl.create_functions_subtraction>`:
# subtracts two given functions.
# - {py:class}`Affine transformation<qugar.impl.create_affinely_transformed_function>`:
# applies an affine transformation to a given function.

# ### Future functions
# QUGaR is continuously updated with new functions and features. In the near/mid future,
# support will be provided for Bezier functions explicitly defined through its control points
# and for B-spline functions.

# ## Generating the unfitted mesh

# First we choose a function among the ones defined above.

func = func_schoen

# and then generate the unfitted Cartesian mesh over a hypercube $[0,1]^d$
# with 8 cells per direction, where $d=2$ or $d=3$.

dim = func.dim
n_cells = [8] * dim
comm = MPI.COMM_WORLD

unf_mesh = qugar.mesh.create_unfitted_impl_Cartesian_mesh(
    comm, func, n_cells, xmin=np.zeros(dim), xmax=np.ones(dim)
)

# ## Visualization

# Finally, we create a visualization of the unfitted mesh's interior a levelset
# through a parameterization.

# +
reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=False)
reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

reparam_srf = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=True)
reparam_srf_pv = qugar.plot.reparam_mesh_to_PyVista(reparam_srf)

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(reparam_pv.get("reparam"), color="white")
pl.add_mesh(reparam_pv.get("wirebasket"), color="blue", line_width=2)

pl.subplot(0, 1)
pl.add_mesh(reparam_srf_pv.get("reparam"), color="white")
pl.add_mesh(reparam_srf_pv.get("wirebasket"), color="blue", line_width=2)

pl.show()
# -
