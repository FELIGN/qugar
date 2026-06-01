# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # PyVista visualization capabilities.

#
# This demo is implemented in {download}`demo_plot.py` and it
# illustrates:
#
# - How to create an unfitted implicit mesh using the library of available implicit functions.
# - The available PyVista based visualization tools.

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
from typing import cast

from mpi4py import MPI

import numpy as np
import pyvista as pv

import qugar
import qugar.impl
import qugar.mesh
import qugar.plot
import qugar.reparam

# -

# ## Domain definition

# We create an unfitted domain using a {py:class}`3D cylinder<qugar.impl.create_cylinder>`
# defined by a point on the axis, the direction of the revolution axis, and the radius.

func = qugar.impl.create_cylinder(
    radius=0.4, origin=np.array([0.55, 0.45, 0.47]), axis=np.array([1.0, 0.9, -0.95]), use_bzr=True
)

# and then generate the unfitted Cartesian mesh over a hypercube $[0,1]^3$
# with 5 cells per direction.

# +
dim = func.dim
n_cells = [5] * dim

comm = MPI.COMM_WORLD

unf_mesh = qugar.mesh.create_unfitted_impl_Cartesian_mesh(
    comm, func, n_cells, xmin=np.zeros(dim), xmax=np.ones(dim)
)
# -

# ## Visualization

# We are going to visualize the quadrature for the unfitted domain, separating the
# quadrature for the cut cells, unfitted boundaries, and cut facets.

# First we create PyVista objects for the quadrature,

# +
quad = qugar.plot.quadrature_to_PyVista(unf_mesh, n_pts_dir=3)

quad_cells = cast(pv.UnstructuredGrid, quad.get("Cut cells quadrature"))
quad_unf_bdry = cast(pv.UnstructuredGrid, quad.get("Unfitted boundary quadrature"))
quad_facets = cast(pv.UnstructuredGrid, quad.get("Cut/unfitted facets quadrature"))
# -

# cut and full cells and cut facets of the unfitted domain,

# +
cut_cells = qugar.plot.unfitted_domain_to_PyVista(unf_mesh, cut=True, full=False, empty=False)
full_cells = qugar.plot.unfitted_domain_to_PyVista(unf_mesh, cut=False, full=True, empty=False)

cut_facets = qugar.plot.unfitted_domain_facets_to_PyVista(
    unf_mesh, cut=True, full=False, empty=False
)
# -

# and for the reparameterization of the domain's interior and its levelset boundary

# +
reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=False)
reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

reparam_srf = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=True)
reparam_srf_pv = qugar.plot.reparam_mesh_to_PyVista(reparam_srf)
# -

# We are going to plot the different quadrature points and the reparameterization of the domain's
# in different subplots.

# - First, we plot the reparameterization of the domain, splitting the interior and the
# wirebasket (the lines separating the reparameterization of each cell).
# We superimpose the cut and full cells.

pl = pv.Plotter(shape=(2, 2))
pl.subplot(0, 0)
pl.add_title("Reparemeterization", font_size=12)
pl.add_mesh(reparam_pv.get("reparam"), color="white")
pl.add_mesh(reparam_pv.get("wirebasket"), color="red", line_width=2)
pl.add_mesh(cut_cells, color="blue", style="wireframe")
pl.add_mesh(full_cells, color="blue", style="wireframe")


# - Second, we plot the quadrature of the cut cells, together with the
# cut and full cells, and a translucent reparameterization of the domain.

pl.subplot(0, 1)
pl.add_title("Cut cells quadrature", font_size=12)
pl.add_mesh(reparam_pv, opacity=0.25, color="white")
pl.add_mesh(quad_cells, point_size=4, render_points_as_spheres=True)
pl.add_mesh(cut_cells, color="blue", style="wireframe")

# - Third, we plot the quadrature for the unfitted boundary as well
# as the normal vectors at those points, together with the
# cut cells, and a translucent reparameterization of the domains unfitted boundary.

pl.subplot(1, 0)
pl.add_title("Unfitted boundary quadrature", font_size=12)
pl.add_mesh(quad_unf_bdry, point_size=4, render_points_as_spheres=True)
pl.add_mesh(cut_cells, color="blue", style="wireframe")
glyphs = quad_unf_bdry.glyph(
    orient=True, scale=False, factor=0.05, geom=pv.Arrow(), color_mode="scalar"
)
pl.add_mesh(glyphs)
pl.add_mesh(reparam_srf_pv.get("reparam"), color="white")
pl.add_mesh(reparam_srf_pv.get("wirebasket"), color="blue", line_width=2)

# - Finally, we plot the quadrature for the cut facets together with the
# cut facets themselves.

pl.subplot(1, 1)
pl.add_title("Cut facets quadrature", font_size=12)
pl.add_mesh(quad_facets, point_size=4, render_points_as_spheres=True)
pl.add_mesh(cut_facets, color="gray", opacity=0.1)
pl.add_mesh(cut_cells, color="blue", style="wireframe")
pl.show()
