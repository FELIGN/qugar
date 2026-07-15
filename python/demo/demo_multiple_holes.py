# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # Domain with several circular holes

#
# This demo is implemented in {download}`demo_multiple_holes.py` and it illustrates:
#
# - How to describe a 2D domain with more than one circular hole.
# - How to combine several implicit functions through
#   {py:func}`create_functions_intersection<qugar.impl.create_functions_intersection>`.
# - How to visualize the resulting unfitted domain in PyVista.

# A single circular hole is obtained by taking the
# {py:func}`negative<qugar.impl.create_negative>` of a
# {py:func}`disk<qugar.impl.create_disk>`: the resulting function is negative *outside* the disk,
# so its (negative) domain is the plate minus the disk.
#
# To carve *several* disjoint holes we intersect the negative regions of several such functions
# with {py:func}`create_functions_intersection<qugar.impl.create_functions_intersection>`. The
# active domain is the region where *all* the functions are simultaneously negative, i.e., the
# points that lie outside *every* disk. Internally this relies on Algoim's multi-polynomial
# quadrature, so all the disks must be described through (Bezier) polynomials (`use_bzr=True`).
#
# ```{note}
# The holes must be pairwise disjoint. Overlapping disks would require a (non-polynomial) union
# operation, which is outside the scope of this polynomial-only construction.
# ```

# +
import qugar.utils

if not qugar.utils.has_FEniCSx:
    raise ValueError("FEniCSx installation is required.")

if not qugar.utils.has_PyVista:
    raise ValueError("PyVista installation is required.")
# -

# +
from mpi4py import MPI

import numpy as np
import pyvista as pv

import qugar.impl
import qugar.mesh
import qugar.plot
import qugar.reparam
from qugar.impl import create_disk, create_functions_intersection, create_negative

# -

# ## Geometry definition
#
# We define a plate $[0,1]^2$ with three disjoint circular holes, each given by its center and
# radius.

# +
dtype = np.float64

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
# We embed the domain in an unfitted Cartesian mesh over $[0,1]^2$.

# +
n_cells = [32, 32]
comm = MPI.COMM_WORLD

unf_mesh = qugar.mesh.create_unfitted_impl_Cartesian_mesh(
    comm, impl_func, n_cells, xmin=np.zeros(2, dtype), xmax=np.ones(2, dtype)
)
# -

# ## Visualization
#
# We reparameterize the (holed) domain interior and its levelset, and plot them with PyVista.

# +
reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=False)
reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

reparam_srf = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=True)
reparam_srf_pv = qugar.plot.reparam_mesh_to_PyVista(reparam_srf)

pl = pv.Plotter()
pl.add_mesh(reparam_pv.get("reparam"), color="white", show_edges=False)
pl.add_mesh(reparam_pv.get("wirebasket"), color="blue", line_width=2)
pl.add_mesh(reparam_srf_pv.get("reparam"), color="red", line_width=3)
pl.view_xy()
pl.show()
# -
