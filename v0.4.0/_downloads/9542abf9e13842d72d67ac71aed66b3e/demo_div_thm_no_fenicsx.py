# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # Divergence theorem (without FEniCSx)
#
# This demo is implemented in {download}`demo_div_thm_no_fenicsx.py` and
# solves the same problem as the [Divergence theorem demo](demo_div_thm.md)
# but without using [FEniCSx](https://fenicsproject.org) features.
# It uses only low-level interfaces to the C++ component of QUGaR.
#
# This demo covers:
# - How to create an unfitted domain described by an implicit function.
# - How to generate custom quadrature rules for cells, facets, and
# unfitted boundaries, and access the generated points, weights, and normals.
#
# all of them using only the C++ component of QUGaR.
#
# ## Theorem and problem definition
#
# This demo follows closely the problem in [Divergence theorem demo](demo_div_thm.md):
# to compute and verify the volumetric and surface integrals of the divergence theorem
# for an immersed domain.
# So, be sure to first read the `Theorem and problem definition` section of that demo
# for a detailed explanation of the problem and the basic concepts of unfitted domains.

# The problem setup is also the same as the one in the [Divergence theorem demo](demo_div_thm.md).
# Namely:
# - $\Omega^\ast = [0,1]^d$ (with $d=2,3$) (a square in 2D, a cube in 3D)
# - $\mathcal{T}$ is created with $16\times16$ quadrangles in 2D, and
#   and $16\times16\times16$ hexahedra in 3D.
# - $\Omega=\left\lbrace x\in\Omega^\ast: \lVert x-x_0\rVert < R\right\rbrace$ (a disk in 2D,
#  a sphere in 3D), with
#    - $x_0 = (-0.25, 0.15)$ in 2D, and $x_0 = (0.25, 0.35, 0.45)$ in 3D
#    - $R=0.6$ in 2D, and $R=0.7$ in 3D
# - $\mathbf{F} = \left(\sin(x),\cos(y)\right)$ in 2D, and
#   $\mathbf{F} = \left(\sin(x),\cos(y),\sin(z)\right)$ in 3D.

# ## Implementation
#
# First we import the needed modules and functions.

# +
from typing import Callable, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

import qugar
import qugar.cpp

# -

# and declare some handy type aliases.

# +
ImplicitFunc: TypeAlias = qugar.cpp.ImplicitFunc_2D | qugar.cpp.ImplicitFunc_3D
UnfittedDomain: TypeAlias = qugar.cpp.UnfittedImplDomain_2D | qugar.cpp.UnfittedImplDomain_3D
ReparamMesh: TypeAlias = (
    qugar.cpp.ReparamMesh_1_2
    | qugar.cpp.ReparamMesh_2_2
    | qugar.cpp.ReparamMesh_2_3
    | qugar.cpp.ReparamMesh_3_3
)
Func = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
DivFunc = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
# -

# We define the floating point type to be used in the computations.
# Be aware that the C++ component of QUGaR exclusively uses 64 bits floating point types (doubles).

dtype = np.float64

# We define the geometry to be considered: a disk in 2D and a sphere in 3D.
# Uncomment the one you want to use.

# +
# radius_2D = dtype(0.6)
# center_2D = np.array([-0.25, 0.15], dtype=dtype)
# impl_func = qugar.cpp.create_disk(center=center_2D, radius=radius_2D)
# name = "disk"

radius_3D = dtype(0.7)
center_3D = np.array([0.25, 0.35, 0.45], dtype=dtype)
impl_func = qugar.cpp.create_sphere(center=center_3D, radius=radius_3D)
name = "sphere"

print(f"{name.capitalize()} - divergence theorem verification")
# -

# and also the number of quadrature points per direction to be used in
# each integration.

n_quad_pts = 5

# We create a Cartesian grid (corresponding to $\mathcal{T}$) in which
# we will embed the domain $\Omega$.

# +
dim = impl_func.dim
n_cells = [16] * dim
cell_breaks = [np.linspace(0.0, 1.0, n_cells[dir] + 1, dtype=dtype) for dir in range(dim)]

grid = qugar.cpp.create_cart_grid(cell_breaks)
# -

# and then create an unfitted domain that stores the partition
# $\mathcal{T}=\mathcal{T}_{\text{cut}}\cup\mathcal{T}_{\text{full}}\cup\mathcal{T}_{\text{empty}}$.

unf_domain = qugar.cpp.create_unfitted_impl_domain(impl_func, grid)

# We create functions for evaluating the vector function $\mathbf{F}$ and $\text{div}\mathbf{F}$.


# +
def F_func(x: npt.NDArray[dtype]) -> npt.NDArray[dtype]:
    vals = np.empty_like(x)
    for i in range(x.shape[1]):
        vals[:, i] = np.sin(x[:, i]) if i % 2 == 0 else np.cos(x[:, i])
    return vals


def divF_func(x: npt.NDArray[dtype]) -> npt.NDArray[dtype]:
    vals = np.zeros(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[1]):
        vals += np.cos(x[:, i]) if i % 2 == 0 else -np.sin(x[:, i])
    return vals


# -


# ### Auxiliar functions.

# Before computing the integrals, we define some auxiliar functions to help us.
# Check their docstrings for more information.


# +
def scale_points_from_01(
    domain: npt.NDArray[dtype], points: npt.NDArray[dtype]
) -> npt.NDArray[dtype]:
    """
    Scales points from the unit interval [0, 1] to a specified domain.

    Args:
        domain (npt.NDArray[dtype]): A 2D array where each row represents the
            [min, max] range for each dimension.
        points (npt.NDArray[dtype]): A 2D array of points to be scaled, where
            each row is a point and each column corresponds to a dimension.

    Returns:
        npt.NDArray[dtype]: A 2D array of scaled points.
    """
    dim = points.shape[1]
    scaled_points = np.empty_like(points)
    for dir in range(dim):
        scaled_points[:, dir] = domain[dir, 0] + points[:, dir] * (domain[dir, 1] - domain[dir, 0])
    return scaled_points


# -


# +
def find_facets_on_boundary(
    unf_domain: UnfittedDomain,
    cells: npt.NDArray[np.int32],
    local_faces: npt.NDArray[np.int32],
    local_face_id: int,
) -> npt.NDArray[np.int32]:
    """
    Among the given cells (and associated faces), finds (and returns) the cells whose
    faces are equal to local_face_id.

    Args:
        unf_domain (UnfittedDomain): The unfitted implicit domain object containing the grid.
        cells (npt.NDArray[np.int32]): Array of cell indices.
        facets (npt.NDArray[np.int32]): Array of local face indices corresponding to the cells.
        local_face_id (int): The ID of the local face to find among the cells.

    Returns:
        npt.NDArray[np.int32]: Array of cell indices that have have an associated
        local face id equal to local_face_id.
    """
    cells = cells[np.where(local_faces == local_face_id)]

    grid = unf_domain.grid
    bound_cells = grid.get_boundary_cells(local_face_id)
    indices = np.where(np.isin(cells, bound_cells))[0]
    return cells[indices]


# -


# +
def find_full_facets_on_boundary(
    unf_domain: UnfittedDomain, local_face_id: int
) -> npt.NDArray[np.int32]:
    """
    Find full facets on the boundary of an unfitted implicit domain that have the
    given local facet_id.

    This function retrieves the full facets from the given unfitted implicit domain
    and identifies which of these facets are on the boundary based on the
    provided local face ID.

    Args:
        unf_domain (UnfittedDomain): The unfitted implicit domain object containing
            the grid data.
        local_face_id (int): The local ID of the face to check for boundary status.

    Returns:
        npt.NDArray[np.int32]: An array of cell IDs associated to the facets.
    """
    cells, facets = unf_domain.get_full_facets()
    return find_facets_on_boundary(unf_domain, cells, facets, local_face_id)


# -


# +
def find_cut_facets_on_boundary(
    unf_domain: UnfittedDomain, local_face_id: int
) -> npt.NDArray[np.int32]:
    """
    Find cut facets on the boundary of an unfitted implicit domain that have the
    given local local_face_id.

    This function retrieves the cut facets from the given unfitted implicit domain
    and identifies which of these facets are on the boundary based on the
    provided local face ID.

    Args:
        unf_domain (UnfittedDomain): The unfitted implicit domain object containing
            the grid data.
        local_face_id (int): The local ID of the face to check for boundary status.

    Returns:
        npt.NDArray[np.int32]: An array of cell IDs associated to the facets.
    """
    cells, facets = unf_domain.get_cut_facets()
    return find_facets_on_boundary(unf_domain, cells, facets, local_face_id)


# -


# +
def create_facet_quadrature(
    facet_points: npt.NDArray[dtype],
    facet_weights: npt.NDArray[dtype],
    local_face_id: int,
) -> Tuple[npt.NDArray[dtype], npt.NDArray[dtype], npt.NDArray[dtype]]:
    """
    Create quadrature points, weights, and normals for a given local face id.

    Extends the points from the facet to the higher-dimensional space by adding
    the constant coordinate of the face.
    It also generated the (constant) normal vectors for the quadrature points.

    Note:
        The weights are not modified.

    Args:
        facet_points (npt.NDArray[dtype]): Array of points on the facet.
        facet_weights (npt.NDArray[dtype]): Array of weights for the quadrature points.
        local_face_id (int): Identifier for the facet, used to determine
            the constant direction and side.

    Returns:
        Tuple[npt.NDArray[dtype], npt.NDArray[dtype], npt.NDArray[dtype]]:
            - points: Array of quadrature points in the higher-dimensional space.
            - facet_weights: Array of weights for the quadrature points.
            - normals: Array of normal vectors for the quadrature points.
    """
    facet_dim = facet_points.shape[1]
    dim = facet_dim + 1
    dtype = facet_points.dtype

    const_dir = local_face_id // 2
    side = local_face_id % 2

    points = np.zeros((facet_points.shape[0], dim), dtype=dtype)

    points[:, const_dir] = dtype.type(side)
    local_dir = 0
    for dir in range(dim):
        if dir != const_dir:
            points[:, dir] = facet_points[:, local_dir]
            local_dir += 1

    normals = np.zeros_like(points)
    normals[:, const_dir] = dtype.type(1.0) if side == 1 else dtype.type(-1.0)

    return points, facet_weights, normals


# -


# +
def create_full_facet_quadrature(
    dim: int, local_face_id: int, n_quad_pts: int
) -> Tuple[npt.NDArray[dtype], npt.NDArray[dtype], npt.NDArray[dtype]]:
    """
    Create a full facet quadrature for a given dimension and local face.

    Args:
        dim (int): The dimension of the space.
        local_face_id (int): The identifier of the local face.
        n_quad_pts (int): The number of quadrature points per direction.

    Returns:
        Tuple[npt.NDArray[dtype], npt.NDArray[dtype], npt.NDArray[dtype]]:
            - points: Array of quadrature points in the higher-dimensional space.
            - facet_weights: Array of weights for the quadrature points.
            - normals: Array of normal vectors for the quadrature points.
    """
    quad = qugar.cpp.create_Gauss_quad_01([n_quad_pts] * (dim - 1))
    return create_facet_quadrature(quad.points, quad.weights, local_face_id)


# -


# ### Computing the volumetric integral

# We compute now the volumetric integral for the cut and full cells.

# So, first, we create a custom quadrature for the cut cells.

cut_cells_quad = qugar.cpp.create_quadrature(unf_domain, unf_domain.get_cut_cells(), n_quad_pts)

# and for the full cells we use the standard quadrature rule.

quad_01 = qugar.cpp.create_Gauss_quad_01([n_quad_pts] * dim)


# We define a function for computing the contribution of a single
# cell, either cut or full.


def compute_cell_integr(points_01, weights_01, cell_id):
    domain = grid.get_cell_domain(cell_id)
    weights = weights_01 * domain.volume
    vals = divF_func(scale_points_from_01(domain.as_array(), points_01))
    return np.dot(vals, weights)


# Loop over the full cells and compute their contributions

vol_intgr = dtype(0.0)
for cell_id in unf_domain.get_full_cells():
    vol_intgr += compute_cell_integr(quad_01.points, quad_01.weights, cell_id)

# and the same for the cut cells (retrieving the associated custom quadrature points
# and weights for every cell)

pt_id = 0
for cell_id, n_pts in zip(cut_cells_quad.cells, cut_cells_quad.n_pts_per_entity):
    pt_id_1 = pt_id + n_pts

    weights = cut_cells_quad.weights[pt_id:pt_id_1]
    points = cut_cells_quad.points[pt_id:pt_id_1]
    vol_intgr += compute_cell_integr(points, weights, cell_id)

    pt_id = pt_id_1


# ### Computing the surface integral

# We compute now the surface integral for the unfitted boundaries (for $\Gamma_{\text{unf}}$)
# and the cut (for $\mathcal{F}_{\text{cut}}$) and full facets (for $\mathcal{F}_{\text{full}}$).

# We define a function for computing the contribution of a single
# cell (containing an unfitted boundary) or (cut or full) facet.


def compute_srf_integral(points_01, weights_01, normals, cell_id):
    cell_domain = grid.get_cell_domain(cell_id)
    vals = F_func(scale_points_from_01(cell_domain.as_array(), points_01))

    mapped_normals = np.empty_like(normals)
    for dir in range(dim):
        mapped_normals[:, dir] = normals[:, dir] / cell_domain.length(dir)
    scaled_weights = weights_01 * cell_domain.volume * np.linalg.norm(mapped_normals, axis=1)

    return np.dot(np.sum(vals * normals, axis=1), scaled_weights)


#

# We first compute the contribution of the external facets (i.e., $\Gamma_{\text{ext}}$).
# For that purpose we iterate along the `dim * 2` faces of the
# hypercube $\Omega^\ast$ and compute first the contribution of $\mathcal{F}_{\text{full}}$

# +
srf_intgr = dtype(0.0)

for face_id in range(dim * 2):
    full_facets = find_full_facets_on_boundary(unf_domain, face_id)
    points, weights, normals = create_full_facet_quadrature(dim, face_id, n_quad_pts)

    for cell_id in full_facets:
        srf_intgr += compute_srf_integral(points, weights, normals, cell_id)
# -

# and then for $\mathcal{F}_{\text{cut}}$ (that requires a custom quadrature).

# +
for face_id in range(dim * 2):
    cells = find_cut_facets_on_boundary(unf_domain, face_id)
    facets = np.full_like(cells, face_id)
    facet_quad = qugar.cpp.create_facets_quadrature_exterior_integral(
        unf_domain, cells, facets, n_quad_pts
    )

    pt_id = 0
    for cell_id, n_pts in zip(facet_quad.cells, facet_quad.n_pts_per_entity):
        pt_id_1 = pt_id + n_pts

        facet_points = facet_quad.points[pt_id:pt_id_1]
        facet_weights = facet_quad.weights[pt_id:pt_id_1]
        points, weights, normals = create_facet_quadrature(facet_points, facet_weights, face_id)
        srf_intgr += compute_srf_integral(points, weights, normals, cell_id)

        pt_id = pt_id_1
# -

# Finally, we compute the contribution of the unfitted boundaries (for $\Gamma_{\text{unf}}$).

# We compute the needed custom quadrature.

include_facet_unf_bry = True
exclude_ext_bdry = True
unf_bdry_quad = qugar.cpp.create_unfitted_bound_quadrature(
    unf_domain,
    unf_domain.get_cut_cells(),
    n_quad_pts,
    include_facet_unf_bry,
    exclude_ext_bdry,
)

# and then iterate along the cells that contain unfitted boundaries
# (retrieving the associated custom quadrature points, weights, and
# normals for every cell).

pt_id = 0
for cell_id, n_pts in zip(unf_bdry_quad.cells, unf_bdry_quad.n_pts_per_entity):
    pt_id_1 = pt_id + n_pts

    weights = unf_bdry_quad.weights[pt_id:pt_id_1]
    normals = unf_bdry_quad.normals[pt_id:pt_id_1]
    points = unf_bdry_quad.points[pt_id:pt_id_1]
    srf_intgr += compute_srf_integral(points, weights, normals, cell_id)

    pt_id = pt_id_1

# Finally, we compare both integrals

print(f"  - Volumetric integral: {vol_intgr}")
print(f"  - Surface integral: {srf_intgr}")
