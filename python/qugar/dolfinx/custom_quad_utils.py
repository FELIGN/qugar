# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tools for creating custom quadratures using generators and extra
information about the domain."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import basix
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.cpp.mesh import CellType


def _map_single_facet_points(
    points: npt.NDArray[np.float64], local_facet: int, cell: CellType
) -> npt.NDArray[np.float64]:
    """Maps the given points from the reference domain of a facet of the
    reference cell. I.e., the number of point coordinates is increased
    by one.

    Args:
        points (npt.NDArray[np.float64]): Points to be mapped. They must
            be defined in the reference domain of the facet.
        local_facet (int): Local id (referred to the cell) of the facet
            to which the `points` belong to.
        cell (CellType): Type of cell for performing the permutations.

    Returns:
        npt.NDArray[np.float64]: Mapped points.
    """

    basix_cell = getattr(basix.CellType, str(cell.name))
    geom: npt.NDArray[np.float64] = basix.geometry(basix_cell)  # type: ignore

    facet_vertices = [geom[i] for i in basix.topology(basix_cell)[-2][local_facet]]  # type: ignore
    tdim = facet_vertices[0].size

    assert (tdim - 1) == points.shape[1]

    origin = facet_vertices[0]

    n_points = points.shape[0]
    mapped_points = np.tile(origin, (n_points, 1))

    for i in range(tdim - 1):
        dir = facet_vertices[i + 1] - origin
        mapped_points += points[:, i].reshape(n_points, 1) * dir

    return mapped_points


def map_facets_points(
    points: npt.NDArray[np.float64],
    facets: npt.NDArray[np.int32],
    cell: CellType,
) -> npt.NDArray[np.float64]:
    """Maps the given points from the reference domain of the given
    facets to the reference cell. I.e., the number of point
    coordinates is increased by one.

    Args:
        points (npt.NDArray[np.float64]): Points to map. They must be
            defined in the reference domain of the facets. I.e., the
            points have as many coordinates as the facet domain
            (one less than the cell). The points for all the facets are
            stored consecutively.
        facets (npt.NDArray[np.int32]): Local facet ids (relative) to
            the reference cell. The length of the array must be the
            same as the number of points.
        cell (CellType): Type of cell in which the points are mapped.

    Returns:
        npt.NDArray[np.float64]: Mapped points for all the facets. These
        new points have one coordinate more than the input `points`.
    """

    n_points = points.shape[0]
    assert facets.size == n_points

    fdim = points.shape[1]
    tdim = fdim + 1
    new_points = np.empty((n_points, tdim), dtype=points.dtype)

    for facet in np.unique(facets):
        ids = np.where(facets == facet)[0]
        new_points[ids, :] = _map_single_facet_points(points[ids, :], facet, cell)

    return new_points


def _create_permutation_operators(
    cell_type: ufl.AbstractCell,
) -> list[npt.NDArray[np.float64]]:
    """Computes all the permutation operators for a given facet cell
    type.

    The generated operators cover all the possible permutations
    between two facets of a given cell type.

    Args:
        cell_type (ufl.AbstractCell): Facet cell type for which the
            permutations are generated. Note it is the type of the
            facet, not the cell. It can be ``interval``, ``triangle``,
            or ``quadrilateral``.

    Returns:
        list[npt.NDArray[np.float64]]: Generated permutations. The
        ordering in which these operators are stored matches the
        ordering of the permuted basis function values in ``FETable``
        and ``ffcx.ir.elementtables.get_ffcx_table_values``.
    """

    cell_name = cell_type.cellname
    assert cell_name in ["interval", "triangle", "quadrilateral"]

    dtype = np.float64
    if cell_name == "interval":
        return [
            np.eye(2, dtype=dtype),
            np.array([[-1.0, 1.0], [0.0, 1.0]], dtype=dtype),
        ]
    else:
        # Rotation
        if cell_name == "triangle":
            Rt = np.array([[0.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [0.0, 0.0, 1.0]], dtype=dtype)
        else:  # if cell_name == "quadrilateral":
            Rt = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=dtype)

        # Reflection
        Rf = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)

        RfRt = Rf @ Rt
        Rt2 = Rt @ Rt
        RfRt2 = Rf @ Rt2
        perms = [np.eye(3), Rf, Rt, RfRt, Rt2, RfRt2]

        if cell_type == "quadrilateral":
            perms.extend([Rt2 @ Rt, RfRt2 @ Rt])

        return perms


def _permute_points(
    points: npt.NDArray[np.float64],
    permutations: npt.NDArray[np.uint8],
    facet_cell: ufl.AbstractCell,
) -> npt.NDArray[np.float64]:
    """Permutes the given points.

    Args:
        points (npt.NDArray[np.float64]): Points to permute stored
            by rows.
        permutations (npt.NDArray[np.int32]): Indices of the permuations
            to apply. There are as many permutations as points. The
            values of this permutations matches the one used in
            ``ffcx.ir.elementtables.get_ffcx_table_values``.

        facet_cell (ufl.AbstractCell): Cell type of the facet in which
            the permutations are applied. It can be ``interval``,
            ``triangle``, or ``quadrilateral``.

    Returns:
        npt.NDArray[np.float64]: Permuted points.
    """

    assert points.shape[0] == permutations.size

    hom_points = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    perm_points = np.copy(points)

    Ps = _create_permutation_operators(facet_cell)

    for perm in np.unique(permutations):
        if perm == 0:
            continue
        ids = np.where(permutations == perm)[0]
        P = Ps[perm].T
        perm_points[ids, :] = hom_points[ids, :] @ P[:, :-1]

    return perm_points


def _get_facet_permutations(
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int32],
    facets: npt.NDArray[np.int32],
) -> npt.NDArray[np.uint8]:
    """Extracts the permutations associated to all the given facets.

    Args:
        mesh (dolfinx.mesh.Mesh): Mesh from which permutations are
            extracted.to which the custom quadrature
        cells (npt.NDArray[np.int32]): Cells of to which the facets
            belong.
        facets (npt.NDArray[np.int32]): Local facet ids (relative) to
            the reference cell for which the permutations are extracted.

    Returns:
        npt.NDArray[np.int32]: Permutations for all the facets.
    """

    assert cells.size == facets.size

    cell = ufl.Cell(mesh.topology.cell_type.name)
    face_tdim = cell.topological_dimension - 1
    n_facets_per_cell = cell.num_sub_entities(face_tdim)

    mesh.topology.create_entity_permutations()
    all_cells_perms = mesh.topology.get_facet_permutations()
    all_cells_perms = all_cells_perms.reshape(-1, n_facets_per_cell)

    return all_cells_perms[cells, facets]


def permute_facet_points(
    points: npt.NDArray[np.float64],
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int32],
    facets: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Permutes the points belonging to (interior or exterior) facet
    accoring to the facet permutations.

    Args:
        points (npt.NDArray[np.float64]): Points to permute stored
        mesh (dolfinx.mesh.Mesh): Mesh to which the cells belong.
        cells (npt.NDArray[np.int32]): Cells to which the facets
            belong.
        facets (npt.NDArray[np.int32]): Local facet ids (relative) to
            the reference cell for which the permutations are performed.

    Returns:
        npt.NDArray[np.float64]: Permuted points.
    """

    perms = _get_facet_permutations(mesh, cells, facets)

    cell = ufl.Cell(mesh.topology.cell_type.name)
    face_tdim = cell.topological_dimension - 1
    facet_cell = cell.sub_entity_types(face_tdim)[0]

    return _permute_points(points, perms, facet_cell)
