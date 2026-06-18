# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tensor-product and Cartesian meshes."""

from typing import Optional, cast

import qugar.utils

if not qugar.utils.has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import basix.ufl
import dolfinx.cpp as dlf_cpp
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl
from basix import CellType
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import coordinate_element as _coordinate_element

import qugar.cpp


def _merge_coincident_points_in_mesh(
    nodes: npt.NDArray[np.float32 | np.float64],
    conn: npt.NDArray[np.int64],
    tolerance: Optional[type[np.float32 | np.float64]] = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Finds conincident points from a mesh and merge them updating the
    cells' connectivity.

    This functionality requires `scipy` package to be installed.

    Args:
        nodes (npt.NDArray[np.float32 | np.float64]): Coordinates of the
            nodes stored in a 2D array. Rows correspond to the different
            points and columns to their coordinates.
        conn (npt.NDArray[np.int64]): Connectivity to update. It is a 2D
            array, where every row stores a list of nodes ids. The
            connectivity of each cells follows the DOLFINx convention.
            See https://github.com/FEniCS/basix/#supported-elements
        tolerance (Optional[type[np.float32 | np.float64]]): Absolute
            tolerance to be used in the comparison of coincident points.
            If not set, it is computed as 1.0e-5 times the maximum
            length along the Cartesian directions of the the bounding
            box enclosing the nodes.

    Raises:
        ValueError: If `scipy` is not found, an exception is thrown.

    Returns:
        npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int64],
        npt.NDArray[np.int64]:
        The first entry is the new array of node coordinates (with same
        format as `nodes`) containing unique (non-coincident) nodes.
        The second entry is the update connectivity (with same format as
        `conn`). The last entry is the map from old nodes to new ones.
    """

    try:
        import scipy as sp
    except ImportError as e:
        raise ValueError(e)

    kdtree = sp.spatial.KDTree(nodes)

    if tolerance is None:
        tolerance = cast(nodes.dtype.type, 1.0e-5 * (np.max(nodes) - np.min(nodes)))
    coincidences = kdtree.query_pairs(tolerance, output_type="ndarray")

    master = coincidences[:, 0]
    slave = coincidences[:, 1]

    n_pts = nodes.shape[0]
    slave_to_master = np.arange(n_pts, dtype=np.int64)
    slave_to_master[slave] = slave_to_master[master]

    convergence = False

    while not convergence:
        slave_to_master_prev = slave_to_master.copy()
        slave_to_master = slave_to_master[slave_to_master]
        convergence = np.all(slave_to_master == slave_to_master_prev)

    master_unique = np.unique(np.sort(slave_to_master))
    master_to_new_ordering = np.empty(n_pts, dtype=np.int64)
    master_to_new_ordering[master_unique] = np.arange(master_unique.size, dtype=np.int64)
    old_to_new = master_to_new_ordering[slave_to_master]

    new_nodes = nodes[master_unique]
    new_conn = np.empty_like(conn)
    for i, cell_conn in enumerate(conn):
        for j, node in enumerate(cell_conn):
            new_conn[i, j] = old_to_new[node]

    return new_nodes, new_conn, old_to_new


def _find_in_array(values_to_search: npt.NDArray, all_values: npt.NDArray) -> npt.NDArray[np.int64]:
    """Finds the index positions of the `values_to_search` referred to
    the array `all_values`.

    Args:
        values_to_search (npt.NDArray): Values to search.
        all_values (npt.NDArray): Array in which the values are sought.
            It must have the same type as `values_to_search`.

    Returns:
        npt.NDArray[np.int64]: Array indicating the position in
        `all_values` of each item in `values_to_search`. It has the same
        length as `values_to_search`. If a value is not present,
        it returns -1.
    """

    index = np.argsort(all_values)
    sorted_all_values = all_values[index]

    sorted_index = np.searchsorted(sorted_all_values, values_to_search)
    values_to_search_index = np.take(index, sorted_index, mode="clip").astype(np.int64)
    mask = all_values[values_to_search_index] != values_to_search

    return np.ma.array(values_to_search_index, mask=mask).filled(-1)


class Mesh(dolfinx.mesh.Mesh):
    """Enriched mesh data structure.

    This class derives from `dolfinx.mesh.Mesh`, easing the management of
    of some mesh quantities.

    For instance, it provides methods for easily quering the id of a cell in
    the reference mesh or its corresponding counterpart referred to a
    mesh created with DOLFINx.

    The mesh can be partitioned among different processes using a MPI
    communicator passed to the contructor. So, when queried about
    indices of vertices, facets, cells, etc., the returns are referred
    to the indices present in the current subdomain of the mesh.

    Note that after the mesh creation, DOLFINx renumbers and partitions
    the mesh, so, this numeration will be different. It is always
    possible to retrieve the original numbering using the maps
    `self.get_original_node_ids` or `self.get_original_cell_ids`.
    Or, from the original numbering used to create the mesh to the local
    DOLFINx numbering using `self.get_DOLFINx_local_node_ids` or
    `self.get_DOLFINx_local_cell_ids`, or the global with
    `self.get_DOLFINx_global_node_ids` or self.get_DOLFINx_global_cell_ids`,
    """

    def __init__(
        self,
        comm: MPI.Comm,
        nodes_coords: npt.NDArray[np.float32 | np.float64],
        conn: npt.NDArray[np.int64],
        cell_type: CellType,
        degree: int = 1,
        active_cells: Optional[npt.NDArray[np.int64]] = None,
        ghost_mode: GhostMode = GhostMode.none,
        merge_nodes: bool = False,
        merge_tol: Optional[type[np.float32 | np.float64]] = None,
    ) -> None:
        """Constructor.

        Args:
            comm (MPI.Comm): MPI communicator to be used for
                distributing the mesh.
            nodes_coords (npt.NDArray[np.float32 | np.float64]): Nodes
                coordinates. The rows correspond to the different nodes
                and the columns to the coordinates of each point.
            conn (npt.NDArray[np.int64]): Connectivity of the mesh.
            cell_type (CellType): Type of the mesh cell.
            degree (int): Degree of the mesh.
            active_cells (Optional[npt.NDArray[np.int64]]): Array of
                active cells. If not set, all the cells are considered
                active. The cells are referred to the original numbering
                used to create the mesh.
            ghost_mode (GhostMode, optional): Ghost mode used for mesh
                partitioning. Defaults to `none`.
            merge_nodes (bool, optional): If `True`, coincident nodes
                will be merged together into a single one. Otherwise,
                duplicated nodes will not be merged. Defaults to
                `False`.
            merge_tol (Optional[type[np.float32 | np.float64]]): Absolute
                tolerance to be used for seeking coincident nodes.
                If not set, and if `merge_nodes` is set to `True`,
                this tolerance will be automatically computed in the
                function `merge_coincident_points_in_mesh`.
        """

        self._degree = degree
        assert 1 <= self.degree, "Invalid degree."

        self._create_mesh(
            comm, nodes_coords, conn, cell_type, active_cells, ghost_mode, merge_nodes, merge_tol
        )

    @property
    def dtype(self) -> np.dtype[np.float32] | np.dtype[np.float64]:
        """Gets the type associated to the breaks.

        Returns:
            np.dtype[np.float32] | np.dtype[np.float64]: Type associated to the
            breaks.
        """
        return self.geometry.x.dtype

    @property
    def degree(self) -> int:
        """Gets the mesh' degree.

        Returns:
            int: Mesh's degree.
        """
        return self._degree

    @property
    def tdim(self) -> int:
        """Gets the topological dimension of the mesh.

        Returns:
            int: Mesh's topological dimension (1, 2, or 3).
        """
        return self.topology.dim

    @property
    def gdim(self) -> int:
        """Gets the geometrical dimension of the mesh.

        Returns:
            int: Mesh's geometrical dimension.
        """
        return self.geometry.dim

    @property
    def has_merged_nodes(self) -> bool:
        """Checks if the mesh has merged nodes.

        Returns:
            bool: Whether the mesh has merged nodes.
        """
        return self._merged_nodes_map.size > 0

    @property
    def has_inactive_cells(self) -> bool:
        """Checks if the mesh has inactive cells.

        Returns:
            bool: Whether the mesh has inactive cells.
        """
        return self._original_active_cells is not None

    @property
    def original_active_cells(self) -> npt.NDArray[np.int64]:
        """Gets the original active cells of the mesh.

        Returns:
            npt.NDArray[np.int64]: Original active cells.
        """
        if self._original_active_cells is None:
            raise ValueError("No original active cells.")
        return self._original_active_cells

    @property
    def merged_nodes_map(self) -> npt.NDArray[np.int64]:
        """Gets the map of original merged nodes: from original
        ids to unique ones.

        Returns:
            npt.NDArray[np.int64]: Map from old to new nodes.
        """
        return self._merged_nodes_map

    @property
    def num_global_cells(self) -> np.int64:
        """Gets the total (global) number of cells of the DOLFINx mesh.

        Returns:
            np.int64: Number of global cells of the DOLFINx mesh.
        """
        return np.int64(self.topology.index_map(self.tdim).size_global)

    @property
    def num_local_cells(self) -> np.int64:
        """Gets the local number of cells (associated to the current
        subdomain, i.e., MPI process).

        Returns:
            np.int64: Number of local cells.
        """
        return np.int64(self.topology.index_map(self.tdim).size_local)

    def _create_mesh(
        self,
        comm: MPI.Comm,
        nodes_coords: npt.NDArray[np.float32 | np.float64],
        conn: npt.NDArray[np.int64],
        cell_type: CellType,
        active_cells: Optional[npt.NDArray[np.int64]],
        ghost_mode: GhostMode,
        merge_nodes: bool,
        merge_tol: Optional[type[np.float32 | np.float64]],
    ) -> None:
        """Creates the base DOLFINx mesh.

        Args:
            comm (MPI.Comm): MPI communicator to be used for
                distributing the mesh.
            nodes_coords (npt.NDArray[np.float32 | np.float64]): Nodes
                coordinates. The rows correspond to the different nodes,
                and the columns to the coordinates of each point.
            conn (npt.NDArray[np.int64]): Connectivity of the mesh.
            cell_type (CellType): Type of the mesh cell.
            active_cells (Optional[npt.NDArray[np.int64]]): Array of
                active cells. If not set, all the cells are considered
                active. The cells are referred to the original numbering
                used to create the mesh.
            ghost_mode (GhostMode, optional): Ghost mode used for mesh
                partitioning.
            merge_nodes (bool, optional): If `True`, coincident nodes
                will be merged together into a single one. Otherwise,
                duplicated nodes will not be merged.
            merge_tol (Optional[type[np.float32 | np.float64]]): Absolute
                tolerance to be used for seeking coincident nodes.
                If not set, and if `merge_nodes` is set to `True`,
                this tolerance will be automatically computed in the
                function `merge_coincident_points_in_mesh`.
        """

        self._merged_nodes_map = np.empty(0, dtype=np.int64)

        nodes_coords = np.asarray(nodes_coords, order="C")
        if nodes_coords.ndim == 1:
            gdim = 1
        else:
            gdim = nodes_coords.shape[1]

        element = basix.ufl.element(
            "Lagrange", cell_type, self.degree, shape=(gdim,), dtype=nodes_coords.dtype
        )
        domain = ufl.Mesh(element)

        e_ufl = domain.ufl_coordinate_element()
        cmap = _coordinate_element(e_ufl.basix_element)
        # TODO: Resolve UFL vs Basix geometric dimension issue
        # assert domain.geometric_dimension() == gdim

        # TODO: to manage active cells for the case of
        # distributed meshes.

        if active_cells is not None and active_cells.size == conn.shape[0]:
            active_cells = None
        self._original_active_cells = active_cells

        if comm.rank == 0:
            if self._original_active_cells is not None:
                conn = conn[self._original_active_cells]

            if merge_nodes:
                nodes_coords, conn, self._merged_nodes_map = _merge_coincident_points_in_mesh(
                    nodes_coords, conn, merge_tol
                )
            conn = np.asarray(conn, dtype=np.int64, order="C")

        else:
            # TODO: this is a temporary hack. All the nodes and cells
            # are created in rank 0, and then distributed. Better
            # performance could be achieved by distributing nodes and
            # cells since their creation.
            # See https://jsdokken.com/dolfinx_docs/meshes.html#mpi-communication
            # for further details.
            assert nodes_coords.shape[0] == 0

            n_nodes_per_cell = cmap.dim
            conn = np.empty((0, n_nodes_per_cell), dtype=np.int64, order="C")

        # DOLFINx 0.11 added a trailing ``max_facet_to_cell_links`` argument to
        # ``create_mesh`` / ``create_cell_partitioner``. ``None`` keeps the
        # pre-0.11 behaviour (no cap on facet-to-cell links in the dual graph).
        max_facet_to_cell_links = None

        if comm.size > 1:
            partitioner = dlf_cpp.mesh.create_cell_partitioner(ghost_mode, max_facet_to_cell_links)
        else:
            partitioner = None

        msh_cpp = dlf_cpp.mesh.create_mesh(
            comm, conn, cmap._cpp_object, nodes_coords, partitioner, max_facet_to_cell_links
        )
        super().__init__(msh_cpp, domain)

    def get_DOLFINx_local_cell_ids(
        self,
        orig_cell_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int32]:
        """Transforms the given `orig_cell_ids` from the original numbering
        into the corresponding local ids of the underlying DOLFINx mesh.

        Args:
            orig_cell_ids (npt.NDArray[np.int64]): Cell indices to be transformed.
                They are the original indices used to create the mesh.

        Returns:
            npt.NDArray[np.int32]: Indices of the cells in
            the underlying DOLFINx mesh. These indices correspond to the
            local numbering associated to the current subdomain
            (process). Note that some indices may be set to -1 in the
            case the associated input original indices are not contained
            in the subdomain. The length of the output
            array is the same as the input.
        """

        if self._original_active_cells is not None:
            orig_cell_ids = _find_in_array(orig_cell_ids, self._original_active_cells)

        dlf_to_orig = self.topology.original_cell_index
        return _find_in_array(orig_cell_ids, dlf_to_orig).astype(np.int32)

    def get_DOLFINx_global_cell_ids(
        self,
        orig_cell_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Transforms the given local `orig_cell_ids` into the corresponding global ids
        of the underlying DOLFINx mesh.

        Args:
            orig_cell_ids (npt.NDArray[np.int64]): Cell indices to be transformed.
                They are the original indices used to create the mesh.

        Note:
            All the indices in `orig_cell_ids` must be contained in the
            current subdomain.

        Returns:
            npt.NDArray[np.int64]: Indices of the cells in the underlying DOLFINx mesh.
            These indices correspond to the global indices belonging to the
            current subdomain (process).
        """

        dlf_local_cell_ids = self.get_DOLFINx_local_cell_ids(orig_cell_ids)
        return self.get_DOLFINx_local_to_global_cell_ids(dlf_local_cell_ids)

    def get_DOLFINx_local_to_global_cell_ids(
        self,
        dlf_local_cell_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Transforms the given local `dlf_local_cell_ids` (DOLFINx) local
        cell ids into the corresponding global ones associated to the current process.

        Args:
            dlf_local_cell_ids (npt.NDArray[np.int32]): DOLFINx local cell indices
                to be transformed.

        Returns:
            npt.NDArray[np.int64]: Global indices of the DOLFINx cells.
        """

        index_map = self.topology.index_map(self.tdim)
        return index_map.local_to_global(dlf_local_cell_ids)

    def get_original_cell_ids(
        self,
        dlf_local_cell_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Transforms given `dlf_local_cell_ids` into
        the original numbering used to create the mesh.

        Args:
            dlf_local_cell_ids (npt.NDArray[np.int32]):
                Cell indices to be transformed. They arelocal
                indices referred to the underlying DOLFINx mesh.

        Returns:
            npt.NDArray[np.int64]: Ids of the cells in the original
            numbering. The length of the output array is the same as the input.
        """

        dlf_to_orig = self.topology.original_cell_index
        n_local_cells = dlf_to_orig.size
        assert np.all(dlf_local_cell_ids < n_local_cells) and np.all(dlf_local_cell_ids >= 0), (
            "Cells not contained in subdomain"
        )

        orig_cells = dlf_to_orig[dlf_local_cell_ids]
        if self._original_active_cells is not None:
            orig_cells = self._original_active_cells[orig_cells]
        return orig_cells

    def get_DOLFINx_local_node_ids(
        self,
        orig_node_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int32]:
        """Transforms the given `orig_node_ids` from the original
        numbering into the corresponding local ids of the underlying
        DOLFINx mesh.

        Args:
            orig_node_ids (npt.NDArray[np.int64]): Node indices to be transformed.
                They are referred to the original numbering used to create the mesh.

        Returns:
            npt.NDArray[np.int32]: Indices of the nodes in
            the underlying DOLFINx mesh. These indices correspond to the
            local numbering associated to the current subdomain
            (process). Note that some indices may be set to -1 in the
            case the input indices are original and some of them
            are not contained in the subdomain. The length of the output
            array is the same as the input.
        """

        dlf_to_orig = self.geometry.input_global_indices

        if self.has_merged_nodes:
            # Mapping "old" to "new" ids.
            orig_node_ids = self.merged_nodes_map[orig_node_ids]

        return _find_in_array(orig_node_ids, dlf_to_orig).astype(np.int32)

    def get_original_node_ids(
        self,
        dlf_local_node_ids: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Transforms given `dlf_local_node_ids` from the (local) DOLFINx
        mesh numbering to original numbering used to create the mesh.

        Args:
            dlf_local_node_ids (npt.NDArray[np.int32]):
                Node indices to be transformed. They are referred to the
                underlying (local) DOLFINx mesh.

        Returns:
            npt.NDArray[np.int64]: Ids of the nodes in the original
            numbering.
        """

        dlf_to_orig = self.geometry.input_global_indices
        n_local_nodes = dlf_to_orig.size
        assert np.all(dlf_local_node_ids < n_local_nodes) and np.all(dlf_local_node_ids >= 0), (
            "Nodes not contained in subdomain."
        )

        orig_nodes = dlf_to_orig[dlf_local_node_ids]
        if self.has_merged_nodes:
            return _find_in_array(orig_nodes, self.merged_nodes_map)
        else:
            return orig_nodes

    def get_cell_facets(
        self,
        dlf_local_cell_id: np.int32,
    ) -> npt.NDArray[np.int32]:
        """Extracts the facet ids of the given cell.

        Args:
            dlf_local_cell_id (np.int32): Id of the queried cell. It is a DOLFINx
                (local) cell id.

        Returns:
            npt.NDArray[np.int32]: Array of DOLFINx cell facets.
        """

        tdim = self.tdim

        self.topology.create_connectivity(tdim, tdim - 1)
        cell_to_facets = self.topology.connectivity(tdim, tdim - 1)

        return cell_to_facets.links(dlf_local_cell_id)

    def get_all_original_cell_ids(self) -> npt.NDArray[np.int64]:
        """
        Retrieves the ids of all the cells from the mesh using the original numbering.

        Returns:
            npt.nDArray[np.int64]: An array containing the original cell IDs.
        """
        all_local_dlf_cell_ids = np.arange(self.num_local_cells, dtype=np.int32)
        return self.get_original_cell_ids(all_local_dlf_cell_ids)
