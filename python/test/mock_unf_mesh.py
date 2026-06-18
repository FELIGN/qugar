# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Class for generating mock custom quadrature for testing purposes."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import Optional

import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from ffcx.ir.representationutils import create_quadrature_points_and_weights

from qugar.mesh.mesh_facets import (
    MeshFacets,
    create_exterior_mesh_facets,
    create_interior_mesh_facets,
)
from qugar.mesh.unfitted_domain_abc import UnfittedDomainABC
from qugar.quad import CustomQuad, CustomQuadFacet, CustomQuadUnfBoundary


class MockUnfittedMesh(dolfinx.mesh.Mesh, UnfittedDomainABC):
    """Class for generating mock unfitted mesh for testing purposes.

    It generates custom quadratures for a few cells (number specified
    in the constructor) and standard quadratures for the rest. The
    custom quadratures are just the same as the standard quadrature but
    repeated several times in the cells, and scaling the weights
    inversely proportional to the number of times the quadrature is
    repeated in a cell.

    Using these custom quadratures the same results should be obtained
    as with the standard quadratures, what makes this mock quadrature
    interesting for testing purposes.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, nnz: float = 0, max_quad_sets: int = 3) -> None:
        """Initializes.

        Args:
            mesh (dolfinx.mesh.Mesh): Mesh for which the quadratures are
                generated.
            nnz (int, optional): Ratio of entities with custom
                quadratures respect to the total number of entities.
                It must be a value in the range [0.0, 1.0].
            max_quad_sets (int, optional): Maximum number of repetitions
                of the standard quadrature in each custom cell. For each
                custom entity a random number is generated between 1 and
                `max_quad_sets`. Defaults to 3.
        """

        assert 0.0 <= nnz <= 1.0
        assert max_quad_sets > 0

        self._mesh = mesh
        self._nnz = nnz
        self._max_quad_sets = max_quad_sets

        self._compute_custom_cells()
        self._compute_custom_facets()

        dolfinx.mesh.Mesh.__init__(self, mesh._cpp_object, mesh._ufl_domain)
        UnfittedDomainABC.__init__(self, mesh)

    def _get_cells_ids(self) -> npt.NDArray[np.int32]:
        """Gets the ids of all the cells in the mesh.

        Returns:
            npt.NDArray[np.int32]: Ids of all the cells in the mesh.
        """
        n_cells = self._mesh.geometry.dofmaps[0].shape[0]
        return np.arange(n_cells, dtype=np.int32)

    def _extract_custom_entities(
        self,
        n_entities: int,
    ) -> npt.NDArray[np.int32]:
        """Given a number of entities `n_entities`, extracts a random subset
        of the ones for which custom quadratures are generated. The
        number of custom cells is built following the ratio `self._nnz`.

        Despite the random generation, same result is produced among
        different Python sessions.

        The entities for which custom quadratures are generated is
        defined through the number of quadrature sets per custom cell
        (in a pseudo-random way) that is a value between 0 and
        `self._max_quad_sets`. If 0, no custom quadrature is generated
        and it is considered a full cell. If greater than 0, it is
        considered a custom cell. No empty cells are considered.

        Args:
            n_entities (int): Number of entities to be considered
                for custom quadratures.

        Returns:
            npt.NDArray[np.int32]: Nmber of custom quadrature sets for
            every entity.
        """

        n_quad_sets = np.arange(n_entities, dtype=np.int32) % self._max_quad_sets + 2

        n_custom_entities = int(np.ceil(n_entities * self._nnz))
        n_non_custom_entities = n_entities - n_custom_entities

        seed = n_entities
        rng = np.random.default_rng(seed)
        entities_ids_ = np.copy(np.arange(n_entities))
        rng.shuffle(entities_ids_)
        n_quad_sets[entities_ids_[:n_non_custom_entities]] = 1

        return n_quad_sets

    def _compute_custom_cells(self) -> None:
        """Sets the list of custom cells. The members
        `self._n_quad_sets`.
        """

        cells_id = self._get_cells_ids()
        self._n_quad_sets = self._extract_custom_entities(cells_id.size)
        self._custom_cells_ids = cells_id[self._n_quad_sets > 1]
        self._full_cells_ids = cells_id[self._n_quad_sets == 1]
        self._empty_cells_ids = cells_id[self._n_quad_sets == 0]

    def _compute_custom_facets(self) -> None:
        """Sets the list of custom facets. The members
        `self._n_quad_sets_facets`, `self._custom_facet_cells_ids`, and
        `self._custom_facet_local_facets_ids` are initialized.
        """

        self._int_facet_ids = create_interior_mesh_facets(self._mesh, single_interior_facet=False)
        self._n_quad_sets_int = np.repeat(
            self._extract_custom_entities(self._int_facet_ids.size // 2), 2
        )
        self._ext_facet_ids = create_exterior_mesh_facets(self._mesh)
        self._n_quad_sets_ext = self._extract_custom_entities(self._ext_facet_ids.size)

        def create_facets(facets, mask):
            return MeshFacets(facets.cell_ids[mask], facets.local_facet_ids[mask])

        cut_ext_facet_ids = self._n_quad_sets_ext > 1
        full_ext_facet_ids = self._n_quad_sets_ext == 1
        empty_ext_facet_ids = self._n_quad_sets_ext == 0

        self._ext_cut_facet_ids = create_facets(self._ext_facet_ids, cut_ext_facet_ids)
        self._ext_full_facet_ids = create_facets(self._ext_facet_ids, full_ext_facet_ids)
        self._ext_empty_facet_ids = create_facets(self._ext_facet_ids, empty_ext_facet_ids)

        cut_int_facet_ids = self._n_quad_sets_int > 1
        full_int_facet_ids = self._n_quad_sets_int == 1
        empty_int_facet_ids = self._n_quad_sets_int == 0

        self._int_cut_facet_ids = create_facets(self._int_facet_ids, cut_int_facet_ids)
        self._int_full_facet_ids = create_facets(self._int_facet_ids, full_int_facet_ids)
        self._int_empty_facet_ids = create_facets(self._int_facet_ids, empty_int_facet_ids)

    def _create_ref_quadrature(
        self, degree: int, facet: bool
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Creates the reference quadrature for the given degree of
        exactness.

        Args:
            degree (int): Degree of exactness of the quadrature to be
                built.
            facet (bool): ``True`` if the quadrature to be generated
                corresponds to a facet, ``False`` otherwise.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                Generated quadrature points (first) and weights
                (second).
        """
        # It doesn't matter to use exterior or interior facet here.
        itg_type = "exterior_facet" if facet else "cell"

        ufl_cell = self._mesh.ufl_domain().ufl_cell()

        # In FEniCSx 0.10.0 ``create_quadrature_points_and_weights`` returns
        # dicts keyed by cell name (one entry per cell type the integral
        # may run on). The mock here only deals with one cell type, so
        # flatten the result to a single ``(points, weights)`` pair.
        points_d, weights_d, _ = create_quadrature_points_and_weights(
            itg_type, ufl_cell, degree, "default", [], False
        )
        assert len(points_d) == 1
        cell_name = next(iter(points_d))
        return points_d[cell_name], weights_d[cell_name]  # type: ignore

    def _create_quadrature(
        self, degree: int, n_sets_per_cell: npt.NDArray[np.int32], facet: bool
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generates custom quadratures for cells or facets.

        Args:
            degree (int): Degree of exactness of the quadrature to be
                generated.
            n_sets_per_cell (npt.NDArray[np.int32]): Number of
                quadrature sets (standard quadratures) per custom set.
            facet (bool): Flag indicating whether the quadrature is to
                be built for a facet (``True``) or a cell (``False``).

        Returns:
            tuple[npt.NDArray[np.int32], npt.NDArray[np.float64],
            npt.NDArray[np.float64]]: Generated quadrature. Namely,
            number of points per custom cell (first), quadrature
            weights (second) and points (third) for all the custom
            cells.
        """

        ref_points, ref_weights = self._create_ref_quadrature(degree, facet)

        n_pts_per_set = ref_weights.size
        n_pts_per_cell = n_sets_per_cell * n_pts_per_set
        n_tot_sets = np.sum(n_sets_per_cell)

        # TODO: to improve this
        scale = np.ones(n_tot_sets * n_pts_per_set)
        nz_n_sets_per_entity = n_sets_per_cell[n_sets_per_cell > 0]
        offset = 0
        for n in nz_n_sets_per_entity:
            scale[offset : offset + n * n_pts_per_set] = 1.0 / float(n)
            offset += n * n_pts_per_set

        weights = np.tile(ref_weights, (1, int(n_tot_sets))).reshape(-1)
        weights = weights * scale

        points = np.tile(ref_points, (int(n_tot_sets), 1))

        return n_pts_per_cell, weights.astype(np.float64), points

    def create_quad_custom_cells(
        self, degree: int, cells: npt.NDArray[np.int32], tag: Optional[int] = None
    ) -> CustomQuad:
        """Returns the custom quadratures for the given `cells`.

        For empty cells, no quadrature is generated and will have 0 points
        associated to them. For full cells, the quadrature will be the
        standard one associated to the cell type. While for
        cut cells a custom quadrature for the cell's interior will be
        generated.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

        Returns:
            CustomQuadInterface: Generated custom quadrature.
        """

        subset_ids = np.searchsorted(self._get_cells_ids(), cells)
        n_quad_sets = self._n_quad_sets[subset_ids]

        n_pts_per_cell, weights, points = self._create_quadrature(degree, n_quad_sets, False)
        return CustomQuad(cells, n_pts_per_cell, points, weights)

    def create_quad_unf_boundaries(
        self, degree: int, cells: npt.NDArray[np.int32], tag: Optional[int] = None
    ) -> CustomQuadUnfBoundary:
        """Returns the custom quadrature for unfitted boundaries for the
        given `cells`.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.

            cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

        Returns:
            CustomQuadUnfBoundaryInterface: Generated custom quadrature
            for unfitted boundaries.
        """

        quad = self.create_quad_custom_cells(degree, cells, tag)

        s = quad.points.shape
        normals = np.random.rand(*s)
        normals_norm = np.linalg.norm(normals, axis=1)
        normals_norm = np.repeat(normals_norm, s[1]).reshape(s)
        normals /= normals_norm

        quad_bdry = CustomQuadUnfBoundary(
            quad.cells, quad.n_pts_per_entity, quad.points, quad.weights, normals
        )
        return quad_bdry

    def create_quad_custom_facets(
        self,
        degree: int,
        facets: MeshFacets,
        ext_integral: bool,
    ) -> CustomQuadFacet:
        """Returns the custom quadratures for the given facets.

        For empty facets, no quadrature is generated and will have 0 points
        associated to them. For full facets, the quadrature will be the
        standard one associated to the facet type. While for
        cut facets a custom quadrature for the facet's interior will be
        generated.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.
            facets (MeshFacets): MeshFacets object containing the
                DOLFINx (local) facets for which quadratures will be
                generated.
            ext_integral (bool): If `True` the quadratures will be generated
                considering the given facets as exterior facets.
                Otherwise, as interior.
                See the documentation of `get_cut_facets`,
                `get_full_facets`, and `get_empty_facets` methods
                for more information about the implications of considering
                interior or exterior facets.

        Returns:
            CustomQuadFacetInterface: Generated custom facet
            quadratures.
        """

        if ext_integral:
            _facets = self._ext_facet_ids
            n_quad_sets = self._n_quad_sets_ext
        else:
            _facets = self._int_facet_ids
            n_quad_sets = self._n_quad_sets_int

        # Finds the indices of facets referred to facets
        inds = _facets.find(facets)

        n_quad_sets_dlf = n_quad_sets[inds]
        n_pts_per_cell, weights, points = self._create_quadrature(degree, n_quad_sets_dlf, True)

        return CustomQuadFacet(_facets, n_pts_per_cell, points, weights)

    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        return self._custom_cells_ids

    def get_full_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the full cells.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Returns:
            npt.NDArray[np.int32]: Array of full cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        return self._full_cells_ids

    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        return self._empty_cells_ids

    def get_cut_facets(
        self,
        ext_integral: bool = True,
    ) -> MeshFacets:
        """Gets the cut facets as a MeshFacets object following
        the DOLFINx local numbering.

        Args:
            ext_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            MeshFacets: Cut facets (following DOLFINx local ordering).
        """
        return self._ext_cut_facet_ids if ext_integral else self._int_cut_facet_ids

    def get_full_facets(
        self,
        ext_integral: bool = True,
    ) -> MeshFacets:
        """Gets the full facets as a MeshFacets object following
        the DOLFINx local numbering.

        Args:
            ext_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            MeshFacets: Full facets (following DOLFINx local ordering).
        """
        return self._ext_full_facet_ids if ext_integral else self._int_full_facet_ids

    def get_empty_facets(
        self,
        ext_integral: bool = True,
    ) -> MeshFacets:
        """Gets the empty facets as a MeshFacets object following
        the DOLFINx local numbering.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            ext_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            MeshFacets: Empty facets (following DOLFINx local ordering).
        """
        return self._ext_empty_facet_ids if ext_integral else self._int_empty_facet_ids
