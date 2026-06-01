# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import cast

import dolfinx.cpp.fem
import dolfinx.cpp.fem as cpp_fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from dolfinx.cpp.fem import _IntegralType as IntegralType
from dolfinx.fem.forms import Form

from qugar.dolfinx.custom_quad_utils import map_facets_points, permute_facet_points
from qugar.dolfinx.integral_data import IntegralData
from qugar.dolfinx.quadrature_data import QuadratureData
from qugar.dolfinx.quadrature_store import FloatingArray, QuadratureStore
from qugar.mesh.mesh_facets import MeshFacets
from qugar.mesh.unfitted_domain_abc import UnfittedDomainABC
from qugar.quad import CustomQuad, CustomQuadFacet, CustomQuadUnfBoundary


def _find_common_rows(A: npt.NDArray[np.int32], B: npt.NDArray[np.int32]) -> npt.NDArray[np.intp]:
    """
    Find rows in A that are also present in B.

    Args:
        A: 2D NumPy array of integers
        B: 2D NumPy array of integers with same number of columns as A

    Returns:
        Indices of rows in A that are also in B
    """
    # Handle empty arrays
    if A.size == 0 or B.size == 0:
        return np.array([], dtype=np.intp)

    # Convert to structured arrays to compare rows as single elements
    dtype = np.dtype([("", np.int32)] * A.shape[1])
    A_view = A.view(dtype).reshape(-1)
    B_view = B.view(dtype).reshape(-1)

    # Use isin to find which rows in A are also in B
    mask = np.isin(A_view, B_view)

    # Return indices where mask is True
    return np.where(mask)[0]


class _CustomCoeffsPackerIntegral:
    """Builder of the value-independent :class:`QuadratureStore` for a
    single custom integral.

    It classifies the domain entities (cut / empty), generates (or reuses,
    via the domain-level cache) the custom quadratures, and assembles a
    ``new_coeffs`` *template* with the smuggled geometry and the relative
    offsets already written and the standard-coefficient block left zeroed.
    The result is exposed via :meth:`build_store`; the cheap per-assembly
    overlay of the standard coefficients happens in
    :meth:`QuadratureStore.pack`.

    Parameters:
        _unf_domain (UnfittedDomainABC): The unfitted domain in which the
            integrals are computed.
        _custom_quads_types: All possible types for the custom
            quadratures. Namely ``CustomQuad``, ``CustomQuadFacet``, ``CustomQuadUnfBoundary``
        _itg_data (IntegralData): Data for the integral whose custom
            coefficients are computed.
        _subdomain_id (int): Id of the subdomain
        _domain (npt.NDArray[np.int32]): Domain in which the integrals
            are computed. If the integral is computed over a cell, this
            is just a 1D array containing the cells for which the
            integral is computed. If the integral is computed over an
            exterior facet, this is a 2D array, where each row
            corresponds to a particular facet, being the first column
            the cell id and the second the id of the local facet in the
            cell. Finally, if the integration domain is an interior
            facet `domain` is a 3D array where the first axis
            corresponds to a particular facet; the second axis (that has
            dimension 2) refers to the two sides of the interior facet;
            and the last axis is the id of the local facet in the cell
            (for each side, the local facet refers to cell the facet
            belongs to).
        _mesh (dolfinx.mesh.Mesh): DOLFINx associated to the integral.
        _old_coeffs (FloatingArray): Original (standard) compute
            coefficients for non custom integrals.
        _new_coeffs (FloatingArray): The ``new_coeffs`` template for the
            custom integral (geometry + offsets written, standard block
            zeroed). See the documentation of the method
            `self._build_template` for a more detailed discussion of the
            structure of this array.
        _dtype (type[np.float32 | np.float64]): `numpy` scalar type of the
            integral quantities.
        _coeffs_dtype (FloatingArray): `numpy` scalar type of the
            integral coefficients.
        _offsets_dtype (np.intp): `numpy` scalar type for the offset
            values.
        _custom_quads (dict[IntegralData, _custom_quads_types]):
            Dictionary mapping each integral data (integrand in the
            integral) to a custom quadrature (for cells, facets, or
            unfitted boundaries) for all the custom entities in the
            domain.
        _n_vals_per_entity (npt.NDArray[np.int32]): Number of values to
            be computed / stored per entity, including the non-custom
            ones. The length of the array is the same as the number of
            entities, i.e., the same length along the first axis of
            `self._domain`.
        _offsets (npt.NDArray[np.intp]): Array of absolute offsets.
            The offsets specify for every entity in the domain (cells,
            exterior facets, or first facet in the case of interior
            facets) in which position of the (flatten) array of extra
            custom coefficients needed for the custom integral start.
            It is an absolute position referred to the start of the
            array.
    """

    _custom_quads_types = CustomQuad | CustomQuadUnfBoundary | tuple[CustomQuad, CustomQuad]

    def __init__(
        self,
        unf_domain: UnfittedDomainABC,
        itg_data: IntegralData,
        subdomain_id: int,
        domain: npt.NDArray[np.int32],
        mesh: dolfinx.mesh.Mesh,
        old_coeffs: npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128],
    ) -> None:
        """Initializes the class.

        Args:
            unf_domain (UnfittedDomainABC): The unfitted domain in which the
                integrals are computed.
            itg_data (IntegralData): Data for the integral whose custom
                coefficients are computed.
            subdomain_id (int): Id of the subdomain.
            domain (npt.NDArray[np.int32]): Domain in which the
                integrals are computed. If the integral is computed over
                a cell, this is just a 1D array containing the cells for
                which the integral is computed. If the integral is
                computed over an exterior facet, this is a 2D array,
                where each row corresponds to a particular facet, being
                the first column the cell id and the second the id of
                the local facet in the cell. Finally, if the integration
                domain is an interior facet `domain` is a 3D array where
                the first axis corresponds to a particular facet; the
                second axis (that has dimension 2) refers to the two
                sides of the interior facet; and the last axis is the id
                of the local facet in the cell (for each side, the local
                facet refers to cell the facet belongs to).
            mesh (dolfinx.mesh.Mesh): DOLFINx associated to the
                integral.
            old_coeffs (FloatingArray): `Standard` coefficients
                associated to the non-custom integral.
        """
        self._unf_domain = unf_domain
        self._itg_data = itg_data
        self._subdomain_id = subdomain_id
        self._domain = domain
        self._mesh = mesh
        # ``old_coeffs`` is used only for its shape (number of standard
        # coefficient columns) and dtype; its values are not baked into the
        # template — they are overlaid on every assembly by
        # ``QuadratureStore.pack``.
        self._old_coeffs = old_coeffs
        self._custom_entity_ids = self._get_custom_entities_ids()
        self._empty_entity_ids = self._get_empty_entities_ids()

        self._set_dtypes()
        self._create_quadratures()
        self._build_template()

        return

    def build_store(self) -> QuadratureStore:
        """Returns the :class:`QuadratureStore` for this integral.

        Returns:
            QuadratureStore: The value-independent runtime geometry
            (prebuilt ``new_coeffs`` template + entity classification +
            generated quadratures + offsets).
        """
        return QuadratureStore(
            template=self._new_coeffs,
            old_rows=int(self._old_coeffs.shape[0]),
            old_cols=int(self._old_coeffs.shape[1]),
            custom_entity_ids=self._custom_entity_ids,
            empty_entity_ids=self._empty_entity_ids,
            custom_quads=self._custom_quads,
            n_vals_per_entity=self._n_vals_per_entity,
            offsets=self._offsets,
        )

    def _set_dtypes(self) -> None:
        """Sets the `numpy` types associated to the integral, the
        coefficients, and the offsets. Namely, the members ``_dtype``,
        ``_coeffs_dtype``, and ``_offsets_dtype``.
        """

        self._dtype = np.dtype(self._itg_data.dtype)
        self._coeffs_dtype = np.dtype(self._old_coeffs.dtype)

        # We use np.intp type to store a ptrdiff_t C type.
        self._offsets_dtype = np.dtype(np.intp)

    @property
    def _integral_type(self) -> str:
        """Returns the integral type.

        Returns:
            str: Integral type.
        """
        return self._itg_data.integral_type

    def _is_interior_facet(self) -> bool:
        """Checks if the current integral corresponds to an
        ``interior_facet``.

        Returns:
            bool: ``True`` if the current integral is on an interior
            facet, ``False`` otherwise.
        """
        return self._integral_type == "interior_facet"

    def _is_mixed_dim(self) -> bool:
        """Checks if the current integral presents mixed dimensions.

        Returns:
            bool: ``True`` if the current integral has mixed dimensions,
            ``False`` otherwise.
        """
        return self._itg_data.is_mixed_dim

    def _is_exterior_facet(self) -> bool:
        """Checks if the current integral corresponds to an
        ``exterior_facet``.

        Returns:
            bool: ``True`` if the current integral is on an exterior
            facet, ``False`` otherwise.
        """
        return self._integral_type == "exterior_facet"

    def _is_facet(self) -> bool:
        """Checks if the current integral corresponds to a facet, either
        exterior or interior.

        Returns:
            bool: ``True`` if the current integral is on a facet,
            ``False`` otherwise.
        """
        return self._is_exterior_facet() or self._is_interior_facet()

    def _get_n_extra_cols(self) -> int:
        """Gets the number of columns in the coefficients array required
        for storing the coefficients offset.

        As the offset may be represented with a type with more than
        32 bits (see `self._offsets_dtype`), in the case the
        coefficients array has 32 bits values one single column may not
        suffice for storing the offset. Thus, this function computes
        how many columns are needed depending on the offsets type used
        (`self._offsets_dtype`) and the coefficients type
        (`self._coeffs_dtype`).

        Returns:
            int: Number of columns required for storing custom
            coefficients offsets.
        """

        # A ptrdiff_t value (signed 64 bits int used for offsets) may not
        # into a single coefficient value. In particular, in the case
        # they don't fit in the case coeffients are float 32 bits.
        # There should be no problem for float 64 bits, as well as for
        # complex 64 and 128 bits.
        # In the case of float 32 bits, we pack the ptrdiff_t into
        # two values.
        n_extra_cols = np.ceil(self._offsets_dtype.itemsize / self._coeffs_dtype.itemsize)
        return int(n_extra_cols)

    def _real_per_complex_ratio(self) -> int:
        """How many ``_dtype`` (real) slots fit in one ``_coeffs_dtype``
        cell. 1 for real coeffs; 2 for complex64 / complex128.
        """
        return self._coeffs_dtype.itemsize // self._dtype.itemsize

    def _allocate_new_coeffs(
        self,
    ) -> None:
        """Allocates the `numpy` array for storing the custom
        coefficients and stores it in `self._new_coeffs`.

        The new array is allocated in ``_coeffs_dtype``; for complex
        coeffs the smuggled data section is later written through a real
        view (``new_coeffs.view(_dtype)``), so each complex cell holds two
        real slots. The row count is computed against the real-unit
        capacity per row accordingly.

        This method also initializes the member `self._n_vals_per_entity`
        (in real units).
        """

        self._n_vals_per_entity = self._compute_n_vals_per_custom_entity()

        n_cols_complex = self._old_coeffs.shape[1] + self._get_n_extra_cols()
        # Real slots per row of ``new_coeffs``.
        real_per_cell = self._real_per_complex_ratio()
        n_cols_real = n_cols_complex * real_per_cell

        n_tot_vals = np.sum(self._n_vals_per_entity, dtype=self._offsets_dtype)

        n_rows = self._offsets_dtype.type(self._domain.shape[0])
        if self._is_interior_facet():
            n_rows *= 2
        n_extra_rows = self._offsets_dtype.type(
            np.ceil(n_tot_vals / n_cols_real)
        )

        self._new_coeffs = np.zeros(
            (n_rows + n_extra_rows, n_cols_complex), dtype=self._coeffs_dtype
        )

    def _compute_n_vals_per_custom_entity(self) -> npt.NDArray[np.int32]:
        """Computes the number of values to be stored per custom entity,
        in real-units (``_dtype``).

        For each quadrature attached to an entity we now pack only the
        per-point geometry — the FE table values are computed on the fly
        inside the kernel via the basix tabulation shim. Layout per entity
        per quadrature:

            [ n_pts (1 real slot) | points (gdim * n_pts)
              | points_s1 (gdim * n_pts, interior-facet only)
              | points_facet ((gdim-1) * n_pts, mixed-dim only)
              | weights (n_pts)
              | normals (gdim * n_pts, unfitted-boundary only) ]

        Returns:
            npt.NDArray[np.int32]: Number of *real* slots required per
            custom entity. For complex coefficient forms the smuggled
            data is written through a real view of ``new_coeffs``, so
            this count is the one that drives both allocation and the
            offset arithmetic.
        """

        n_custom_entities = self._custom_entity_ids.size
        n_vals_per_custom_entity = np.zeros(n_custom_entities, dtype=np.int32)

        interior_facet = self._is_interior_facet()
        mixed_dim = self._is_mixed_dim()

        for quad_data, _FE_tables in self._itg_data.quad_data_FE_tables.items():
            custom_quad = self._custom_quads[quad_data]
            if interior_facet:
                custom_quad = custom_quad[0]

            gdim = custom_quad.points.shape[1]
            # gdim point coordinates + 1 weight per point.
            n_vals_per_pt = gdim + 1
            if interior_facet:
                # Side-1 cell-mapped coords (same dim as side 0).
                n_vals_per_pt += gdim
            if mixed_dim:
                # Facet-reference coords for facet-dim elements.
                n_vals_per_pt += self._custom_quads_facets[quad_data].points.shape[1]
            if quad_data.unfitted_boundary:
                n_vals_per_pt += gdim

            n_pts_per_entity = custom_quad.n_pts_per_entity
            assert n_pts_per_entity.size == n_custom_entities

            n_vals_per_custom_entity += n_pts_per_entity * n_vals_per_pt
            # +1 for the n_pts header (cast from int32; all real coeffs
            # dtypes are at least 32 bits).
            n_vals_per_custom_entity += 1

        return n_vals_per_custom_entity

    def _compute_offsets(self) -> None:
        """Computes the custom coefficients offset and copies them into
        the extra (right-most) columns of `self._new_coeffs`.

        Offsets are stored in *real units*: the kernel reads them as
        ``ptrdiff_t`` and adds them to ``(const T*)w`` (cast first, then
        add), so the same byte arithmetic works for both real-coefficient
        and complex-coefficient forms (complex viewed as 2 real slots per
        cell).

        In addition to copying the relative offsets to ``new_coeffs``,
        the absolute (non-relative) offsets are stored in
        ``self._offsets``.
        """

        interior_facet = self._is_interior_facet()

        n_tot_entities = self._domain.shape[0]
        n_entities = 2 * n_tot_entities if interior_facet else n_tot_entities

        # Convert column count from coeffs-dtype cells to real-dtype slots.
        n_cols_complex = self._new_coeffs.shape[1]
        ratio = self._real_per_complex_ratio()
        n_cols_real = n_cols_complex * ratio
        first_extra_pos = self._offsets_dtype.type(n_entities * n_cols_real)

        first_col_pos = (
            np.arange(n_entities, dtype=self._offsets_dtype) * n_cols_real
        )
        if interior_facet:
            first_col_pos = first_col_pos[::2]

        n_custom_entities = self._custom_entity_ids.size
        if n_custom_entities == 0:
            self._offsets = np.empty(0, dtype=self._offsets_dtype)
        else:
            self._offsets = (
                np.concatenate(([0], np.cumsum(self._n_vals_per_entity[:-1]))) + first_extra_pos
            ).astype(self._offsets_dtype)
        assert self._offsets.size == self._custom_entity_ids.size

        rel_offsets = self._offsets - first_col_pos[self._custom_entity_ids]

        # The offset for the full entities are set to -1.
        all_rel_offsets = np.full(n_entities, -1, dtype=rel_offsets.dtype)
        # Then we set the offsets for the custom entities.
        if interior_facet:
            all_rel_offsets_ = all_rel_offsets.reshape(-1, 2)
            all_rel_offsets_[self._custom_entity_ids, 0] = rel_offsets
        else:
            all_rel_offsets[self._custom_entity_ids] = rel_offsets
        # And finally for the empty entities we set the offset to 0.
        all_rel_offsets[self._empty_entity_ids] = 0

        # Copy intp offsets into the trailing coeff cells via a byte view.
        # For coeffs whose cell is smaller-or-equal to intp (float32, float64,
        # complex64) one intp spans n_extra_cols cells and the direct
        # `int -> coeffs_dtype` view works. For coeffs whose cell is larger
        # than intp (complex128 = 16 B vs intp 8 B) we need to pad each intp
        # with zeros to fill one cell before viewing.
        n_extra_cols = self._get_n_extra_cols()
        if self._coeffs_dtype.itemsize <= self._offsets_dtype.itemsize:
            all_rel_offsets_view = all_rel_offsets.view(
                self._coeffs_dtype
            ).reshape([-1, n_extra_cols])
        else:
            # complex128: pad each intp to coeffs_itemsize bytes (intp in the
            # first int slot of the cell, zeros after).
            intps_per_cell = (
                self._coeffs_dtype.itemsize // self._offsets_dtype.itemsize
            )
            padded = np.zeros(
                all_rel_offsets.size * intps_per_cell,
                dtype=self._offsets_dtype,
            )
            padded[::intps_per_cell] = all_rel_offsets
            all_rel_offsets_view = padded.view(self._coeffs_dtype).reshape(
                [-1, n_extra_cols]
            )
        self._new_coeffs[:n_entities, -n_extra_cols:] = all_rel_offsets_view

    def _copy_vals(
        self,
        vals_all_quads: list[
            list[npt.NDArray[np.int32] | npt.NDArray[np.float32] | npt.NDArray[np.float64]]
        ],
    ) -> None:
        """Copies the per-cell smuggled values (n_pts header + points +
        weights + normals + side-1 / facet points where applicable) into
        ``self._new_coeffs`` at the absolute positions stored in
        ``self._offsets``.

        For complex coefficient arrays the underlying memory is viewed as
        the real type ``self._dtype`` (each ``complex128`` cell holds 2
        ``float64`` slots, each ``complex64`` cell holds 2 ``float32``
        slots). Offsets are computed in *real units* throughout
        (``_compute_offsets``), so the same memcpy logic works for both
        real and complex coefficient forms.
        """

        # View the raw memory of new_coeffs as the real per-point dtype.
        # For real coeffs this is identity; for complex coeffs this gives
        # 2x the number of cells with the real layout the kernel reads.
        new_coeffs_real = self._new_coeffs.view(self._dtype).ravel()
        odtype = self._offsets_dtype

        offsets = np.copy(self._offsets)
        n_custom_entities = self._custom_entity_ids.size
        assert offsets.size == n_custom_entities

        for vals_for_quad in vals_all_quads:
            assert len(vals_for_quad) > 0

            vals_offsets = np.zeros(len(vals_for_quad), dtype=odtype)

            n_pts_per_cell = vals_for_quad.pop(0).astype(odtype)

            new_coeffs_real[offsets] = n_pts_per_cell
            offsets += 1

            n_vals_per_pt = np.array([vals.size for vals in vals_for_quad], dtype=odtype)

            n_tot_pts = np.sum(n_pts_per_cell)
            if n_tot_pts > 0:
                n_vals_per_pt //= n_tot_pts

            # TODO: This loop over the cells may be slow and could be
            # improved. Maybe to be done with numba.
            for cell_id in range(n_custom_entities):
                n_pts = n_pts_per_cell[cell_id]
                c_off0 = offsets[cell_id]
                for i, vals in enumerate(vals_for_quad):
                    n_vals = n_pts * n_vals_per_pt[i]

                    c_off1 = c_off0 + n_vals

                    v_off0 = vals_offsets[i]
                    v_off1 = v_off0 + n_vals

                    new_coeffs_real[c_off0:c_off1] = vals[v_off0:v_off1]

                    c_off0 = c_off1
                    vals_offsets[i] = v_off1
                offsets[cell_id] = c_off0

    def _create_quadrature_cell(
        self, quad_data: QuadratureData
    ) -> CustomQuad | CustomQuadUnfBoundary:
        """Creates the custom quadratures for a certain integrand in the
        integral.

        This method manages the case in which the integral is performed
        either in a cell or in an unfitted boundary.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the quadrature is generated.

        Returns:
            Generated quadrature.
        """

        assert not self._is_facet()

        degree = quad_data.degree
        all_cells = self._domain

        custom_cells = all_cells[self._custom_entity_ids]

        flavor = "unf_boundary" if quad_data.unfitted_boundary else "cell"
        return self._unf_domain.get_cached_custom_quadrature(flavor, degree, custom_cells)

    def _create_quadrature_facet(
        self, quad_data: QuadratureData
    ) -> tuple[CustomQuadFacet, CustomQuad | tuple[CustomQuad, CustomQuad]]:
        """Creates the custom quadratures for a certain integrand in the
        the integral.

        This method manages the case in which the integral is performed
        over (internal or external) facets.

        Args:
            quad_data (QuadratureData): Quadrature data of the integrand
                for which the quadrature is generated.

        Returns:
            Generated quadrature. The first item of the tuple is the
            quadrature associated to the facet, while the second is
            that quadrature but mapped to the reference domain of the
            parent cell. In the case of internal facets, the second
            item is a tuple with two mapped quadratures (one for each
            side of the interface).
        """

        assert self._is_facet()

        degree = quad_data.degree

        all_facets = self._get_single_facets()
        custom_facets = MeshFacets(
            all_facets.cell_ids[self._custom_entity_ids],
            all_facets.local_facet_ids[self._custom_entity_ids],
        )

        quad_facet = self._unf_domain.get_cached_custom_quadrature(
            "facet", degree, custom_facets, ext_integral=self._is_exterior_facet()
        )

        quad = self._map_facet_quadrature(quad_facet, custom_facets)

        if not self._is_interior_facet():
            return quad_facet, quad

        custom_facets_1 = MeshFacets(
            self._domain[:, 1, 0].reshape(-1)[self._custom_entity_ids],
            self._domain[:, 1, 1].reshape(-1)[self._custom_entity_ids],
        )
        quad_1 = self._map_facet_quadrature(quad_facet, custom_facets_1)

        return quad_facet, (quad, quad_1)

    def _map_facet_quadrature(
        self,
        quad_facet: CustomQuadFacet,
        facets: MeshFacets,
    ) -> CustomQuad:
        """Maps the given quadrature from a facet to the parent cell
        domain.

        Args:
            quad_facet (CustomQuadFacet): Facet quadrature to
                map.
            facets (MeshFacets): Facets to which the facet quadrature is
                associated to.

        Returns:
            CustomQuad: Generated cell quadrature.
        """

        assert self._is_facet()

        mesh = self._mesh
        needs_permutations = self._is_interior_facet() or self._is_mixed_dim()

        n_pts_per_entity = quad_facet.n_pts_per_entity
        points = quad_facet.points

        cells = cast(npt.NDArray[np.int32], facets.cell_ids)
        local_facets = facets.local_facet_ids
        cells_rep = np.repeat(cells.reshape(-1), n_pts_per_entity)
        facets_rep = np.repeat(local_facets.reshape(-1), n_pts_per_entity)

        if needs_permutations:
            points = permute_facet_points(points, mesh, cells_rep, facets_rep)

        mapped_points = map_facets_points(points, facets_rep, mesh.topology.cell_type)

        return CustomQuad(cells, n_pts_per_entity, mapped_points, quad_facet.weights)

    def _get_single_facets(self) -> MeshFacets:
        """Returns the facets associated to the current integral.

        In the case of interior facets, only the first facet is
        returned.

        Returns:
            MeshFacets: Facets associated to the current integral.
        """
        assert self._is_facet()

        if self._is_interior_facet():
            return MeshFacets(self._domain[:, 0, 0].reshape(-1), self._domain[:, 0, 1].reshape(-1))
        else:
            return MeshFacets(self._domain[:, 0].reshape(-1), self._domain[:, 1].reshape(-1))

    def _get_empty_entities_ids(
        self,
    ) -> npt.NDArray[np.intp]:
        """Returns the empty entities ids referred to all the entities
        in the `self._domain`.

        Returns:
            npt.NDArray[np.intp]: Empty entities ids.
        """

        if self._is_facet():
            exterior_facet = not self._is_interior_facet()

            all_facets = self._get_single_facets()
            empty_facets = self._unf_domain.get_empty_facets(ext_integral=exterior_facet)

            all_cells_facets = cast(npt.NDArray[np.int32], all_facets.as_array().reshape(-1, 2))
            empty_cells_facets = cast(npt.NDArray[np.int32], empty_facets.as_array().reshape(-1, 2))

            return _find_common_rows(all_cells_facets, empty_cells_facets)

        else:
            all_cells = self._domain
            empty_cells = self._unf_domain.get_empty_cells()

            _, empty_ids, _ = np.intersect1d(
                all_cells, empty_cells, assume_unique=True, return_indices=True
            )
            return empty_ids

    def _get_custom_entities_ids(
        self,
    ) -> npt.NDArray[np.intp]:
        """Returns the custom entities ids referred to all the entities
        in the `self._domain`.

        Returns:
            npt.NDArray[np.intp]: Custom entities ids.
        """

        if self._is_facet():
            exterior_facet = not self._is_interior_facet()

            all_facets = self._get_single_facets()
            target_facets = self._unf_domain.get_cut_facets(ext_integral=exterior_facet)

            all_cells_facets = cast(npt.NDArray[np.int32], all_facets.as_array().reshape(-1, 2))
            target_cells_facets = cast(
                npt.NDArray[np.int32], target_facets.as_array().reshape(-1, 2)
            )

            return _find_common_rows(all_cells_facets, target_cells_facets)

        else:
            all_cells = self._domain
            custom_cells = self._unf_domain.get_cut_cells()

            _, custom_entity_ids, _ = np.intersect1d(
                all_cells, custom_cells, assume_unique=True, return_indices=True
            )
            return custom_entity_ids

    def _create_quadratures(self) -> None:
        """Creates the custom quadratures for the integrands in the
        integral and stores them in ``self._custom_quads`` (and, for
        mixed-dimension integrals where a facet-dim element is also
        tabulated, the facet-reference quadrature in
        ``self._custom_quads_facets``).
        """

        self._custom_quads = {}
        self._custom_quads_facets = {}

        for quad_data in self._itg_data.quad_data_FE_tables.keys():
            if self._is_facet():
                quad_facet, quad = self._create_quadrature_facet(quad_data)
                if self._is_mixed_dim():
                    self._custom_quads_facets[quad_data] = quad_facet
            else:
                quad = self._create_quadrature_cell(quad_data)

            self._custom_quads[quad_data] = quad

    def _compute_new_vals(
        self,
    ) -> list[list[npt.NDArray[np.int32] | npt.NDArray[np.float32] | npt.NDArray[np.float64]]]:
        """Computes the per-entity coefficients packed into ``w_custom``
        for the custom integral.

        Under the on-the-fly tabulation design we only pack per-cell
        *geometry* — the FE table values are computed inside the kernel
        by ``qugar_tabulate_<t>``. The order here MUST match the one read
        by ``load_points_..._Q...`` emitted by ``codegeneration.py``:

            [ n_pts | points (gdim * n_pts) | weights (n_pts)
              | normals (gdim * n_pts) if unfitted_boundary ]

        Returns:
            list[list[npt.NDArray[...]]]: One sublist per quadrature; each
            holds flat per-cell-concatenated arrays for ``_copy_vals``.
        """

        vals_all_quads = []
        mixed_dim = self._is_mixed_dim()
        interior_facet = self._is_interior_facet()
        for quad_data, _FE_tables in self._itg_data.quad_data_FE_tables.items():
            if interior_facet:
                custom_quad = self._custom_quads[quad_data][0]
                custom_quad_side1 = self._custom_quads[quad_data][1]
            else:
                custom_quad = self._custom_quads[quad_data]
                custom_quad_side1 = None

            vals_for_quad: list = [custom_quad.n_pts_per_entity]
            vals_for_quad.append(np.ascontiguousarray(custom_quad.points).ravel())
            if interior_facet:
                vals_for_quad.append(
                    np.ascontiguousarray(custom_quad_side1.points).ravel())
            if mixed_dim:
                vals_for_quad.append(
                    np.ascontiguousarray(
                        self._custom_quads_facets[quad_data].points).ravel())
            vals_for_quad.append(custom_quad.weights)
            if quad_data.unfitted_boundary:
                vals_for_quad.append(custom_quad.normals.ravel())

            vals_all_quads.append(vals_for_quad)

        return vals_all_quads

    def _build_template(self) -> None:
        """Builds the ``new_coeffs`` *template* and stores it in
        `self._new_coeffs`.

        The template holds all the parts of the custom coefficients that
        do *not* depend on the form's coefficient values: the smuggled
        per-entity geometry (points, weights, normals, ...) and the
        relative offset columns. The standard-coefficient block (the
        top-left ``[:old_rows, :old_cols]`` region) is left zeroed; it is
        overlaid with the freshly packed DOLFINx coefficients on every
        assembly by :meth:`QuadratureStore.pack`.

        The offset columns work as follows: if an entity's relative offset
        is ``-1`` it is a full (standard) entity using only the standard
        coefficients; if it is ``0`` the entity is empty; otherwise the
        value is the position (relative to the start of the entity's row)
        where that entity's smuggled custom data starts.
        """

        self._allocate_new_coeffs()
        self._compute_offsets()
        self._copy_vals(self._compute_new_vals())


def _build_quadrature_store(
    form: Form,
    unf_domain: UnfittedDomainABC,
    itg_data: IntegralData,
    itg_info: tuple[IntegralType, int],
    old_coeffs: FloatingArray,
) -> QuadratureStore:
    """Builds the :class:`QuadratureStore` for an integral.

    Args:
        form (Form): DOLFINx form associated to the integral whose
            custom coefficients are computed.
        unf_domain (UnfittedDomainABC): The unfitted domain in which the
            integrals are computed.
        itg_data (IntegralData): Data of the integral whose custom
            coefficients are computed.
        itg_info (tuple[IntegralType, int]): Integral type and position
            identifying the integral within the form.
        old_coeffs (FloatingArray): `Standard` coefficients associated
            to the non-custom integral, used only for their shape and
            dtype.

    Returns:
        QuadratureStore: The value-independent runtime geometry for the
        integral.
    """

    domain = form._cpp_object.domains(*itg_info)
    mesh = form._cpp_object.mesh

    builder = _CustomCoeffsPackerIntegral(
        unf_domain, itg_data, itg_info[1], domain, mesh, old_coeffs
    )

    return builder.build_store()


class CustomCoeffsPacker:
    """Class for computing coefficients for custom integrals.

    Beyond its constructor, its public API is the methods
    `pack_coefficients` and `update_coefficients`.

    Parameters:
        _form (Form): DOLFINx form (containing the custom integral
            module).
        _domain (UnfittedDomainABC): The unfitted domain in which the
            integrals are computed.
        _itg_data (list[IntegralData]): Data of all the integrals
            contained in the `form`.
        _stores (dict[tuple[IntegralType, int], QuadratureStore] | None):
            Cache of the per-integral quadrature stores. Built lazily on
            the first ``pack_coefficients`` call (when the standard
            coefficient shape/dtype become known) and reused afterwards so
            the runtime quadrature is generated only once.
    """

    def __init__(self, form: Form, domain: UnfittedDomainABC, itg_data: list[IntegralData]) -> None:
        """Initializes the class.

        Args:
            form (Form): DOLFINx form (containing the custom integral).
            domain (UnfittedDomainABC): The unfitted domain in which the
                integrals are computed.
            itg_data (list[IntegralData]): Data of all the integrals
                contained in the `form`.
        """

        self._form = form
        self._domain = domain
        self._itg_data = itg_data
        self._stores: dict[tuple[IntegralType, int], QuadratureStore] = {}
        return

    def _get_itg_data(self, itg_info: tuple[IntegralType, int]) -> IntegralData:
        """Gets the integral data whose info matches `itg_info`.

        Args:
            itg_info (tuple[IntegralType, int]): Integral info as
                returned by DOLFINx ``pack_coefficients``. In v0.9.0
                this was ``(integral_type, subdomain_id)``; in v0.10.0
                it became ``(integral_type, position_in_per_type_list)``.

        Returns:
            IntegralData: Integral data instance found.
        """

        valid_integral_types = (
            IntegralType.cell,
            IntegralType.exterior_facet,
            IntegralType.interior_facet,
        )

        assert itg_info[0] in valid_integral_types

        if not hasattr(self, "_position_to_data"):
            self._position_to_data = self._build_position_to_data()

        if itg_info in self._position_to_data:
            return self._position_to_data[itg_info]

        assert False, f"Integral data not found for {itg_info!r}."

    def _build_position_to_data(self) -> dict[tuple[IntegralType, int], IntegralData]:
        """Build a (integral_type, position) -> IntegralData map.

        DOLFINx 0.10.0 changed the convention for ``pack_coefficients``
        keys from ``(integral_type, subdomain_id)`` to
        ``(integral_type, position_in_per_type_list)``. The
        position-to-subdomain_id mapping is exactly what dolfinx itself
        reads from ``ufcx_form.form_integral_ids`` /
        ``form_integral_offsets`` when it builds the form, so replicate
        that here to translate the new positional keys back to the
        IntegralData objects qugar keeps around.
        """
        ufcx_form = self._form._ufcx_form
        ufcx_types = ("cell", "exterior_facet", "interior_facet", "vertex", "ridge")
        type_to_enum = {
            "cell": IntegralType.cell,
            "exterior_facet": IntegralType.exterior_facet,
            "interior_facet": IntegralType.interior_facet,
        }
        offsets = [
            ufcx_form.form_integral_offsets[i] for i in range(len(ufcx_types) + 1)
        ]

        result: dict[tuple[IntegralType, int], IntegralData] = {}
        for i, type_str in enumerate(ufcx_types):
            if type_str not in type_to_enum:
                continue
            type_enum = type_to_enum[type_str]
            for position, j in enumerate(range(offsets[i], offsets[i + 1])):
                sid = ufcx_form.form_integral_ids[j]
                for data in self._itg_data:
                    if data.integral_type == type_str and sid in data.subdomain_ids:
                        result[(type_enum, position)] = data
                        break
        return result

    def pack_coefficients(
        self,
    ) -> dict[tuple[IntegralType, int], npt.NDArray]:
        """Generates the custom coefficients consumed by custom
        integrals for all the integral types and subdomains.

        Returns:
            dict[tuple[IntegralType, int], npt.NDArray]: Generated
            custom coefficients (it follows the same data structure as
            ``dolfinx.cpp.fem.pack_coefficients``).
        """

        coeffs = cpp_fem.pack_coefficients(self._form._cpp_object)

        new_pack_coeffs: dict[
            tuple[IntegralType, int],
            npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128],
        ] = {}

        for itg_info, old_coeffs in coeffs.items():
            store = self._stores.get(itg_info)
            if store is None:
                itg_data = self._get_itg_data(itg_info)
                store = _build_quadrature_store(
                    self._form, self._domain, itg_data, itg_info, old_coeffs
                )
                self._stores[itg_info] = store
            new_pack_coeffs[itg_info] = store.pack(old_coeffs)

        return new_pack_coeffs

    def update_coefficients(
        self,
        coeffs: dict[tuple[IntegralType, int], npt.NDArray],
    ) -> dict[tuple[IntegralType, int], npt.NDArray]:
        """Updates the custom coefficients for all the integral types
        and subdomains.

        It updated coefficients previously generated by just recomputing
        the part of the coefficients associated to the DOLFINx coefficients
        and keeping the part of the coefficients associated to the
        custom integrals.

        Args:
            coeffs (dict[tuple[IntegralType, int], npt.NDArray]):
                Old coefficients to be updated.

        Returns:
            dict[tuple[IntegralType, int], npt.NDArray]: Generated
            custom coefficients (it follows the same data structure as
            ``dolfinx.cpp.fem.pack_coefficients``).
        """

        new_coeffs_dlf = cpp_fem.pack_coefficients(self._form._cpp_object)

        new_coeffs = {}
        for itg_info, _new_coeffs_dlf in new_coeffs_dlf.items():
            shape = _new_coeffs_dlf.shape
            c = coeffs[itg_info]
            c[: shape[0], : shape[1]] = _new_coeffs_dlf
            new_coeffs[itg_info] = c

        return new_coeffs
