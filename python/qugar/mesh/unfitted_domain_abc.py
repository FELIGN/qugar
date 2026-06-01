# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------


from qugar import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from abc import ABC, abstractmethod
from typing import Optional

import dolfinx.mesh
import numpy as np
import numpy.typing as npt
from dolfinx.mesh import MeshTags, meshtags

from qugar import has_FEniCSx
from qugar.mesh.mesh_facets import MeshFacets, create_all_mesh_facets
from qugar.quad.custom_quad import (
    CustomQuad,
    CustomQuadFacet,
    CustomQuadUnfBoundary,
)


class UnfittedDomainABC(ABC):
    """Abstract base class for storing unfitted domains.

    This class provides (abstract) methods for accessing the cut, full,
    and empty cells and facets of the mesh. It also provides methods for
    creating quadrature rules for the cut cells and facets, as well as
    for unfitted boundaries.

    Classes that inherit from this class should implement the
    `get_cut_cells`, `get_full_cells`, `get_empty_cells`, `get_cut_facets`,
    `get_full_facets`, and `get_empty_facets` methods
    to access the cut, full, empty cells and facets, and the cells
    containing unfitted boundaries, respectively. The `create_quad_custom_cells`,
    `create_quad_unf_boundaries`, and `create_quad_custom_facets` methods
    should be implemented to create custom quadrature rules for the cut
    cells and unfitted boundaries.

    This class also provides methods for creating subdomain data for the
    different types of cells and faces.

    Note:
        This class is purely abstract and should not be instantiated
        directly. It is intended to be used as a base class for other
        classes that implement the abstract methods.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh):
        """Initializes the UnfittedDomainABC class.

        Args:
            mesh: The mesh object to be used.
        """
        self._mesh = mesh
        setattr(self._mesh._ufl_domain, "unf_domain", self)

        # Cache of generated custom quadratures, shared across every form
        # built on this domain. Keyed by (flavor, degree, ...entity bytes)
        # so that, e.g., the bilinear and linear forms of a problem (or a
        # Jacobian and a residual) reuse the same cut-cell / cut-facet /
        # unfitted-boundary quadrature instead of regenerating it. See
        # `get_cached_custom_quadrature`.
        self._quad_cache: dict = {}

    @abstractmethod
    def get_cut_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the cut cells.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Returns:
            npt.NDArray[np.int32]: Array of cut cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_empty_cells(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the empty cells.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Returns:
            npt.NDArray[np.int32]: Array of empty cells associated to the
            current process following the DOLFINx local numbering.
            The cell ids are sorted.
        """
        pass

    @abstractmethod
    def get_cut_facets(
        self,
        ext_integral: bool = True,
    ) -> MeshFacets:
        """Gets the cut facets as a MeshFacets object following
        the DOLFINx local numbering.

        The list of facets will be filtered for exterior or interior
        integrals according to the argument `ext_integral`.

        Note:
            The selection of facets is performed differently depending
            on whether exterior or interior integrals are to be
            computed.
            For interior integrals we consider interior facets (shared
            by two cells) that are cut (i.e., they are partially inside
            the domain).
            For exterior integrals we consider either interior or
            exterior (that belong to a single cell) facets that
            partially belong to the domain's boundary (either the mesh's
            domain or the unfitted boundary).
            In the case of exterior integrals, if a facet is fully
            contained in an unfitted boundary, it is considered as full.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            ext_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            MeshFacets: Cut facets (following DOLFINx local ordering).
        """
        pass

    @abstractmethod
    def get_full_facets(
        self,
        ext_integral: bool = True,
    ) -> MeshFacets:
        """Gets the full facets as a MeshFacets object following
        the DOLFINx local numbering.

        The list of facets will be filtered for exterior or interior
        integrals according to the argument `ext_integral`.

        Note:
            The selection of facets is performed differently depending
            on whether exterior or interior integrals are to be
            computed.
            For interior integrals we consider interior facets (shared
            by two cells) that are fully inside the domain (and not
            contained in any unfitted boundary).
            For exterior integrals we consider facets (interior or
            exterior) that are fully contained in the domain's boundary
            (either the mesh's domain or the unfitted boundary).

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            ext_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            MeshFacets: Full facets (following DOLFINx local ordering).
        """
        pass

    @abstractmethod
    def get_empty_facets(
        self,
        ext_integral: bool = True,
    ) -> MeshFacets:
        """Gets the empty facets as a MeshFacets object following
        the DOLFINx local numbering.

        The list of facets will be filtered to only exterior or interior
        facets according to the argument `exterior`.

        Note:
            The selection of facets is performed differently depending
            on whether exterior or interior integrals are to be
            computed.
            For interior integrals we consider interior facets (shared
            by two cells) that are not contained in the domain (they
            may be contained (fully or partially) in the unfitted
            boundary).
            For exterior integrals we consider exterior facets (that
            belong to a single cell) that are not contained in the
            domain or its boundary.

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
        pass

    def get_all_facets(self, ext_integral: bool = True) -> MeshFacets:
        """Gets all the facets of the mesh.

        The list of facets will be filtered to only exterior or interior
        facets according to the argument `exterior`.

        Note:
            The selection of facets is performed differently depending
            on whether exterior or interior integrals are to be
            computed.
            For exterior integrals we consider all facets that belong to
            the domain's boundary (either the mesh's domain or the
            unfitted boundary).
            For interior integrals we consider the complementary.

        Note:
            This is an abstract method and can be overridden in derive
            classes.

        Args:
            ext_integral (bool): Whether the list of facets is
                retrieved for computing exterior or interior integrals
                (see note above).

        Returns:
            MeshFacets: All facets (following DOLFINx local ordering).
        """

        ext_facets = self.get_cut_facets(ext_integral=True)
        ext_facets = ext_facets.concatenate(self.get_full_facets(ext_integral=True))
        ext_facets = ext_facets.concatenate(self.get_empty_facets(ext_integral=True))

        if ext_integral:
            return ext_facets
        else:
            all_facets = create_all_mesh_facets(self._mesh, single_interior_facet=False)
            return all_facets.difference(ext_facets)

    @abstractmethod
    def create_quad_custom_cells(
        self,
        degree: int,
        cells: npt.NDArray[np.int32],
    ) -> CustomQuad:
        """Returns the custom quadratures for the given `cells`.

        For empty cells, no quadrature is generated and will have 0 points
        associated to them. For full cells, the quadrature will be the
        standard one associated to the cell type. While for
        cut cells a custom quadrature for the cell's interior will be
        generated.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.
            cells (npt.NDArray[np.int32]): Array of DOLFINx cell ids
                (local to current MPI process) for which quadratures
                will be generated.

        Returns:
            CustomQuadInterface: Generated custom quadrature.
        """
        pass

    @abstractmethod
    def create_quad_unf_boundaries(
        self,
        degree: int,
        cells: npt.NDArray[np.int32],
    ) -> CustomQuadUnfBoundary:
        """Returns the custom quadrature for unfitted boundaries for the
        given `cells`.

        Note:
            Some unfitted boundary parts may lay over facets.
            The quadrature corresponding to those facets will be generated
            with the method `create_quad_custom_facets`.

        Note:
            For cells not containing unfitted boundaries, no quadrature
            is generated and will have 0 points associated to them.

        Note:
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

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
        pass

    @abstractmethod
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
            This call may require the generation of the quadratures on
            the fly, what can be potentially expensive.

        Note:
            This is an abstract method and should be implemented in
            derived classes.

        Args:
            degree (int): Expected degree of exactness of the quadrature
                to be generated.
            facets (MeshFacets): MeshFacets object containing the
                DOLFINx (local) facets for which quadratures will be
                generated.
            ext_integral (bool): Whether exterior integrals are
                to be computed using the generated quadratures.
                If `True`, the quadrature will be generated for the
                facets that belong to the domain's boundary (either
                the mesh's domain or the unfitted boundary).
                Otherwise, the quadrature will be generated only for
                the parts of the interior facets that are inside the
                domain (but not its boundary).

        Returns:
            CustomQuadFacetInterface: Generated custom facet
            quadratures.
        """
        pass

    def get_cached_custom_quadrature(
        self,
        flavor: str,
        degree: int,
        entities: "npt.NDArray[np.int32] | MeshFacets",
        ext_integral: Optional[bool] = None,
    ) -> "CustomQuad | CustomQuadUnfBoundary | CustomQuadFacet":
        """Returns a custom quadrature, generating it on the fly the
        first time and reusing the cached result afterwards.

        This is a thin memoizing wrapper around the
        ``create_quad_custom_cells``, ``create_quad_unf_boundaries``, and
        ``create_quad_custom_facets`` generators. The cache lives on the
        domain instance, so distinct forms built on the same domain
        (e.g. a bilinear and a linear form, or a Jacobian and a residual)
        reuse the same generated quadrature instead of recomputing it,
        as long as they request the same `entities` at the same `degree`.

        Args:
            flavor (str): One of ``"cell"``, ``"unf_boundary"``, or
                ``"facet"``, selecting which generator to call.
            degree (int): Expected degree of exactness of the quadrature.
            entities (npt.NDArray[np.int32] | MeshFacets): The cells
                (for ``"cell"`` / ``"unf_boundary"``) or facets (for
                ``"facet"``) for which the quadrature is generated.
            ext_integral (Optional[bool]): For the ``"facet"`` flavor,
                whether the quadrature is for an exterior integral. Unused
                for the other flavors.

        Returns:
            CustomQuad | CustomQuadUnfBoundary | CustomQuadFacet: The
            generated (or cached) custom quadrature.
        """

        if flavor in ("cell", "unf_boundary"):
            # ``entities`` is a 1D array of cell ids; its bytes uniquely
            # identify the request (no hash collisions).
            key = (flavor, degree, entities.tobytes())  # type: ignore[union-attr]
        elif flavor == "facet":
            key = (
                flavor,
                degree,
                bool(ext_integral),
                entities.cell_ids.tobytes(),  # type: ignore[union-attr]
                entities.local_facet_ids.tobytes(),  # type: ignore[union-attr]
            )
        else:
            raise ValueError(f"Unknown quadrature flavor: {flavor!r}.")

        quad = self._quad_cache.get(key)
        if quad is None:
            if flavor == "cell":
                quad = self.create_quad_custom_cells(degree, entities)  # type: ignore[arg-type]
            elif flavor == "unf_boundary":
                quad = self.create_quad_unf_boundaries(degree, entities)  # type: ignore[arg-type]
            else:
                quad = self.create_quad_custom_facets(
                    degree, entities, bool(ext_integral)
                )  # type: ignore[arg-type]
            self._quad_cache[key] = quad

        return quad

    if has_FEniCSx:

        def create_cell_meshtags(
            self,
            cut_tag: Optional[int] = None,
            full_tag: Optional[int] = None,
            empty_tag: Optional[int] = None,
        ) -> MeshTags:
            """Creates a cell mesh tags container to identify the cut, full,
            and/or empty cells.

            These tags can be used to prescribe the `subdomain_data` to which
            integration measure apply. E.g., `ufl.dx(subdomain_data=cell_meshtags)`.

            If the tag for `cut`, `full`, or `empty` tags are not provided
            (they are set to `None`), those cells will not be included in the
            generated mesh tags.

            Args:
                cut_tag (Optional[int]): Tag to assign to cut cells. Defaults to None.
                full_tag (Optional[int]): Tag to assign to full cells. Defaults to None.
                empty_tag (Optional[int]): Tag to assign to empty cells. Defaults to None.

            Returns:
                MeshTags: Generated mesh tags.
            """

            cells = np.empty(0, dtype=np.int32)
            values = np.empty(0, dtype=np.int32)

            def add_cells(getter, tag):
                nonlocal cells, values

                new_cells = getter()
                cells = np.append(cells, new_cells)
                values = np.append(values, np.full_like(new_cells, tag))

            for getter, tag in {
                self.get_cut_cells: cut_tag,
                self.get_full_cells: full_tag,
                self.get_empty_cells: empty_tag,
            }.items():
                if tag is not None:
                    add_cells(getter, tag)

            tdim = self._mesh.topology.dim

            # Sort cells and values according to cell indices for consistent ordering
            order = np.argsort(cells)
            cells = cells[order]
            values = values[order]

            return meshtags(self._mesh, tdim, cells, values)

    def create_facet_tags(
        self,
        cut_tag: Optional[int] = None,
        full_tag: Optional[int] = None,
        empty_tag: Optional[int] = None,
        ext_integral: bool = True,
    ) -> list[tuple[int, npt.NDArray[np.int32]]]:
        """Creates a facet tags container to identify the cut, full,
        and empty cells.

        These tags can be used to prescribe the `subdomain_data` to which
        integration measure apply. E.g., `ufl.ds(subdomain_data=cell_tags)`.

        If the tag for cut, full, or empty tags are not provided, those
        facets will not be included in the list.

        Note:
            The reason to return such type of object is to be able to
            identify one sided interior facets (shared by two cells)
            that may be associated to unfitted boundaries (what is not
            possible using the DOLFINx `MeshTags` data structure).

        Args:
            cut_tag (Optional[int]): Tag to assign to cut facets.
                Defaults to None.
            full_tag (Optional[int]): Tag to assign to full facets.
                Defaults to None.
            empty_tag (Optional[int]): Tag to assign to empty facets.
                Defaults to None.
            ext_integral (bool): If `True`, the generated facet
                lists will be used for computing exterior integrals.
                Otherwise, for interior integrals. Check the
                documentation of `get_cut_facets`, `get_full_facets`,
                and `get_empty_facets` methods for more information
                about the implications of considering interior or
                exterior integrals. Defaults to True.

        Returns:
            list[tuple[int, npt.NDArray[np.int32]]]: Generated facet
            tags. It is a list where each entry is a tuple with a tag
            (identifier) and an array of facets. The array of facets is
            made of consecutive pairs of cells and local facet ids.
        """

        facet_tags = {}

        def add_facets(facets, tag):
            if tag in facet_tags:
                if not facets.empty:
                    # No need for unique, as the sets of cut, empty, and
                    # full facets should be disjoint. Also, the facets
                    # are not required to be returned in any particular order.
                    facet_tags[tag] = facet_tags[tag].concatenate(facets)
            else:
                facet_tags[tag] = facets

        if cut_tag is not None:
            add_facets(self.get_cut_facets(ext_integral), cut_tag)

        if full_tag is not None:
            add_facets(self.get_full_facets(ext_integral), full_tag)

        if empty_tag is not None:
            add_facets(self.get_empty_facets(ext_integral), cut_tag)

        return list((tag, facets.as_array()) for tag, facets in facet_tags.items())
