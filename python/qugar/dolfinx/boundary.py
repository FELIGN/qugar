# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"Data structures for computing quantities on custom boundaries."

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import Optional

import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl
from ufl.core.ufl_type import ufl_type
from ufl.geometry import GeometricCellQuantity, Jacobian
from ufl.measure import integral_type_to_measure_name
from ufl.operators import inv, transpose
from ufl.protocols import id_or_none


@ufl_type()
class UnfittedReferenceNormal(GeometricCellQuantity):
    """UFL geometric terminal: the reference-domain normal of an unfitted
    boundary, evaluated per quadrature point at runtime.

    This is the primitive runtime input behind :func:`dsu_normal`: the
    physical normal is obtained by mapping it through the cell Jacobian
    (Nanson's formula). It replaces the former ``ParamNormal`` zero-valued
    ``Constant`` placeholder. Modelling the normal as a proper geometric
    quantity lets FFCx lower it *structurally* — see
    :mod:`qugar.dolfinx._ffcx_patches`, which registers a backend handler
    emitting ``normals_<quad>[tdim * iq + i]`` — instead of rewriting the
    generated C text.

    Note:
        The terminal is (deliberately) cellwise-constant, like every UFL
        ``GeometricQuantity`` by default. FFCx assumes a genuinely
        per-point quantity is backed by a basis table and rejects a
        table-less one, so we keep the cellwise-constant classification;
        the per-point variation is reintroduced downstream when qugar
        relocates the pre-loop block into the quadrature loop (see
        :meth:`qugar.dolfinx._kernel_body.KernelBody.inline_pre_loop_into_loops`).
    """

    __slots__ = ()
    name = "unfitted_reference_normal"

    @property
    def ufl_shape(self):
        """Vector shape, one component per topological dimension."""
        return (self._domain.topological_dimension(),)


def _compute_vector_norm(vec):
    """Computes the norm of a vector.

    Args:
        vec: Vector whose norm is computed.

    Returns:
        Computed norm.
    """
    return ufl.sqrt(ufl.inner(vec, vec))


def dsu_normal(domain: ufl.AbstractDomain | dolfinx.mesh.Mesh, normalize: bool = True):
    """Physical (mapped) outward normal of an unfitted boundary, for use
    inside a :class:`dsu` integrand.

    The reference-domain normal (:class:`UnfittedReferenceNormal`) is
    mapped to physical space through the cell Jacobian, following Nanson's
    formula.

    Args:
        domain (ufl.AbstractDomain | dolfinx.mesh.Mesh): An AbstractDomain object (most
            often a Mesh) to which the normal is associated to.
        normalize (bool): If ``True``, the returned normal is
            normalized, otherwise (if ``False``) it is not normalized
            and its norm is the ratio of the deformed to reference
            area cells.

    Returns:
        Mapped normal.
    """

    N = UnfittedReferenceNormal(domain)

    DF = Jacobian(domain)

    if DF.ufl_shape[0] != DF.ufl_shape[1]:
        metric = transpose(DF) * DF  # type: ignore
        n = DF * inv(metric) * N  # type: ignore
    else:
        n = transpose(inv(DF)) * N  # type: ignore

    if normalize:
        return n / _compute_vector_norm(n)  # type: ignore
    else:
        return n


class dsu(ufl.Measure):
    """UFL measure for integration over an unfitted custom boundary
    (``dsu`` = "ds, unfitted").

    It behaves like ``ufl.dx`` except that, when multiplied by an
    integrand, it introduces the Nanson correction that accounts for the
    boundary orientation through its normal.

    The measure is recognised downstream solely by the
    ``custom_unfitted_boundary`` flag it sets in the quadrature metadata
    (see :func:`qugar.dolfinx.quadrature_data.extract_quadrature_data`).
    It also routes through FFCx's custom-quadrature path, which requires
    placeholder points/weights to be present even though the real ones
    are supplied per cell at runtime; see :meth:`_placeholder_quadrature`.
    """

    def __init__(
        self,
        domain: ufl.AbstractDomain | dolfinx.mesh.Mesh,
        subdomain_id: str | int | tuple[int] = "everywhere",
        metadata: dict | None = None,
        subdomain_data: dolfinx.mesh.MeshTags
        | list[tuple[int, npt.NDArray[np.int32]]]
        | None = None,
        degree: Optional[int] = None,
    ):
        """Initialize.

        Args:
            domain (ufl.AbstractDomain | dolfinx.mesh.Mesh): An AbstractDomain object (most
                often a Mesh).
            subdomain_id (str | int | tuple[int], optional): either
                string "everywhere", a single subdomain id int, or tuple
                of ints. Defaults to "everywhere".
            metadata (dict | None, optional): Dictionary with additional
                compiler-specific parameters for optimization or
                debugging of generated code. Defaults to None.
            subdomain_data (dolfinx.mesh.MeshTags | list[tuple[int, npt.NDArray[np.int32]]] | None, optional):
                Object representing data to interpret subdomain_id with. Defaults to None.
            degree (int, optional): The degree of the quadrature rule.
        """  # noqa: E501

        assert domain is not None

        metadata = {} if metadata is None else metadata.copy()
        # When called from reconstruct, ``degree`` is None but the old
        # metadata carries ``quadrature_degree``. Recover it so that
        # ``_create_custom_metadata`` produces degree-specific placeholder
        # points instead of the degree=None fall-back (which would make
        # all restricted measures hash to the same quadrature name).
        if degree is None:
            degree = metadata.get("quadrature_degree")
        if degree is not None:
            metadata["quadrature_degree"] = degree

        metadata.update(self._create_custom_metadata(domain, degree))

        super().__init__(
            integral_type="cell",
            domain=domain,
            subdomain_id=subdomain_id,  # type: ignore
            metadata=metadata,
            subdomain_data=subdomain_data,
        )

        # The dolfinx mesh is needed (rather than ufl_domain(), which is
        # a plain ufl.Mesh) for ``reconstruct`` to rebuild this subclass
        # and re-run UnfittedReferenceNormal/dsu_normal on subdomain_id changes.
        self._dolfinx_domain = domain

        n = dsu_normal(domain, normalize=False)
        # This is the missing term in Nanson's formula (the Jacobian
        # determinant should by already included in dx).
        self._measure_complement = _compute_vector_norm(n)

        self._integral_type_mod = "unfitted_custom_boundary"
        self._measure_name = "dsu"

    @staticmethod
    def _placeholder_quadrature(
        tdim: int, degree: Optional[int]
    ) -> tuple[list[list[float]], list[float]]:
        """Builds the placeholder quadrature (points and weights) for the
        custom-quadrature metadata.

        These values carry no quadrature meaning: the real points and
        weights are supplied per cell at runtime. The negative weights
        are a sentinel that no genuine rule produces.

        FFCx derives a quadrature's identifier from ``sha1(points)`` only
        (see ``ffcx.ir.representationutils.QuadratureRule.id``). The
        points must therefore differ per ``(tdim, degree)``: otherwise
        two unfitted-boundary integrals of different degree would hash to
        the same name and clobber each other in
        :func:`qugar.dolfinx.quadrature_data.extract_quadrature_data`. We
        offset the points by the degree to keep them unique. When
        ``degree`` is ``None`` the effective degree is only known later
        (FFCx fills it in), so we fall back to a fixed offset and rely on
        ``extract_quadrature_data`` to raise on any residual collision.

        Args:
            tdim (int): Topological dimension of the cell.
            degree (int | None): Quadrature degree, if specified.

        Returns:
            tuple[list[list[float]], list[float]]: Placeholder points
            (two per rule) and their weights.
        """

        step = 0.1 * (1 if degree is None else degree + 1)
        points = [[step] * tdim, [2.0 * step] * tdim]
        weights = [-1.0, -1.0]
        return points, weights

    def _create_custom_metadata(self, domain: ufl.AbstractDomain, degree: Optional[int]) -> dict:
        """Creates the custom measure metadata.

        The only semantically meaningful entry is the
        ``custom_unfitted_boundary`` flag, which is the single source of
        truth used downstream to recognise this measure. The
        ``quadrature_rule="custom"`` entry and the placeholder
        points/weights are FFCx ceremony (see
        :meth:`_placeholder_quadrature`).

        Args:
            domain (ufl.AbstractDomain): An AbstractDomain object (most
                often a Mesh).
            degree (int | None): Quadrature degree, if specified.

        Returns:
            dict: Generated metadata.
        """

        assert isinstance(domain, dolfinx.mesh.Mesh)

        points, weights = self._placeholder_quadrature(domain.topology.dim, degree)

        return {
            "quadrature_points": points,
            "quadrature_weights": weights,
            "quadrature_rule": "custom",
            "custom_unfitted_boundary": True,
        }

    def __rmul__(self, integrand):
        """Multiply a scalar expression with measure to construct a form
        with a single integral.

        This is to implement the notation
        ``form = integrand * measure_complement * self`` where
        ``measure_complement`` is the norm of a mapped normal to the
        boundary, according to Nanson's formula.

        Integration properties are taken from this Measure object.
        """
        return super().__rmul__(integrand * self._measure_complement)

    def reconstruct(
        self,
        integral_type=None,
        subdomain_id=None,
        domain=None,
        metadata=None,
        subdomain_data=None,
    ):
        # Without this override the syntax ``ds(cut_tag)`` (and the
        # tuple-id branch in ufl.Measure.__rmul__) would return a plain
        # ufl.Measure, silently dropping the Nanson correction.
        new_domain = domain if domain is not None else self._dolfinx_domain
        new = type(self)(
            domain=new_domain,
            subdomain_id=subdomain_id if subdomain_id is not None else self.subdomain_id(),
            metadata=metadata if metadata is not None else self.metadata(),
            subdomain_data=subdomain_data if subdomain_data is not None else self.subdomain_data(),
        )
        # Reuse the existing Nanson term when the domain is unchanged so
        # that ``f * ds(id)`` and a freshly-constructed equivalent form
        # share form signatures (otherwise each call would mint a new
        # normal terminal and bust the FFCx JIT cache).
        if new_domain is self._dolfinx_domain:
            new._measure_complement = self._measure_complement
        return new

    def __str__(self) -> str:
        """Formats the class instance as a string.

        Returns:
            str: Generated string.
        """
        parent_name = integral_type_to_measure_name[self._integral_type]
        parent_str = super().__str__()
        return parent_str.replace(parent_name, self._measure_name, 1)

    def __repr__(self) -> str:
        """Return a repr string for this Measure.

        Returns:
            str: Generated representation.
        """
        parent_repr = super().__repr__()
        new_repr = parent_repr.replace(type(self).__name__, "Measure", 1)
        new_repr = new_repr.replace(repr(self._integral_type), repr(self._integral_type_mod), 1)
        return new_repr

    def __hash__(self) -> int:
        """Return a hash value for this Measure.

        Returns:
            int: Generated hash.
        """
        metadata_hashdata = tuple(sorted((k, id(v)) for k, v in list(self._metadata.items())))
        hashdata = (
            self._integral_type_mod,
            self._subdomain_id,
            hash(self._domain),
            metadata_hashdata,
            id_or_none(self._subdomain_data),
        )
        return hash(hashdata)

    def __eq__(self, other) -> bool:
        """Checks if two `dsu` measures are equal.

        Returns:
            bool: ``True`` if both measures are equal, ``False``
            otherwise.
        """
        return isinstance(other, dsu) and super().__eq__(other)
