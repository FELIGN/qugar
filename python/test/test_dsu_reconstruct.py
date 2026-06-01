# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Regression test for issue #3: ``dsu`` losing its subclass
identity (and therefore the Nanson correction in ``__rmul__``) when
restricted via ``ds(subdomain_id)`` or summed via ``ds(a) + ds(b)``.

The bug is purely in UFL's subclassing contract: ``ufl.Measure.__call__``
delegates to ``ufl.Measure.reconstruct`` which hard-codes
``Measure(...)`` rather than ``type(self)(...)``. The fix is a thin
``reconstruct`` override in :py:class:`dsu`.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import dolfinx
import dolfinx.mesh
import numpy as np
import ufl

from qugar.dolfinx import dsu


def _mesh():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


def test_call_preserves_subclass():
    """``ds(id)`` must return a ``dsu``, not a plain Measure."""
    mesh = _mesh()
    ds = dsu(domain=mesh)

    restricted = ds(2)

    assert isinstance(restricted, dsu), (
        f"Expected dsu, got {type(restricted).__name__}"
    )
    assert restricted.subdomain_id() == 2


def test_call_preserves_nanson_correction():
    """Restricting carries over ``_measure_complement`` (Nanson term)."""
    mesh = _mesh()
    ds = dsu(domain=mesh)

    restricted = ds(2)

    # Same domain → identical Nanson term (same UFL object).
    assert restricted._measure_complement is ds._measure_complement
    assert restricted._integral_type_mod == ds._integral_type_mod
    assert restricted._measure_name == ds._measure_name


def _integrand_contains(integrand, target):
    """Return True if ``target`` appears as a subexpression of
    ``integrand``. Used to confirm the Nanson term is injected."""
    if integrand is target:
        return True
    return any(_integrand_contains(op, target) for op in integrand.ufl_operands)


def test_rmul_applies_nanson_after_restriction():
    """``f * ds(id)`` must still inject ``_measure_complement`` and
    the Constant identity must be preserved (cache stability)."""
    mesh = _mesh()
    ds = dsu(domain=mesh)

    f = dolfinx.fem.Constant(mesh, 1.0)
    form = f * ds(3)
    integrals = form.integrals()
    assert len(integrals) == 1
    integral = integrals[0]
    assert integral.subdomain_id() == 3
    assert _integrand_contains(integral.integrand(), ds._measure_complement), (
        "Nanson `_measure_complement` term missing from integrand after ds(3)"
    )

    # Re-calling ds(3) must produce a form with the same signature
    # (FFCx JIT cache stability).
    form_again = f * ds(3)
    assert form.signature() == form_again.signature()


def test_tuple_subdomain_id_preserves_subclass():
    """Tuple subdomain ids are spread inside ``Measure.__rmul__`` via
    ``self.reconstruct(subdomain_id=d)`` — every per-id integrand must
    contain the Nanson term."""
    mesh = _mesh()
    ds = dsu(domain=mesh)

    f = dolfinx.fem.Constant(mesh, 1.0)
    form = f * ds((1, 2))
    integrals = form.integrals()
    assert len(integrals) == 2
    assert {integ.subdomain_id() for integ in integrals} == {1, 2}
    for integ in integrals:
        assert _integrand_contains(integ.integrand(), ds._measure_complement), (
            f"Nanson term missing from tuple-id integrand "
            f"(subdomain_id={integ.subdomain_id()})"
        )


def test_measure_sum_preserves_nanson():
    """``ds(1) + ds(2)`` builds a MeasureSum whose elements must remain
    ``dsu`` so that each integral gets the Nanson term."""
    mesh = _mesh()
    ds = dsu(domain=mesh)

    measure_sum = ds(1) + ds(2)
    assert isinstance(measure_sum, ufl.measure.MeasureSum)
    for m in measure_sum._measures:
        assert isinstance(m, dsu), (
            f"MeasureSum element should be dsu, got {type(m).__name__}"
        )

    f = dolfinx.fem.Constant(mesh, 1.0)
    form = f * measure_sum
    integrals = form.integrals()
    assert len(integrals) == 2
    for integ in integrals:
        assert _integrand_contains(integ.integrand(), ds._measure_complement), (
            f"Nanson term missing from MeasureSum integrand "
            f"(subdomain_id={integ.subdomain_id()})"
        )


def _placeholder_points(measure):
    """The placeholder quadrature points stored in the measure metadata."""
    return measure.metadata()["quadrature_points"]


def test_placeholder_points_differ_by_degree():
    """FFCx names a quadrature from ``sha1(points)`` only, so two ``dsu``
    integrals of different degree must carry different placeholder points
    — otherwise they collide on the same name and the wrong degree wins
    (silently) in ``extract_quadrature_data``."""
    mesh = _mesh()

    pts2 = _placeholder_points(dsu(domain=mesh, degree=2))
    pts4 = _placeholder_points(dsu(domain=mesh, degree=4))

    assert pts2 != pts4, "dsu placeholder points must differ across degrees"


def test_placeholder_points_stable_for_equal_degree():
    """Equal degree (and equal dimension) must yield identical placeholder
    points so equivalent forms keep a stable FFCx JIT signature."""
    mesh = _mesh()

    assert _placeholder_points(dsu(domain=mesh, degree=3)) == _placeholder_points(
        dsu(domain=mesh, degree=3)
    )
    # The ``degree=None`` fallback must also be deterministic.
    assert _placeholder_points(dsu(domain=mesh)) == _placeholder_points(dsu(domain=mesh))


def test_reconstruct_preserves_degree_placeholder_points():
    """``dsu(degree=N)(subdomain_id)`` must keep the degree-N placeholder
    points, not fall back to the degree=None ones.

    Without this, ``dsu(degree=2)(1)`` and ``dsu(degree=5)(2)`` would both
    get degree=None placeholder points (same hash), triggering the
    collision guard in ``extract_quadrature_data`` with a misleading
    error even though the user explicitly specified distinct degrees.
    """
    mesh = _mesh()

    for degree in (2, 5):
        direct = _placeholder_points(dsu(domain=mesh, degree=degree))
        restricted = _placeholder_points(dsu(domain=mesh, degree=degree)(3))
        assert direct == restricted, (
            f"dsu(degree={degree})(subdomain_id) must preserve degree-specific "
            "placeholder points"
        )


def test_subdomain_data_is_carried_through():
    """``ds(id)`` with ``subdomain_data`` set on the original measure
    keeps the data — this is the natural call pattern from issue #3."""
    mesh = _mesh()
    cell_indices = np.array([0, 1], dtype=np.int32)
    cell_values = np.array([7, 7], dtype=np.int32)
    tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cell_indices, cell_values)

    ds = dsu(domain=mesh, subdomain_data=tags)
    restricted = ds(7)

    assert isinstance(restricted, dsu)
    assert restricted.subdomain_data() is tags
