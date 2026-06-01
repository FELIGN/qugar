# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Runtime quadrature/geometry store for custom integrals.

The custom coefficients consumed by qugar's runtime-quadrature kernels are
made of two parts: the *standard* DOLFINx coefficients (which depend on the
``Function`` / ``Constant`` values in the form and change between solves)
and the *smuggled geometry* (quadrature points, weights, normals, plus the
cut/empty entity classification and the per-entity offsets), which depends
only on the unfitted domain, the integral type, and the quadrature degree —
**not** on the coefficient values.

This module factors that value-independent part into a :class:`QuadratureStore`
that is computed once per custom integral and reused across every assembly.
Packing the full coefficient array then becomes a cheap two-step operation:

1. build the store (quadrature rule + entity classification + a prebuilt
   ``new_coeffs`` *template* with the geometry and offsets already written and
   the standard-coefficient block left zeroed) — done once; and
2. :meth:`QuadratureStore.pack` — copy the template and overlay the freshly
   packed standard DOLFINx coefficients into its top-left block.

See :mod:`qugar.dolfinx.custom_coefficients` for the builder and
:class:`qugar.dolfinx.forms.CustomForm` for where the store is cached.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found.")

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qugar.quad import CustomQuad, CustomQuadUnfBoundary

#: All coefficient array dtypes qugar supports.
FloatingArray = npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128]

#: Per-integrand quadrature stored in a ``QuadratureStore``: a
#: :class:`CustomQuad` for cells, :class:`CustomQuadUnfBoundary` for
#: unfitted boundaries, or a ``(side0, side1)`` tuple for interior facets.
StoredQuad = CustomQuad | CustomQuadUnfBoundary | tuple[CustomQuad, CustomQuad]


@dataclass(frozen=True)
class QuadratureStore:
    """Value-independent runtime geometry for a single custom integral.

    An instance holds everything needed to assemble a custom integral that
    does *not* depend on the form's coefficient values, so it can be built
    once (the first time the form is packed) and reused on every subsequent
    assembly. The expensive part — generating the cut-cell / cut-facet /
    unfitted-boundary quadratures — happens during construction (and is
    further shared across forms via
    :meth:`qugar.mesh.unfitted_domain_abc.UnfittedDomainABC.get_cached_custom_quadrature`).

    Parameters:
        template (FloatingArray): The ``new_coeffs`` array with the smuggled
            geometry (per-entity points, weights, normals, ...) and the
            relative offsets already written, and the standard-coefficient
            block (the top-left ``[:old_rows, :old_cols]`` region) left
            zeroed. :meth:`pack` overlays the freshly packed standard
            coefficients onto that block.
        old_rows (int): Number of rows of the standard-coefficient block
            (one per integration entity; twice the number of facets for
            interior-facet integrals).
        old_cols (int): Number of columns of the standard-coefficient block
            (the number of columns DOLFINx packs per entity).
        custom_entity_ids (npt.NDArray[np.intp]): Indices, into the form's
            domain entity array, of the entities that carry a custom
            (cut) quadrature.
        empty_entity_ids (npt.NDArray[np.intp]): Indices, into the form's
            domain entity array, of the empty entities.
        custom_quads (dict[Any, StoredQuad]): Map from each integrand's
            quadrature data to its generated :data:`StoredQuad`.
        n_vals_per_entity (npt.NDArray[np.int32]): Number of real-unit slots
            of smuggled data stored per custom entity.
        offsets (npt.NDArray[np.intp]): Absolute start positions (in real
            units) of each custom entity's smuggled data within ``template``.
    """

    template: FloatingArray
    old_rows: int
    old_cols: int
    custom_entity_ids: npt.NDArray[np.intp]
    empty_entity_ids: npt.NDArray[np.intp]
    custom_quads: dict[Any, StoredQuad]
    n_vals_per_entity: npt.NDArray[np.int32]
    offsets: npt.NDArray[np.intp]

    def __post_init__(self) -> None:
        self.template.setflags(write=False)

    def pack(self, old_coeffs: FloatingArray) -> FloatingArray:
        """Builds the full custom-coefficient array for one assembly.

        Copies the prebuilt :attr:`template` (so successive calls return
        independent arrays) and overlays the freshly packed standard DOLFINx
        coefficients ``old_coeffs`` onto its top-left block. The smuggled
        geometry and the offset columns live outside that block, so they are
        preserved untouched.

        Args:
            old_coeffs (FloatingArray): Standard coefficients packed by
                DOLFINx for the non-custom part of this integral.

        Returns:
            FloatingArray: The custom coefficients consumed by the kernel
            (same layout as ``dolfinx.cpp.fem.pack_coefficients``).
        """
        assert old_coeffs.shape == (self.old_rows, self.old_cols), (
            f"old_coeffs shape {old_coeffs.shape} does not match store "
            f"({self.old_rows}, {self.old_cols})"
        )
        new_coeffs = self.template.copy()
        new_coeffs[: self.old_rows, : self.old_cols] = old_coeffs
        return new_coeffs
