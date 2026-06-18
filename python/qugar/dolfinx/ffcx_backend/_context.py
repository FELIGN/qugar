# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Derive, from the FFCx integral IR alone, everything the runtime-quadrature
code generation needs.

The FFCx language-backend entry point is ``integral.generator(ir, domain,
options)`` — a narrower input than qugar's current flow (which also has the
UFL analysis / form metadata via
:func:`qugar.dolfinx.integral_data.extract_integral_data`). This module shows
that narrower input is sufficient:

* ``tdim`` — from the geometric-quantity terminal in the factorization graph
  (same as :func:`qugar.dolfinx.integral_data._get_integral_dimension`);
* the quadrature rules — from ``ir.expression.integrand`` keys
  (``QuadratureRule``), with their FFCx ``_Q`` id via ``rule.id()``;
* the **unfitted-boundary** flag — detected from the presence of the
  :class:`qugar.dolfinx.boundary.UnfittedReferenceNormal` terminal in the
  integrand (the codegen never needs the quadrature ``degree`` or the
  placeholder points, which were the only ``ufl_analysis``-sourced data).
"""

from __future__ import annotations

import numpy.typing as npt
import ufl.domain
import ufl.geometry
from ffcx.ir.representation import IntegralIR

from qugar.dolfinx.boundary import UnfittedReferenceNormal
from qugar.dolfinx.quadrature_data import QuadratureData


def integral_tdim(ir_itg: IntegralIR) -> int:
    """Topological dimension of the integral's domain, from the coordinate
    element of the geometric-quantity terminal in the factorization graph."""
    cells = set()
    for integrand in ir_itg.expression.integrand.values():
        for node in integrand["factorization"].nodes.values():
            mt = node.get("mt")
            if mt is not None and isinstance(mt.terminal, ufl.geometry.GeometricQuantity):
                domain = ufl.domain.extract_unique_domain(mt.terminal)
                assert domain is not None
                cells.add(domain.ufl_cell())
    assert len(cells) == 1, "qugar supports a single cell type per integral"
    return cells.pop().topological_dimension


def _rule_has_unfitted_normal(integrand) -> bool:
    """Whether an integrand's factorization references the unfitted-boundary
    normal terminal (hence needs per-point normals at runtime)."""
    for node in integrand["factorization"].nodes.values():
        mt = node.get("mt")
        if mt is not None and isinstance(mt.terminal, UnfittedReferenceNormal):
            return True
    return False


def quad_data_from_ir(
    ir_itg: IntegralIR, real_dtype: npt.DTypeLike
) -> dict[str, QuadratureData]:
    """Build the per-quadrature data (keyed by FFCx ``_Q`` id) from the IR.

    The ``degree`` is irrelevant to code generation (runtime quadrature), so a
    placeholder is used; ``unfitted_boundary`` is detected from the normal
    terminal; ``rule`` is the FFCx ``QuadratureRule`` carried by the IR.
    """
    quads: dict[str, QuadratureData] = {}
    for (_cell, rule), integrand in ir_itg.expression.integrand.items():
        hash(rule)  # ensure rule.id() is available
        name = rule.id()
        if name not in quads:
            quads[name] = QuadratureData(
                name=name,
                degree=-1,  # unused by codegen
                unfitted_boundary=_rule_has_unfitted_normal(integrand),
                rule=rule,
            )
    return quads
