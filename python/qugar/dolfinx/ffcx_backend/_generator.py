# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Runtime-quadrature integral generator for qugar's FFCx language backend.

:class:`QugarIntegralGenerator` subclasses FFCx's ``IntegralGenerator`` and
post-processes the language-agnostic LNode AST it produces, instead of
parsing the rendered C text (the legacy
:mod:`qugar.dolfinx._kernel_body` path). The transforms mirror, at the AST
level, the three text passes of the custom kernel:

* **strip static tables** — remove the ``const`` declarations of the
  point-varying basis tables and the quadrature-weights array; on the
  runtime path these are filled per cell from ``custom_data`` (the buffer
  fill / loader prologue is emitted by the backend's ``integral`` module).
* **flatten table accesses** — rewrite ``FE..[perm][entity][iq][dof]`` to
  the 1-D ``FE..[funcs * iq + dof]`` form used by the runtime buffers.
* **dynamic loop bounds** — replace the compile-time quadrature-loop upper
  bound with the runtime point count ``n_pts_Q<quad>``.

Constant-for-points tables keep their static declaration and 4-D access.

Working only on typed nodes (``ArrayDecl`` / ``ArrayAccess`` / ``ForRange``
keyed on the ``element_tables`` symbol registry) is far more robust across
FFCx releases than regex over emitted C.
"""

from __future__ import annotations

import re

import basix
import ffcx.codegeneration.lnodes as L
from ffcx.codegeneration.integral_generator import IntegralGenerator

from qugar.dolfinx.ffcx_backend._tables import IRTable

_QUAD_ID_RE = re.compile(r"_Q(\w+)$")
_WEIGHTS_RE = re.compile(r"^weights_(\w+)$")


def _quad_id(symbol_name: str) -> str | None:
    """Return the FFCx quadrature id encoded in a table/weights symbol name."""
    m = _WEIGHTS_RE.match(symbol_name)
    if m:
        return m.group(1)
    m = _QUAD_ID_RE.search(symbol_name)
    return m.group(1) if m else None


def _iter_nodes(node, seen=None):
    """Depth-first iterate every LNode reachable from ``node`` (incl. through
    expression operands), each yielded once."""
    if seen is None:
        seen = set()
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    if not hasattr(node, "__dict__"):
        return
    for val in vars(node).values():
        items = val if isinstance(val, (list, tuple)) else (val,)
        for it in items:
            if isinstance(it, L.LNode):
                yield from _iter_nodes(it, seen)


class QugarIntegralGenerator(IntegralGenerator):
    """``IntegralGenerator`` that emits a runtime-quadrature kernel body."""

    def __init__(self, ir, backend, tables: list[IRTable]):
        super().__init__(ir, backend)
        self._tables: dict[str, IRTable] = {t.name: t for t in tables}
        self._varying: set[str] = {
            t.name for t in tables if not t.is_constant_for_pts()
        }
        self._has_perms = ir.expression.integral_type == "interior_facet"

    def generate(self, domain: basix.CellType):
        """Generate the runtime-quadrature ``tabulate_tensor`` body AST."""
        parts = super().generate(domain)
        self._strip_static_decls(parts)
        self._flatten_table_accesses(parts)
        self._dynamic_loop_bounds(parts)
        return parts

    # -- transforms ---------------------------------------------------------

    def _strip_static_decls(self, parts) -> None:
        """Remove ``const`` decls of varying tables and the weights arrays
        from every statement list / section in the tree."""
        for node in _iter_nodes(parts):
            stmts = getattr(node, "statements", None)
            if not isinstance(stmts, list):
                continue
            kept = []
            for st in stmts:
                if isinstance(st, L.ArrayDecl):
                    name = getattr(st.symbol, "name", "")
                    if name in self._varying or _WEIGHTS_RE.match(name):
                        continue
                kept.append(st)
            stmts[:] = kept

    def _flatten_table_accesses(self, parts) -> None:
        """Rewrite 4-D varying-table accesses to the 1-D buffer form
        ``FE..[funcs * iq + dof]`` (mutating each ``ArrayAccess`` in place so
        every parent expression sees the change)."""
        for node in _iter_nodes(parts):
            if not isinstance(node, L.ArrayAccess):
                continue
            name = getattr(node.array, "name", None)
            if name is None or name not in self._varying:
                continue
            idx = tuple(node.indices)
            assert len(idx) == 4, f"expected 4-D table access for {name}"
            perm, _entity, iq_expr, dof_expr = idx

            if self._has_perms and not _is_literal_zero(perm):
                raise NotImplementedError(
                    "interior-facet permutation table access is handled in a "
                    "later stage of the backend migration"
                )

            funcs = self._tables[name].funcs
            node.indices = (L.Add(L.Mul(L.LiteralInt(funcs), iq_expr), dof_expr),)

    def _dynamic_loop_bounds(self, parts) -> None:
        """Replace the static upper bound of each quadrature loop (``index
        == iq``) with the runtime ``n_pts_Q<quad>`` symbol."""
        for node in _iter_nodes(parts):
            if not isinstance(node, L.ForRange):
                continue
            if getattr(node.index, "name", None) != "iq":
                continue
            quad = self._quad_of_loop(node)
            if quad is None:
                raise RuntimeError("could not resolve the quadrature of a loop")
            node.end = L.Symbol(f"n_pts_Q{quad}", L.DataType.INT)

    def _quad_of_loop(self, loop) -> str | None:
        """Resolve the FFCx quadrature id of a loop from the table/weights
        symbols accessed inside it."""
        for node in _iter_nodes(loop):
            if isinstance(node, L.Symbol):
                quad = _quad_id(node.name)
                if quad is not None:
                    return quad
        return None


def _is_literal_zero(expr) -> bool:
    return isinstance(expr, L.LiteralInt) and expr.value == 0
