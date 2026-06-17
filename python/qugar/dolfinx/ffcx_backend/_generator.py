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
import basix.ufl
import ffcx.codegeneration.lnodes as L
from ffcx.codegeneration.integral_generator import IntegralGenerator

_QUAD_ID_RE = re.compile(r"_Q(\w+)$")
_WEIGHTS_RE = re.compile(r"^weights_(\w+)$")


def _resolve_mixed_component(element, flat_component):
    """Drill into the sub-element of a ``_MixedElement`` that owns
    ``flat_component``; pass non-mixed elements through. Recursive."""
    if not isinstance(element, basix.ufl._MixedElement):
        return element, flat_component
    acc = 0
    for sub in element.sub_elements:
        if flat_component < acc + sub.reference_value_size:
            return _resolve_mixed_component(sub, flat_component - acc)
        acc += sub.reference_value_size
    raise ValueError(f"flat_component {flat_component} out of range")


def _table_value_size(element, component) -> int:
    """Value-size stride ``vs`` of a table's basix block: 1 for a blocked
    (vector/tensor) element (the scalar block is tabulated and the block
    expansion is interleaved), else the element's reference value size."""
    el, _local = _resolve_mixed_component(element, component)
    block_size = getattr(el, "block_size", 1)
    return 1 if block_size > 1 else int(el.reference_value_size)


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

    def __init__(self, ir, backend, tables, strip_all_tables: bool = False):
        """Args:
            ir: FFCx integral IR.
            backend: ``FFCXBackend`` built from ``ir``.
            tables: the integral's tables (``IRTable`` or the legacy
                ``FETable``; only ``.name``, ``.funcs`` and
                ``.is_constant_for_pts()`` are used).
            strip_all_tables: when ``True`` the constant-for-points table
                declarations are also removed from the body (they are then
                re-emitted by the caller's prologue); when ``False`` they are
                kept in the body. Their accesses are never rewritten.
        """
        super().__init__(ir, backend)
        self._tables = {t.name: t for t in tables}
        self._vs = {t.name: _table_value_size(t.element, t.component) for t in tables}
        self._varying: set[str] = {
            t.name for t in tables if not t.is_constant_for_pts()
        }
        if strip_all_tables:
            self._strip_decl_names = set(self._tables)
        else:
            self._strip_decl_names = set(self._varying)
        self._has_perms = ir.expression.integral_type == "interior_facet"

    def generate(self, domain: basix.CellType):
        """Generate the runtime-quadrature ``tabulate_tensor`` body AST."""
        parts = super().generate(domain)
        self._strip_static_decls(parts)
        self._flatten_table_accesses(parts)
        self._dynamic_loop_bounds(parts)
        self._inline_preloop_into_loops(parts)
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
                    if name in self._strip_decl_names or _WEIGHTS_RE.match(name):
                        continue
                kept.append(st)
            stmts[:] = kept

    def _flatten_table_accesses(self, parts) -> None:
        """Rewrite 4-D varying-table accesses to the 1-D buffer form
        ``FE..[funcs * iq + dof]`` (mutating each ``ArrayAccess`` in place so
        every parent expression sees the change)."""
        # Collect the target accesses before mutating: rewriting an access
        # introduces a new inner ``ArrayAccess`` on the same FE symbol, which
        # the live walker would otherwise re-visit.
        targets = [
            n for n in _iter_nodes(parts)
            if isinstance(n, L.ArrayAccess)
            and getattr(n.array, "name", None) in self._varying
        ]
        for node in targets:
            name = node.array.name
            idx = tuple(node.indices)
            assert len(idx) == 4, f"expected 4-D table access for {name}"
            perm, _entity, iq_expr, dof_expr = idx

            funcs = self._tables[name].funcs
            one_d = L.Add(L.Mul(L.LiteralInt(funcs), iq_expr), dof_expr)
            # Strided access into the basix block (the buffer pointer carries
            # the derivative/value-axis offset): FE[vs * (funcs*iq + dof)].
            # For scalar (vs == 1) this is the plain FE[funcs*iq + dof]; for
            # blocked/vector elements (vs > 1) it indexes the interleaved
            # block directly, so no per-cell repack copy is needed.
            vs = self._vs[name]
            if vs != 1:
                one_d = L.Mul(L.LiteralInt(vs), one_d)

            # Interior-facet two-sided tables are declared as FE..[2] (one
            # buffer per side); FFCx indexes them with quadrature_permutation
            # [side]. Map FE..[quadrature_permutation[s]][.][iq][dof] to
            # FE..[s][funcs*iq + dof]. Single-buffer tables (perm == 0) and
            # cell/exterior-facet accesses become FE..[funcs*iq + dof].
            side = _perm_side(perm) if self._has_perms else None
            if side is not None:
                node.array = L.ArrayAccess(node.array, (L.LiteralInt(side),))
            node.indices = (one_d,)

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

    def _inline_preloop_into_loops(self, parts) -> None:
        """Move the pre-loop band (cell-affine setup + the piecewise
        partition) into the quadrature loop body, mirroring the legacy
        ``inline_pre_loop_into_loops``.

        FFCx hoists cellwise-constant work before the quadrature loop. On the
        runtime path the per-point unfitted normal is lowered there too (the
        terminal is statically cellwise-constant from FFCx's viewpoint but
        reads ``normals_<quad>[tdim*iq + i]``), so it must sit *inside* the
        loop for ``iq`` to be in scope and for the normal to be re-read at
        every point.
        """
        stmts = getattr(parts, "statements", None)
        if not isinstance(stmts, list):
            return
        loop_positions = [
            i for i, s in enumerate(stmts)
            if isinstance(s, L.ForRange) and getattr(s.index, "name", None) == "iq"
        ]
        if not loop_positions:
            return
        first = loop_positions[0]
        preloop = list(stmts[:first])
        if not preloop:
            return
        for i in loop_positions:
            _prepend_to_body(stmts[i], preloop)
        del stmts[:first]

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


def _perm_side(perm) -> int | None:
    """Return the side index ``s`` of a ``quadrature_permutation[s]`` table
    index (interior-facet two-sided tables), or ``None`` otherwise."""
    if isinstance(perm, L.ArrayAccess) and getattr(perm.array, "name", None) == (
        "quadrature_permutation"
    ):
        k = perm.indices[0]
        if isinstance(k, L.LiteralInt):
            return k.value
    return None


def _prepend_to_body(loop, stmts: list) -> None:
    """Prepend ``stmts`` to the body of a ``ForRange`` loop, supporting both
    ``StatementList`` and bare-list body representations."""
    body = loop.body
    if hasattr(body, "statements") and isinstance(body.statements, list):
        body.statements[:0] = stmts
    elif isinstance(body, list):
        body[:0] = stmts
    else:
        loop.body = L.StatementList(list(stmts) + [body])
