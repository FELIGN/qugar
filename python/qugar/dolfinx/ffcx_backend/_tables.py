# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""IR-only model of the finite-element basis tables of an integral.

This is the language-backend replacement for :mod:`qugar.dolfinx.fe_table`,
which recovered the same information by parsing FFCx-rendered C text. Every
field here is read directly from the FFCx intermediate representation
(``IntegralIR``) *before* any code is rendered:

* the table name, type and shape come from ``ir.expression.unique_tables`` /
  ``unique_table_types`` and the ``UniqueTableReferenceT`` (``tr``) node;
* the originating Basix element comes from the factorization-graph terminal
  via :func:`ffcx.ir.elementtables.get_modified_terminal_element`;
* derivatives / component / averaging / quadrature id are parsed from the
  table *name*, which is an IR field (``tr.name``) following FFCx's stable
  ``FE#_C#[_D###][_AC|_AF][_F|V]_Q#`` naming convention.

The ``tr`` node additionally exposes ``offset`` and ``block_size`` (the dof
strides of blocked elements) and ``is_piecewise`` / ``ttype`` — everything
needed to emit runtime tabulation and the dof-access expressions without
touching rendered C.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from basix.ufl import _BasixElement as BasixElement
from ffcx.ir.elementtables import get_modified_terminal_element
from ffcx.ir.representation import IntegralIR

from qugar.dolfinx.quadrature_data import QuadratureData

# FFCx table-name grammar: ``FE{id}_C{comp}[_D{derivs}][_AC|_AF][_F|_V]_Q{qid}``.
_NAME_RE = re.compile(r"^FE(\d+)_C(\d+)(.*)_Q(\w+)$")

_CONSTANT_FOR_PTS_TTYPES = ("fixed", "piecewise", "zeros", "ones")


@dataclass
class IRTable:
    """A finite-element basis table of an integral, built from the IR.

    Mirrors the public surface of :class:`qugar.dolfinx.fe_table.FETable`
    so downstream consumers (runtime tabulation, dof-access codegen, custom
    coefficient packing) can use either interchangeably during the
    backend migration.
    """

    name: str
    FE_id: int
    component: int
    derivatives: tuple[int, ...]
    avg: str | None
    element: BasixElement
    integral_type: str
    entity: str
    table_type: str
    # Shape of the FFCx table: (permutations, entities, points, funcs).
    permutations: int
    entities: int
    points: int
    funcs: int
    # Dof strides of (possibly blocked) elements, from the ``tr`` node.
    offset: int
    block_size: int
    dtype: type[np.floating]
    quad_data: QuadratureData

    @property
    def quad_name(self) -> str:
        """Name (FFCx ``_Q`` id) of the associated quadrature."""
        return self.quad_data.name

    @property
    def element_dim(self) -> int:
        """Topological dimension of the table's element."""
        return self.element.cell.topological_dimension

    def is_constant_for_pts(self) -> bool:
        """Whether the table values are constant across quadrature points.

        Such tables (``fixed`` / ``piecewise`` / ``zeros`` / ``ones`` with
        more than one point) keep their static declaration even on the
        runtime-quadrature path. Matches
        :meth:`qugar.dolfinx.fe_table.FETable.is_constant_for_pts`.
        """
        n_pts = self.quad_data.rule.points.shape[0]
        return n_pts > 1 and self.table_type in _CONSTANT_FOR_PTS_TTYPES


def _parse_name(name: str, elem_dim: int) -> tuple[int, int, tuple[int, ...], str | None, str]:
    """Parse ``(FE_id, component, derivatives, avg, quad_id)`` from a table
    name, following FFCx's ``FE#_C#[_D###][_AC|_AF][_F|V]_Q#`` grammar."""
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"Unrecognised FFCx table name: {name!r}")

    fe_id = int(m.group(1))
    component = int(m.group(2))
    extra = m.group(3)
    quad_id = m.group(4)

    derivatives: tuple[int, ...] = ()
    avg: str | None = None
    if extra:
        assert extra[0] == "_"
        for opt in extra[1:].split("_"):
            if opt and opt[0] == "D":
                derivatives = tuple(int(i) for i in opt[1:])
            elif opt == "AC":
                avg = "cell"
            elif opt == "AF":
                avg = "facet"
            elif opt in ("F", "V"):
                pass  # entity dimension marker; entity_type already known
            else:
                raise ValueError(f"Invalid FE table option {opt!r} in {name!r}")

    if not derivatives or all(d == 0 for d in derivatives):
        derivatives = (0,) * elem_dim

    return fe_id, component, derivatives, avg, quad_id


def _table_name_to_tr_and_element(ir_itg: IntegralIR):
    """Harvest ``name -> (tr, element)`` from the factorization graph.

    Each terminal node carries its ``UniqueTableReferenceT`` (``node["tr"]``,
    whose ``.name`` is the C table name) and its ``ModifiedTerminal``
    (``node["mt"]``), from which the Basix element is recovered. Mirrors
    :func:`qugar.dolfinx.fe_table._build_table_name_to_element_map` but also
    keeps the ``tr`` node (for shape / offset / block_size / ttype).
    """
    result: dict[str, tuple[object, BasixElement]] = {}
    for _key, integrand in ir_itg.expression.integrand.items():
        for node in integrand["factorization"].nodes.values():
            tr = node.get("tr")
            mt = node.get("mt")
            if tr is None or mt is None:
                continue
            mte = get_modified_terminal_element(mt)
            if mte is None:
                continue
            result[tr.name] = (tr, mte.element)
    return result


def extract_ir_tables(
    ir_itg: IntegralIR,
    quad_data_by_name: dict[str, QuadratureData],
    real_dtype: npt.DTypeLike,
) -> list[IRTable]:
    """Build the list of :class:`IRTable` for an integral, from the IR only.

    Args:
        ir_itg: FFCx intermediate representation of the integral.
        quad_data_by_name: Map from FFCx quadrature id to its data
            (see :func:`qugar.dolfinx.quadrature_data.extract_quadrature_data`).
        real_dtype: Real (geometry) dtype in which FFCx declares the tables.

    Returns:
        The tables, ordered reproducibly across Python sessions by hashing
        the associated element (matching ``fe_table._sort_FE_tables``).
    """
    expr = ir_itg.expression
    table_types_by_cell = expr.unique_table_types
    assert len(table_types_by_cell) == 1, (
        "qugar supports a single cell type per integral"
    )
    cell = next(iter(table_types_by_cell))
    table_types = table_types_by_cell[cell]
    tables = expr.unique_tables[cell]

    tr_and_elem = _table_name_to_tr_and_element(ir_itg)
    integral_type = expr.integral_type
    entity_type = expr.entity_type
    real_np = np.dtype(real_dtype).type

    out: list[IRTable] = []
    for name in sorted(tables):
        tr, element = tr_and_elem[name]
        perms, entities, points, funcs = (int(s) for s in np.asarray(tr.values).shape)
        elem_dim = element.cell.topological_dimension
        fe_id, component, derivatives, avg, quad_id = _parse_name(name, elem_dim)
        out.append(
            IRTable(
                name=name,
                FE_id=fe_id,
                component=component,
                derivatives=derivatives,
                avg=avg,
                element=element,
                integral_type=integral_type,
                entity=entity_type,
                table_type=table_types[name],
                permutations=perms,
                entities=entities,
                points=points,
                funcs=funcs,
                offset=int(tr.offset),
                block_size=int(tr.block_size),
                dtype=real_np,
                quad_data=quad_data_by_name[quad_id],
            )
        )

    out.sort(key=lambda t: int(hashlib.sha1(str(t.element).encode()).hexdigest(), 32))
    return out
