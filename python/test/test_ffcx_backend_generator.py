# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Structural tests for QugarIntegralGenerator (the runtime-quadrature AST).

The generator must transform FFCx's standard integral AST so that, for the
point-varying basis tables, the static declarations are removed, the
quadrature loop bound becomes the runtime ``n_pts_Q<quad>`` symbol, and the
4-D table accesses are flattened to the 1-D runtime-buffer form — while
constant-for-points tables keep their static declaration and 4-D access.
"""

import basix
import basix.ufl
import ffcx.codegeneration.lnodes as L
import ffcx.options
import numpy as np
import pytest
import ufl
from ffcx.analysis import analyze_ufl_objects
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.ir.representation import compute_ir

from qugar.dolfinx.ffcx_backend._generator import QugarIntegralGenerator, _iter_nodes
from qugar.dolfinx.ffcx_backend._tables import extract_ir_tables
from qugar.dolfinx.quadrature_data import extract_quadrature_data

_TRI = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))


def _poisson(degree):
    V = ufl.FunctionSpace(_TRI, basix.ufl.element("Lagrange", "triangle", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return (ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * ufl.dx


def _vector_mass(degree):
    V = ufl.FunctionSpace(_TRI, basix.ufl.element("Lagrange", "triangle", degree, shape=(2,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return ufl.inner(u, v) * ufl.dx


FORMS = {
    "P2-poisson": lambda: _poisson(2),
    "P3-poisson": lambda: _poisson(3),
    "P2-vec-mass": lambda: _vector_mass(2),
}


def _prepare(form):
    opts = ffcx.options.get_options()
    analysis = analyze_ufl_objects([form], opts["scalar_type"])
    ir = compute_ir(analysis, {}, "gen", opts, False)
    ir_itg = ir.integrals[0]
    domain = next(iter(ir_itg.expression.unique_table_types))
    quads = extract_quadrature_data(analysis, opts)
    tables = extract_ir_tables(ir_itg, quads, np.float64)
    return opts, ir_itg, domain, tables


@pytest.mark.parametrize("name", list(FORMS))
def test_runtime_kernel_ast(name):
    opts, ir_itg, domain, tables = _prepare(FORMS[name]())
    varying = {t.name for t in tables if not t.is_constant_for_pts()}
    funcs_by_name = {t.name: t.funcs for t in tables}
    assert varying, "test form should have point-varying tables"

    gen = QugarIntegralGenerator(ir_itg, FFCXBackend(ir_itg, opts), tables)
    ast = gen.generate(domain)
    nodes = list(_iter_nodes(ast))

    # No static decls left for varying tables or the weights arrays.
    for n in nodes:
        if isinstance(n, L.ArrayDecl):
            nm = getattr(n.symbol, "name", "")
            assert nm not in varying, f"static decl of varying table {nm} remains"
            assert not nm.startswith("weights_"), f"static weights decl {nm} remains"

    # Quadrature loops use the runtime bound; accesses to varying tables are 1-D.
    iq_loops = [n for n in nodes if isinstance(n, L.ForRange)
                and getattr(n.index, "name", None) == "iq"]
    assert iq_loops
    for lp in iq_loops:
        assert isinstance(lp.end, L.Symbol) and lp.end.name.startswith("n_pts_Q")

    for n in nodes:
        if isinstance(n, L.ArrayAccess):
            nm = getattr(n.array, "name", None)
            if nm in varying:
                assert len(n.indices) == 1, f"{nm} access not flattened"

    # Renders to valid C mentioning the runtime bound; sanity vs. stock.
    rendered = Formatter(opts["scalar_type"])(ast)
    assert "n_pts_Q" in rendered
    for nm in varying:
        assert f"const {nm}" not in rendered  # no leftover static table decl
    assert funcs_by_name  # used implicitly above


def test_stock_generator_keeps_static_tables():
    """Contrast: the stock generator still emits static tables (used by the
    _original non-cut kernel)."""
    opts, ir_itg, domain, tables = _prepare(_poisson(2))
    varying = {t.name for t in tables if not t.is_constant_for_pts()}
    ast = IntegralGenerator(ir_itg, FFCXBackend(ir_itg, opts)).generate(domain)
    decl_names = {getattr(n.symbol, "name", "") for n in _iter_nodes(ast)
                  if isinstance(n, L.ArrayDecl)}
    assert varying & decl_names, "stock generator should declare varying tables statically"
