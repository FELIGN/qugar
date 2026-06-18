# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Structural checks for the IR-only FE-table extractor
(:func:`qugar.dolfinx.ffcx_backend.extract_ir_tables`), which recovers every
table's metadata from the FFCx IR (no rendered-C parsing). End-to-end
correctness is covered by the full assembly suite; this pins the extractor's
shape for a range of elements as a fast unit test.
"""

import basix
import basix.ufl
import ffcx.options
import numpy as np
import pytest
import ufl
from ffcx.analysis import analyze_ufl_objects
from ffcx.ir.representation import compute_ir

from qugar.dolfinx.ffcx_backend import extract_ir_tables
from qugar.dolfinx.quadrature_data import extract_quadrature_data

_TRI = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
_TET = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))


def _poisson(mesh, cell, degree):
    V = ufl.FunctionSpace(mesh, basix.ufl.element("Lagrange", cell, degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return (ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * ufl.dx


def _vector_mass(mesh, cell, degree):
    V = ufl.FunctionSpace(mesh, basix.ufl.element("Lagrange", cell, degree, shape=(2,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return ufl.inner(u, v) * ufl.dx


def _taylor_hood(mesh, cell):
    el = basix.ufl.mixed_element(
        [
            basix.ufl.element("Lagrange", cell, 2, shape=(2,)),
            basix.ufl.element("Lagrange", cell, 1),
        ]
    )
    W = ufl.FunctionSpace(mesh, el)
    up, vq = ufl.TrialFunction(W), ufl.TestFunction(W)
    u, p = ufl.split(up)
    v, q = ufl.split(vq)
    return (ufl.inner(ufl.grad(u), ufl.grad(v)) + p * q) * ufl.dx


FORMS = {
    "P2-tri-poisson": (lambda: _poisson(_TRI, "triangle", 2), 2, True),
    "P3-tet-poisson": (lambda: _poisson(_TET, "tetrahedron", 3), 3, True),
    "P2-vec-tri-mass": (lambda: _vector_mass(_TRI, "triangle", 2), 2, False),
    "TH-tri-mixed": (lambda: _taylor_hood(_TRI, "triangle"), 2, True),
}


@pytest.mark.parametrize("name", list(FORMS))
def test_ir_tables_structure(name):
    form_fn, tdim, has_grad = FORMS[name]
    opts = ffcx.options.get_options()
    analysis = analyze_ufl_objects([form_fn()], opts["scalar_type"])
    ir = compute_ir(analysis, {}, "test", opts, False)
    quads = extract_quadrature_data(analysis, opts)

    for ir_itg in ir.integrals:
        tables = extract_ir_tables(ir_itg, quads, np.float64)
        assert tables, "expected at least one FE table"
        for t in tables:
            assert t.name.startswith("FE")
            assert len(t.derivatives) == t.element_dim
            assert t.funcs >= 1 and t.points >= 1
            assert t.quad_data is quads[t.quad_name]
        if has_grad:
            assert any(any(d > 0 for d in t.derivatives) for t in tables), (
                "a gradient form should produce derivative tables"
            )
