# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Validate the IR-only FE-table extractor (qugar.dolfinx.ffcx_backend) against
the legacy C-text-parsing extractor (qugar.dolfinx.fe_table).

Both must produce the same set of tables — name, element, derivatives,
component, shape, constant-for-points classification and averaging — for a
range of elements. The IR extractor is the foundation of the new FFCx
language backend, so this pins its equivalence with the established path.
"""

import basix
import basix.ufl
import ffcx.codegeneration.codegeneration
import ffcx.options
import numpy as np
import pytest
import ufl
from ffcx.analysis import analyze_ufl_objects
from ffcx.ir.representation import compute_ir

from qugar.dolfinx.fe_table import extract_FE_tables
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
    "P1-tri-poisson": lambda: _poisson(_TRI, "triangle", 1),
    "P2-tri-poisson": lambda: _poisson(_TRI, "triangle", 2),
    "P3-tri-poisson": lambda: _poisson(_TRI, "triangle", 3),
    "P2-vec-tri-mass": lambda: _vector_mass(_TRI, "triangle", 2),
    "P2-tet-poisson": lambda: _poisson(_TET, "tetrahedron", 2),
    "TH-tri-mixed": lambda: _taylor_hood(_TRI, "triangle"),
}


def _key(t):
    """Comparable fingerprint of a table (legacy FETable or new IRTable)."""
    return (
        t.name,
        str(t.element),
        tuple(t.derivatives),
        t.component,
        (t.permutations, t.entities, t.points, t.funcs),
        t.is_constant_for_pts(),
        t.avg,
    )


@pytest.mark.parametrize("name", list(FORMS))
def test_ir_tables_match_legacy(name):
    form = FORMS[name]()
    opts = ffcx.options.get_options()
    analysis = analyze_ufl_objects([form], opts["scalar_type"])
    ir = compute_ir(analysis, {}, "test", opts, False)
    code_blocks, _ = ffcx.codegeneration.codegeneration.generate_code(ir, opts)
    quads = extract_quadrature_data(analysis, opts)

    for i, ir_itg in enumerate(ir.integrals):
        _, impl = code_blocks.integrals[i]
        cell = next(iter(ir_itg.expression.unique_table_types))
        tdim = len(basix.cell.topology(cell)) - 1

        legacy = sorted(_key(t) for t in extract_FE_tables(impl, ir_itg, quads, tdim))
        new = sorted(_key(t) for t in extract_ir_tables(ir_itg, quads, np.float64))

        assert new == legacy, f"{name} integral #{i}"
        assert new, "expected at least one table"
