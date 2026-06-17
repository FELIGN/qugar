# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""The runtime-quadrature codegen context is derivable from the FFCx IR alone.

This pins the feasibility of the FFCx language backend's narrower input
(integral.generator gets only ir/domain/options, not the UFL analysis):
tdim and the per-quadrature data (id + unfitted-boundary flag) come straight
from the IR. The unfitted-boundary detection is exercised against fitted
forms here (must be False); the True case is covered by the unfitted-boundary
integration tests.
"""

import basix
import basix.ufl
import ffcx.options
import numpy as np
import pytest
import ufl
from ffcx.analysis import analyze_ufl_objects
from ffcx.ir.representation import compute_ir

from qugar.dolfinx.ffcx_backend._context import integral_tdim, quad_data_from_ir

_TRI = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
_TET = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))


def _ir(form):
    opts = ffcx.options.get_options()
    analysis = analyze_ufl_objects([form], opts["scalar_type"])
    return compute_ir(analysis, {}, "ctx", opts, False)


@pytest.mark.parametrize(
    "mesh,cell,expected_tdim",
    [(_TRI, "triangle", 2), (_TET, "tetrahedron", 3)],
)
def test_tdim_and_quads_from_ir(mesh, cell, expected_tdim):
    V = ufl.FunctionSpace(mesh, basix.ufl.element("Lagrange", cell, 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = (ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * ufl.dx
    ir_itg = _ir(form).integrals[0]

    assert integral_tdim(ir_itg) == expected_tdim

    quads = quad_data_from_ir(ir_itg, np.float64)
    assert quads, "expected at least one quadrature"
    for qid, qdata in quads.items():
        assert qdata.name == qid
        # A plain dx form has no unfitted-boundary normal.
        assert qdata.unfitted_boundary is False
        assert qdata.rule.points.shape[0] >= 1
