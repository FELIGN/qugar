# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""qugar's FFCx language backend is selectable via options["language"].

FFCx's ``get_language`` imports the module path verbatim, so setting
``options["language"] = "qugar.dolfinx.ffcx_backend"`` routes code generation
through qugar's backend. At this stage the integral generator still delegates
to the stock C backend, so the generated code must match exactly; later
stages replace it with qugar's runtime-quadrature kernel.
"""

import basix.ufl
import ffcx.options
import ufl
from ffcx.analysis import analyze_ufl_objects
from ffcx.codegeneration.codegeneration import generate_code
from ffcx.ir.representation import compute_ir

_QUGAR_BACKEND = "qugar.dolfinx.ffcx_backend"


def _generate(language):
    tri = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(tri, basix.ufl.element("Lagrange", "triangle", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * ufl.dx

    opts = ffcx.options.get_options()
    opts["language"] = language
    analysis = analyze_ufl_objects([a], opts["scalar_type"])
    ir = compute_ir(analysis, {}, "plugin", opts, False)
    code_blocks, suffixes = generate_code(ir, opts)
    return code_blocks, suffixes


def test_backend_generates_runtime_quadrature_dual_kernel():
    """FFCx resolves the qugar backend and its integral generator emits the
    runtime-quadrature dual kernel (static _original + runtime _custom +
    dispatch wrapper + shim/loader prologue)."""
    q_blocks, q_suffixes = _generate(_QUGAR_BACKEND)

    assert q_suffixes == (".h", ".c")
    impl = q_blocks.integrals[0][1]
    for marker in (
        "_original(",
        "_custom(",
        "load_points_",
        "w_custom_offset",
        "qugar_register_element",
        "qugar_get_scratch",
        "ufcx_integral integral_",
    ):
        assert marker in impl, f"missing {marker!r} in generated kernel"
