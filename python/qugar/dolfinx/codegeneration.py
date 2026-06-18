# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Generation of FFCx code for runtime-quadrature integrals.

Code generation is delegated to qugar's FFCx language backend
(:mod:`qugar.dolfinx.ffcx_backend`, selected via ``options["language"]``),
which owns the runtime-quadrature dual kernel and lowers the unfitted-boundary
normal on its own backend access object. The per-cell coefficient packer's
integral data is rebuilt from the IR (no rendered-C parsing)."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import re

import ffcx.codegeneration.codegeneration
import numpy.typing as npt
from ffcx.analysis import UFLData
from ffcx.codegeneration.codegeneration import CodeBlocks
from ffcx.ir.representation import DataIR

from qugar.dolfinx.integral_data import IntegralData, extract_integral_data_from_ir

_QUGAR_FFCX_LANGUAGE = "qugar.dolfinx.ffcx_backend"


def _modify_header(code_blocks: CodeBlocks) -> CodeBlocks:
    """Add a ``#include <stddef.h>`` to the generated header (needed for the
    ``ptrdiff_t`` used by the dispatch wrapper)."""
    assert len(code_blocks[0]) == 1

    header = code_blocks[0][0][1]
    match = re.search(r"#include <ufcx.h>", header)
    assert match

    new_header = header[: match.start()] + "#include <stddef.h>\n" + header[match.start() :]
    code_blocks[0][0] = (code_blocks[0][0][0], new_header)
    return code_blocks


def generate_code(
    ufl_data: UFLData,
    ir: DataIR,
    ffcx_options: dict[str, int | float | npt.DTypeLike],
) -> tuple[CodeBlocks, list[IntegralData]]:
    """Generate the runtime-quadrature kernels and the per-integral data.

    Generation goes through qugar's FFCx language backend
    (``qugar.dolfinx.ffcx_backend.integral.generator``); the per-cell
    coefficient packer's :class:`~qugar.dolfinx.integral_data.IntegralData` is
    rebuilt from the IR.

    Args:
        ufl_data: UFL (analysis) data holding the form data objects.
        ir: FFCx Intermediate Representation (elements, forms, integrals, ...).
        ffcx_options: FFCx options for generating the code.

    Returns:
        The generated code blocks and the list of integral data needed to pack
        the runtime custom coefficients.
    """
    opts = dict(ffcx_options)
    opts["language"] = _QUGAR_FFCX_LANGUAGE
    code_blocks, _suffixes = ffcx.codegeneration.codegeneration.generate_code(ir, opts)
    code_blocks = _modify_header(code_blocks)

    itg_datas = [
        extract_integral_data_from_ir(ufl_data, ir, itg_ir, ffcx_options)
        for itg_ir in ir.integrals
    ]
    return code_blocks, itg_datas
