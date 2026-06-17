# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""qugar's FFCx language backend for runtime-quadrature kernels.

Selected via ``options["language"] = "qugar.dolfinx.ffcx_backend"``. It
exposes the FFCx language-backend protocol (``integral`` / ``form`` /
``expression`` / ``file`` generators + a ``Formatter`` + ``file.suffixes``);
only the integral generator is qugar-specific, the rest delegate to the
stock C backend.

Work in progress (FEniCSx 0.11 port): incrementally replacing the
text-parsing code-generation path (:mod:`qugar.dolfinx.codegeneration` +
:mod:`qugar.dolfinx._kernel_body`).
"""

from qugar.dolfinx.ffcx_backend import expression, file, form, integral
from qugar.dolfinx.ffcx_backend._generator import QugarIntegralGenerator
from qugar.dolfinx.ffcx_backend._tables import IRTable, extract_ir_tables
from qugar.dolfinx.ffcx_backend.formatter import Formatter

__all__ = [
    "Formatter",
    "IRTable",
    "QugarIntegralGenerator",
    "expression",
    "extract_ir_tables",
    "file",
    "form",
    "integral",
]
