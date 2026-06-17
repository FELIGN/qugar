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

Work in progress (see the FEniCSx 0.11 port). This package incrementally
replaces the text-parsing code-generation path
(:mod:`qugar.dolfinx.codegeneration` + :mod:`qugar.dolfinx._kernel_body`)
with a proper FFCx language backend built on the intermediate
representation.
"""

from qugar.dolfinx.ffcx_backend._tables import IRTable, extract_ir_tables

__all__ = ["IRTable", "extract_ir_tables"]
