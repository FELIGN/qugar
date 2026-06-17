# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Integral generator for qugar's FFCx language backend.

Selected via ``options["language"] = "qugar.dolfinx.ffcx_backend"`` (FFCx's
``get_language`` imports the module path verbatim). This is the seam where
qugar will emit its runtime-quadrature dual kernel (``_original`` static +
``_custom`` runtime + dispatch wrapper) and lower the unfitted-boundary
normal *inside the backend* — removing the need for the global FFCx
monkeypatch.

For now it delegates to the stock C generator; the qugar integral assembly
is ported in the following stages of the backend migration.
"""

from ffcx.codegeneration.C.integral import generator

__all__ = ["generator"]
