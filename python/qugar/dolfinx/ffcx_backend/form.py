# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Form generator for qugar's FFCx language backend (delegates to the stock C
backend; qugar customises only the integral generator)."""

from ffcx.codegeneration.C.form import generator

__all__ = ["generator"]
