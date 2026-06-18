# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------


"""Custom DOLFINx forms for runtime quadratures"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

# Note: the unfitted-boundary normal terminal is lowered inside qugar's FFCx
# language backend (qugar.dolfinx.ffcx_backend), on each backend's own access
# object -- no global FFCx monkeypatch is needed.

# Patch DOLFINx so its stock assemblers and high-level solvers
# (``dolfinx.fem.assemble_*``, ``dolfinx.fem.petsc.LinearProblem`` /
# ``NonlinearProblem``, ...) transparently use qugar's runtime-quadrature
# kernels and custom coefficients whenever a form lives on an unfitted
# mesh. This replaces qugar's previously bespoke ``LinearProblem`` /
# ``NonlinearProblem`` classes: users now use the stock DOLFINx ones
# directly. See qugar.dolfinx._assembly_patches.
from qugar.dolfinx._assembly_patches import apply_patches as _apply_assembly_patches

_apply_assembly_patches()

from qugar.dolfinx.boundary import UnfittedNormal, dsu
from qugar.dolfinx.forms import CustomForm, form_custom

__all__ = ["CustomForm", "UnfittedNormal", "dsu", "form_custom"]
