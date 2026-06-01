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

# Apply in-place fixes to known upstream dolfinx bugs before exposing any
# qugar API that depends on the patched functions. See
# qugar.dolfinx._dolfinx_patches for the list of patches and the
# upstream issues / PRs each one tracks.
from qugar.dolfinx._dolfinx_patches import apply_patches as _apply_dolfinx_patches

_apply_dolfinx_patches()

# Teach FFCx's code-generation backend to lower the unfitted-boundary
# normal terminal (see qugar.dolfinx._ffcx_patches).
from qugar.dolfinx._ffcx_patches import apply_patches as _apply_ffcx_patches

_apply_ffcx_patches()

from qugar.dolfinx.boundary import dsu, dsu_normal
from qugar.dolfinx.forms import CustomForm, form_custom

__all__ = ["CustomForm", "dsu", "dsu_normal", "form_custom"]


from qugar.utils import has_PETSc

if has_PETSc:
    from qugar.dolfinx.petsc import LinearProblem, NonlinearProblem

    __all__ += ["LinearProblem", "NonlinearProblem"]
