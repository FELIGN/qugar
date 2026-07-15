# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------


"""Implicit domains module for QUGaR"""

from qugar.impl.impl_function import (
    ImplicitFunc,
    create_affinely_transformed_functions,
    create_annulus,
    create_box,
    create_constant,
    create_cylinder,
    create_dim_linear,
    create_disk,
    create_ellipse,
    create_ellipsoid,
    create_Fischer_Koch_S,
    create_functions_addition,
    create_functions_intersection,
    create_functions_subtraction,
    create_line,
    create_negative,
    create_plane,
    create_Schoen,
    create_Schoen_FRD,
    create_Schoen_IWP,
    create_Schwarz_Diamond,
    create_Schwarz_Primitive,
    create_sphere,
    create_torus,
)

__all__ = [
    "ImplicitFunc",
    "create_disk",
    "create_sphere",
    "create_ellipse",
    "create_ellipsoid",
    "create_cylinder",
    "create_annulus",
    "create_torus",
    "create_constant",
    "create_plane",
    "create_line",
    "create_dim_linear",
    "create_box",
    "create_negative",
    "create_functions_addition",
    "create_functions_subtraction",
    "create_functions_intersection",
    "create_affinely_transformed_functions",
    "create_Schoen",
    "create_Schoen_IWP",
    "create_Schoen_FRD",
    "create_Fischer_Koch_S",
    "create_Schwarz_Primitive",
    "create_Schwarz_Diamond",
]
