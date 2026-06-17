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

Selected via ``options["language"] = "qugar.dolfinx.ffcx_backend"``. Emits, per
integral, the runtime-quadrature dual kernel:

* ``tabulate_tensor_<factory>_original`` — the stock static kernel (used on
  non-cut cells); the unfitted-boundary normal is lowered to ``0.0`` so full
  cells contribute nothing to a boundary integral;
* ``tabulate_tensor_<factory>_custom`` — the runtime kernel
  (:class:`QugarIntegralGenerator`) preceded by the loader + on-the-fly
  tabulation prologue, reading per-cell points/weights/normals from
  ``custom_data``;
* ``tabulate_tensor_<factory>`` — the dispatch wrapper (template slot) that
  picks the kernel from a sentinel smuggled in ``w``.

The unfitted-boundary normal is lowered on the backend's own access object
(see :func:`qugar.dolfinx.ffcx_backend._generator._register_unfitted_normal`),
so no global FFCx monkeypatch is needed.
"""

import sys

import numpy as np
import numpy.typing as npt
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.C import integral_template as ufcx_integrals
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.ir.representation import IntegralIR

from qugar.dolfinx.ffcx_backend import _assemble
from qugar.dolfinx.ffcx_backend._context import integral_tdim, quad_data_from_ir
from qugar.dolfinx.ffcx_backend._generator import (
    QugarIntegralGenerator,
    _register_unfitted_normal,
)
from qugar.dolfinx.ffcx_backend._tables import extract_ir_tables
from qugar.dolfinx.quadrature_data import QuadratureData

__all__ = ["generator"]


def generator(
    ir: IntegralIR, domain, options: dict[str, int | float | npt.DTypeLike]
) -> tuple[str, str]:
    """Generate ``(declaration, implementation)`` for a runtime-quadrature
    integral."""
    factory_name = f"{ir.expression.name}_{domain.name}"
    scalar_c = dtype_to_c_type(options["scalar_type"])  # type: ignore[arg-type]
    geom_dtype = dtype_to_scalar_dtype(options["scalar_type"])  # type: ignore[arg-type]
    geom_c = dtype_to_c_type(geom_dtype)
    real_np = np.dtype(geom_dtype)
    suffix = "f64" if real_np == np.dtype(np.float64) else "f32"

    tdim = integral_tdim(ir)
    quads = quad_data_from_ir(ir, real_np)
    tables = extract_ir_tables(ir, quads, real_np)
    is_mixed_dim = any(t.element_dim != tdim for t in tables)

    quad_tables: dict[QuadratureData, list] = {}
    for t in tables:
        quad_tables.setdefault(t.quad_data, []).append(t)

    asm = _assemble.IntegralAssembly(
        factory_name=factory_name,
        real_c=geom_c,
        suffix=suffix,
        tdim=tdim,
        is_mixed_dim=is_mixed_dim,
        integral_type=ir.expression.integral_type,
        coeffs_offset=_assemble.compute_coeffs_offset(ir),
        quad_tables=quad_tables,
    )

    fmt = Formatter(options["scalar_type"])  # type: ignore[arg-type]

    # Custom (runtime-quadrature) kernel.
    custom_ast = QugarIntegralGenerator(ir, FFCXBackend(ir, options), tables).generate(domain)
    recover = f"const {geom_c}* restrict w_custom = (const {geom_c}*)custom_data;\n"
    custom_fn = (
        _assemble.std_signature(f"tabulate_tensor_{factory_name}_custom", scalar_c, geom_c)
        + "\n{\n"
        + recover
        + _assemble.callers(asm)
        + fmt(custom_ast)
        + "\n}\n\n"
    )

    # Original (static) kernel; the unfitted normal lowers to 0.0.
    backend_orig = FFCXBackend(ir, options)
    _register_unfitted_normal(backend_orig, zero=True)
    original_ast = IntegralGenerator(ir, backend_orig).generate(domain)
    original_fn = (
        _assemble.std_signature(f"tabulate_tensor_{factory_name}_original", scalar_c, geom_c)
        + "\n{\n"
        + fmt(original_ast)
        + "\n}\n\n"
    )

    # ufcx struct + dispatch wrapper (template slot).
    code: dict[str, str] = {}
    if len(ir.enabled_coefficients) > 0:
        values = ", ".join("1" if i else "0" for i in ir.enabled_coefficients)
        code["enabled_coefficients_init"] = (
            f"bool enabled_coefficients_{factory_name}"
            f"[{len(ir.enabled_coefficients)}] = {{{values}}};"
        )
        code["enabled_coefficients"] = f"enabled_coefficients_{factory_name}"
    else:
        code["enabled_coefficients_init"] = ""
        code["enabled_coefficients"] = "NULL"

    for k in ("float32", "float64", "complex64", "complex128"):
        code[f"tabulate_tensor_{k}"] = f".tabulate_tensor_{k} = NULL,"
    if sys.platform.startswith("win32"):
        code["tabulate_tensor_complex64"] = ""
        code["tabulate_tensor_complex128"] = ""
    np_scalar = np.dtype(options["scalar_type"]).name  # type: ignore[arg-type]
    code[f"tabulate_tensor_{np_scalar}"] = (
        f".tabulate_tensor_{np_scalar} = tabulate_tensor_{factory_name},"
    )

    assert ir.expression.coordinate_element_hash is not None
    declaration = ufcx_integrals.declaration.format(factory_name=factory_name)
    factory = ufcx_integrals.factory.format(
        factory_name=factory_name,
        enabled_coefficients=code["enabled_coefficients"],
        enabled_coefficients_init=code["enabled_coefficients_init"],
        tabulate_tensor=_assemble.wrapper_body(asm),
        needs_facet_permutations=(
            "true" if ir.expression.needs_facet_permutations else "false"
        ),
        scalar_type=scalar_c,
        geom_type=geom_c,
        coordinate_element_hash=f"UINT64_C({ir.expression.coordinate_element_hash})",
        tabulate_tensor_float32=code["tabulate_tensor_float32"],
        tabulate_tensor_float64=code["tabulate_tensor_float64"],
        tabulate_tensor_complex64=code["tabulate_tensor_complex64"],
        tabulate_tensor_complex128=code["tabulate_tensor_complex128"],
        domain=int(domain),
    )

    implementation = (
        _assemble.shim_decls(geom_c, suffix)
        + original_fn
        + _assemble.loaders(asm)
        + custom_fn
        + factory
    )
    return declaration, implementation
