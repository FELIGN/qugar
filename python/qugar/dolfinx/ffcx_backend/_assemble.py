# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""C assembly for the runtime-quadrature dual kernel, IR/IRTable-driven.

Ports the loader / prologue / dispatch-wrapper string generation from the
legacy :class:`qugar.dolfinx.codegeneration._IntegralModifier` so it can run
inside the FFCx language backend's ``integral.generator`` (no rendered-C
parsing, no ``ufl_analysis`` dependency). The per-cell points/weights/normals
are delivered through ``custom_data``; the FE basis tables are tabulated on
the fly by the basix shim (repack-free: pointers into the shim block).

The constant-for-points tables stay statically declared *in the kernel body*
(emitted by :class:`QugarIntegralGenerator`), so — unlike the legacy prologue
— they are not re-emitted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import basix
import basix.ufl

from qugar.dolfinx.ffcx_backend._tables import IRTable
from qugar.dolfinx.quadrature_data import QuadratureData


@dataclass
class IntegralAssembly:
    """Everything the C assembly needs, derived from the IR."""

    factory_name: str  # e.g. "integral_<hash>_triangle"
    real_c: str  # C type of points/weights/normals (geometry/real dtype)
    suffix: str  # shim entry-point suffix: "f64" / "f32"
    tdim: int
    is_mixed_dim: bool
    integral_type: str
    coeffs_offset: int
    quad_tables: dict[QuadratureData, list[IRTable]]


def _resolve_mixed_component(element, flat_component):
    if not isinstance(element, basix.ufl._MixedElement):
        return element, flat_component
    acc = 0
    for sub in element.sub_elements:
        if flat_component < acc + sub.reference_value_size:
            return _resolve_mixed_component(sub, flat_component - acc)
        acc += sub.reference_value_size
    raise ValueError(f"flat_component {flat_component} out of range")


def _table_codegen_info(table: IRTable) -> dict:
    """create_element params + repack indices for a table's basix block."""
    el, local_c = _resolve_mixed_component(table.element, table.component)
    block_size = getattr(el, "block_size", 1)
    scalar = el.sub_elements[0] if block_size > 1 else el
    be = scalar.basix_element
    params = (
        int(be.family), int(be.cell_type), be.degree,
        int(be.lagrange_variant), int(be.dpc_variant), int(be.discontinuous),
    )
    gdim = table.element_dim
    vs = 1 if block_size > 1 else int(el.reference_value_size)
    derivs = list(table.derivatives) if table.derivatives else []
    derivs = (derivs + [0] * gdim)[:gdim]
    return {
        "params": params, "gdim": gdim, "vs": vs, "ndofs": table.funcs,
        "didx": basix.index(*derivs), "vaxis": 0 if block_size > 1 else local_c,
        "maxnd": sum(derivs),
    }


def shim_decls(real_c: str, suffix: str) -> str:
    """``extern`` declarations of the basix tabulation shim entry points."""
    return (
        f"extern int qugar_register_element_{suffix}(int, int, int, int, int, int);\n"
        f"extern int qugar_tabulate_{suffix}"
        f"(int, int, const {real_c}*, int, int, {real_c}*, long);\n"
        f"extern {real_c}* qugar_get_scratch_{suffix}(int, long);\n\n"
    )


def loaders(a: IntegralAssembly) -> str:
    """Per-quadrature loaders that unpack points/weights/(normals) from the
    smuggled ``custom_data`` and return the runtime point count."""
    real_c, gdim = a.real_c, a.tdim
    mixed_dim = a.is_mixed_dim
    interior_facet = a.integral_type == "interior_facet"
    fdim = gdim - 1
    code = ""
    for quad_data in a.quad_tables:
        q = quad_data.name
        fn = f"load_points_{a.factory_name}_Q{q}"
        n_pts = f"n_pts_Q{q}"
        has_normals = quad_data.unfitted_boundary
        ind = " " * len(f"int {fn}(")
        code += f"\nint {fn}(const {real_c}* restrict *w_custom"
        code += f",\n{ind}const {real_c}* restrict *points_{q}"
        if interior_facet:
            code += f",\n{ind}const {real_c}* restrict *points_side1_{q}"
        if mixed_dim:
            code += f",\n{ind}const {real_c}* restrict *points_facet_{q}"
        code += f",\n{ind}const {real_c}* restrict *weights_{q}"
        if has_normals:
            code += f",\n{ind}const {real_c}* restrict *normals_{q}"
        code += ")\n{\n"
        code += f"const int {n_pts} = (int) **w_custom;\n*w_custom += 1;\n\n"
        code += f"*points_{q} = *w_custom;\n*w_custom += {gdim} * {n_pts};\n\n"
        if interior_facet:
            code += f"*points_side1_{q} = *w_custom;\n*w_custom += {gdim} * {n_pts};\n\n"
        if mixed_dim:
            code += f"*points_facet_{q} = *w_custom;\n*w_custom += {fdim} * {n_pts};\n\n"
        code += f"*weights_{q} = *w_custom;\n*w_custom += {n_pts};\n\n"
        if has_normals:
            code += f"*normals_{q} = *w_custom;\n*w_custom += {gdim} * {n_pts};\n\n"
        code += f"return {n_pts};\n}}\n\n\n"
    return code


def callers(a: IntegralAssembly) -> str:
    """Prologue for the custom kernel: load points/weights/(normals), then
    tabulate each element once into shim scratch and point the FE buffers into
    it (repack-free, strided). Constant-for-points tables stay in the body."""
    real_c, suffix, tdim = a.real_c, a.suffix, a.tdim
    mixed_dim = a.is_mixed_dim
    interior_facet = a.integral_type == "interior_facet"
    code = ""
    for quad_slot, (quad_data, tables) in enumerate(a.quad_tables.items()):
        q = quad_data.name
        n_pts = f"n_pts_Q{q}"
        has_normals = quad_data.unfitted_boundary
        fn = f"load_points_{a.factory_name}_Q{q}"

        code += f"const {real_c}* restrict points_{q};\n"
        if interior_facet:
            code += f"const {real_c}* restrict points_side1_{q};\n"
        if mixed_dim:
            code += f"const {real_c}* restrict points_facet_{q};\n"
        code += f"const {real_c}* restrict weights_{q};\n"
        if has_normals:
            code += f"const {real_c}* restrict normals_{q};\n"

        call = f"const int {n_pts} = {fn}("
        ci = " " * len(call)
        code += f"{call}&w_custom,\n{ci}&points_{q}"
        if interior_facet:
            code += f",\n{ci}&points_side1_{q}"
        if mixed_dim:
            code += f",\n{ci}&points_facet_{q}"
        code += f",\n{ci}&weights_{q}"
        if has_normals:
            code += f",\n{ci}&normals_{q}"
        code += ");\n\n"

        varying = [t for t in tables if not t.is_constant_for_pts()]
        groups: dict[tuple, list[tuple[IRTable, dict]]] = {}
        for t in varying:
            info = _table_codegen_info(t)
            groups.setdefault(info["params"], []).append((t, info))

        # scratch sizing: only the basix blocks (repack-free => no extra bufs).
        scratch_terms: list[str] = []
        gmeta = []
        for gi, (params, items) in enumerate(groups.items()):
            info0 = items[0][1]
            gdim, ndofs, vs = info0["gdim"], info0["ndofs"], info0["vs"]
            maxnd = max(i["maxnd"] for _t, i in items)
            nderiv = math.comb(maxnd + gdim, gdim)
            any_perm = interior_facet and any(t.permutations > 1 for t, _ in items)
            blk = f"{nderiv} * {n_pts} * {ndofs} * {vs}"
            gmeta.append((gi, params, items, gdim, ndofs, vs, maxnd, any_perm, blk))
            scratch_terms.append(blk)
            if any_perm:
                scratch_terms.append(blk)

        scratch = f"scratch_{q}"
        if scratch_terms:
            code += (
                f"{real_c}* {scratch} = qugar_get_scratch_{suffix}"
                f"({quad_slot}, (long)({' + '.join(scratch_terms)}));\n\n"
            )

        offsets: list[str] = []
        def _off() -> str:
            return " + ".join(offsets) if offsets else "0"

        for (gi, params, items, gdim, ndofs, vs, maxnd, any_perm, blk) in gmeta:
            fam, cell, deg, lv, dv, disc = params
            h = f"h_{q}_{gi}"
            blk0, blk1 = f"block_{q}_{gi}", f"block_side1_{q}_{gi}"
            pts = f"points_facet_{q}" if (mixed_dim and gdim != tdim) else f"points_{q}"
            code += (
                f"const int {h} = qugar_register_element_{suffix}"
                f"({fam}, {cell}, {deg}, {lv}, {dv}, {disc});\n"
                f"if ({h} < 0) return;\n"
            )
            code += f"{real_c}* {blk0} = {scratch} + ({_off()});\n"
            offsets.append(blk)
            code += (
                f"if (qugar_tabulate_{suffix}({h}, {maxnd}, {pts}, {n_pts}, {gdim}, "
                f"{blk0}, (long)({blk})) != 0) return;\n"
            )
            if any_perm:
                code += f"{real_c}* {blk1} = {scratch} + ({_off()});\n"
                offsets.append(blk)
                code += (
                    f"if (qugar_tabulate_{suffix}({h}, {maxnd}, points_side1_{q}, "
                    f"{n_pts}, {gdim}, {blk1}, (long)({blk})) != 0) return;\n"
                )
            code += "\n"
            for t, info in items:
                didx, vaxis = info["didx"], info["vaxis"]
                stride = ndofs * vs
                off = f"{didx} * {n_pts} * {stride}"
                if vaxis:
                    off += f" + {vaxis}"
                two_sided = interior_facet and t.permutations > 1
                if two_sided:
                    code += f"const {real_c}* restrict {t.name}[2];\n"
                    code += f"{t.name}[0] = {blk0} + ({off});\n"
                    code += f"{t.name}[1] = {blk1} + ({off});\n\n"
                else:
                    code += f"const {real_c}* restrict {t.name} = {blk0} + ({off});\n\n"
    return code


def wrapper_body(a: IntegralAssembly) -> str:
    """Dispatch-wrapper body (goes in the template tabulate_tensor slot):
    read the smuggled offset from ``w`` and call the custom or original kernel."""
    f = a.factory_name
    return (
        f"const ptrdiff_t w_custom_offset = *((const ptrdiff_t *) &w[{a.coeffs_offset}]);\n"
        "const bool is_custom = w_custom_offset > 0;\n"
        "const bool is_full   = w_custom_offset < 0;\n\n"
        "if (is_custom)\n{\n"
        f"  const {a.real_c}* restrict w_custom = (const {a.real_c} *)w + w_custom_offset;\n"
        f"  tabulate_tensor_{f}_custom(A, w, c, coordinate_dofs, "
        "entity_local_index, quadrature_permutation, (void*)w_custom);\n"
        "}\nelse if (is_full)\n{\n"
        f"  tabulate_tensor_{f}_original(A, w, c, coordinate_dofs, "
        "entity_local_index, quadrature_permutation, custom_data);\n"
        "}\n"
    )


def std_signature(fn_name: str, scalar_c: str, geom_c: str) -> str:
    """Standard ufcx tabulate_tensor signature for a named kernel."""
    return (
        f"void {fn_name}({scalar_c}* restrict A,\n"
        f"    const {scalar_c}* restrict w,\n"
        f"    const {scalar_c}* restrict c,\n"
        f"    const {geom_c}* restrict coordinate_dofs,\n"
        "    const int* restrict entity_local_index,\n"
        "    const uint8_t* restrict quadrature_permutation,\n"
        "    void* custom_data)"
    )


def compute_coeffs_offset(ir) -> int:
    """Offset in ``w`` where qugar's smuggled per-cell data begins (just past
    all of the integral's coefficients), matching DOLFINx's packing."""
    offsets = list(ir.expression.coefficient_offsets.values())
    if not offsets:
        return 0
    coeff = list(ir.expression.coefficient_offsets.keys())[-1]
    element = coeff.ufl_function_space().element
    return offsets[-1] + element.space_dimension
