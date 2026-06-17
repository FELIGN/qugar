# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Functionalities for modifying FFCx generated code for integrals."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import math
import os
import re

import basix
import basix.ufl
import ffcx.codegeneration.codegeneration
import numpy as np
import numpy.typing as npt
from ffcx.analysis import UFLData
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.codegeneration import CodeBlocks
from ffcx.ir.representation import DataIR, IntegralIR

from qugar.dolfinx._kernel_body import KernelBody
from qugar.dolfinx.fe_table import FETable
from qugar.dolfinx.ffcx_backend._generator import QugarIntegralGenerator
from qugar.dolfinx.integral_data import IntegralData, extract_integral_data
from qugar.dolfinx.parsing_utils import dtype_to_C_str


def _use_ast_backend() -> bool:
    """Whether to generate the custom kernel body via the FFCx language
    backend (AST) instead of the legacy text transforms.

    The AST backend is the default; set ``QUGAR_FFCX_BACKEND=0`` to fall back
    to the legacy text-transform path (kept during the migration)."""
    return os.environ.get("QUGAR_FFCX_BACKEND", "1") not in ("0", "false", "False")


_QUGAR_FFCX_LANGUAGE = "qugar.dolfinx.ffcx_backend"


def _use_language_backend() -> bool:
    """Whether to generate the whole integral via the FFCx language backend
    (``options["language"]``), which owns the dual kernel and lowers the
    unfitted normal in-backend (no global monkeypatch, no rendered-C parsing).

    Opt-in during the migration via ``QUGAR_FFCX_LANGUAGE_BACKEND=1``; off by
    default (the AST-hybrid path remains the default)."""
    return os.environ.get("QUGAR_FFCX_LANGUAGE_BACKEND", "") in ("1", "true", "True")


def _generate_code_language_backend(
    ufl_data: UFLData, ir: DataIR, ffcx_options: dict[str, int | float | npt.DTypeLike]
) -> tuple[CodeBlocks, list[IntegralData]]:
    """Generate the runtime-quadrature kernels via the qugar FFCx language
    backend (``options["language"]``). The kernels are produced by
    ``qugar.dolfinx.ffcx_backend.integral.generator``; the per-cell packer's
    integral data is rebuilt from the IR (no rendered-C parsing)."""
    from qugar.dolfinx.integral_data import extract_integral_data_from_ir

    opts = dict(ffcx_options)
    opts["language"] = _QUGAR_FFCX_LANGUAGE
    code_blocks, _suffixes = ffcx.codegeneration.codegeneration.generate_code(ir, opts)
    code_blocks = _modify_header(code_blocks)

    itg_datas = [
        extract_integral_data_from_ir(ufl_data, ir, itg_ir, ffcx_options)
        for itg_ir in ir.integrals
    ]
    return code_blocks, itg_datas


def _modify_header(code_blocks: CodeBlocks) -> CodeBlocks:
    """Adds stddef.h include in the header of the given `code_blocks`.

    This include may be required for ``ptrdiff_t`` type used in the
    modified integral code.

    It modifies the header of the C code associated to a form to be
    computed by appending the new include before ``ufcx.h``

    Args:
        code_blocks (CodeBlocks): Original code blocks generated
            by the FFCx library.

    Returns:
        CodeBlocks: Modified code blocks including extra header.
    """

    assert len(code_blocks[0]) == 1

    header = code_blocks[0][0][1]
    match = re.search(r"#include <ufcx.h>", header)
    assert match

    new_header = header[: match.start()]
    new_header += "#include <stddef.h>\n"
    new_header += header[match.start() :]

    code_blocks[0][0] = (code_blocks[0][0][0], new_header)

    return code_blocks


class _IntegralModifier:
    """Class for modifying a C code corresponding a tabulate tensor
    integral function, allowing the use of custom quadrature rules
    changing cell by cell, for a subset of cells.

    The main use of this class after initialization is to call its
    public method `create_new_code` to create the modified integral
    code. This new code replaces the original C code with a new version
    that can invoke two possible functions: the original version with
    static definitions for weights and FE tables (to be used when all
    the cells have a static quadrature defined at compile time), and a
    new version where those values are loaded dynamically at runtime
    for every cell from provided custom coefficients. The latter also
    allows to use normals at custom unfitted boundaries that do not
    coincide with cells' facets.

    Parameters:
        _ir (IntegralIR): FFCx Intermediate Representation of the
            integral.
        _data (IntegralData): Data structure containing information
            associated to the quadrature, as integral type, subdomain
            ids, quadratures, and FE tables.
        _body (KernelBody): Structured representation of the parsed
            kernel (signature, per-quadrature loops, pre/post bands).
        _integral_name (str): FFCx-generated hash part of the integral
            function name.
        _coeffs_dtype (type[np.float32 | np.float64 | np.complex64 |
            np.complex128]): Type of the function's coefficients and
            constants.
    """

    def __init__(
        self,
        code: str,
        itg_ir: IntegralIR,
        itg_data: IntegralData,
        ffcx_options: dict[str, int | float | npt.DTypeLike] | None = None,
    ) -> None:
        """Constructor.

        Args:
            code (str): Original C code associated to the tabulate
                tensor integral function.
            itg_ir (IntegralIR): ffcx Intermediate Representation of the
                integral to be modified.
            data (IntegralData): Data structure containing information
                associated to the quadrature, as integral type,
                subdomain ids, quadratures, and FE tables.
            ffcx_options: FFCx options (needed by the AST language
                backend; optional for the legacy text path).
        """

        self._ir = itg_ir
        self._data = itg_data
        self._ffcx_options = ffcx_options
        self._body = KernelBody.from_ffcx(code, itg_ir, itg_data)
        self._integral_name = self._body.integral_name
        self._coeffs_dtype = self._body.coeffs_dtype
        self._check_FFCx_input()

    def _compute_coeffs_offset(self) -> int:
        """Computes the offfset of the position where the
        custom information starts in the coefficients array pointer
        ``w``.

        Note that this offset corresponds to all the coefficients of all
        the integrands for a single integral, according to the way it is
        computed in DOLFINx.
        See https://github.com/FEniCS/dolfinx/blob/5a20e2ba3907c1f108cb0af45931f46e32250351/cpp/dolfinx/fem/utils.h#L957C1-L987C2

        Returns:
            int: Offset value.
        """

        offsets = list(self._ir.expression.coefficient_offsets.values())
        if len(offsets) == 0:
            return 0
        else:
            coeff = list(self._ir.expression.coefficient_offsets.keys())[-1]
            element = coeff.ufl_function_space().element  # type: ignore
            return offsets[-1] + element.space_dimension

    def _check_FFCx_input(self) -> None:
        """Checks the validity of the integral to be modified.
        If the integral type is not supported, an exception is raised.

        Right now, this function only supports integrals to be computed
        over cells, exterior and interior facets. Vertices are not
        supported. It does not support either the use of
        sum-factorization quadratures.

        It also checks that some assumptions about the
        `entity_local_index` and `quadrature_permutation` arrays on
        which the implementation relies upon, are valid.

        Raises:
            ValueError: An exception is raised if the integral is not
                performed in a cell or facet.
            ValueError: An exception is raised is thrown if the
                the hypotheses about `entity_local_index` array are
                wrong.
            ValueError: An exception is raised is thrown if the
                the hypotheses about `quadrature_permutation` array are
                wrong.
            ValueError: An exception is raised is thrown if the
                integral is to be performed using sum-factorization.
        """

        # Checking integral and entity types.

        if self._ir.expression.entity_type not in ["cell", "facet"]:
            raise ValueError("Only cell and facet entity types are supported.")

        if self._data.integral_type not in [
            "cell",
            "exterior_facet",
            "interior_facet",
        ]:
            raise ValueError(
                "Only cell, exterior and interior facets integral types are supported."
            )

        # Checking that is the integral presents mixed dimension,
        # then it must be an integral over exterior facets.
        if self._data.is_mixed_dim:
            assert self._data.integral_type == "exterior_facet", "Not implemented."

        # Checking that sum-factorization is disabled.

        for _cell_type, quad in self._ir.expression.integrand.keys():
            if quad.has_tensor_factors:  # type: ignore
                raise ValueError(
                    "Sum-factorization quadrature it is incompatible with "
                    "custom quadratures. Toogle off sum-factorization in the "
                    "ffcx options passed when building the form."
                )

        # Checking the indices of `entity_local_index` /
        # `quadrature_permutation` being accessed. Scan the pre-loop
        # band of the first loop + every loop body. FFCx emits these
        # accesses only in those two places; inter-loop pre_text
        # (loops[1+].pre_text) is always blank whitespace in practice.
        body_text = (
            (self._body.loops[0].pre_text if self._body.loops else "")
            + "".join(L.body for L in self._body.loops)
        )

        def _collected_indices(pattern: str) -> set[int]:
            indices: set[int] = set()
            for match in re.finditer(pattern, body_text):
                index_str = match.group(1)
                assert index_str.isnumeric()
                indices.add(int(index_str))
            return indices

        target_indices = {
            "cell": set(),
            "exterior_facet": set([0]),
            "interior_facet": set([0, 1]),
        }
        idx = _collected_indices(r"entity_local_index\[(\w+)\]")
        if idx != target_indices[self._data.integral_type]:
            raise ValueError(
                "Implementation error: Invalid indices for 'entity_local_index' array."
            )

        idx = _collected_indices(r"quadrature_permutation\[(\w+)\]")
        if len(idx) > 0 and idx != target_indices[self._data.integral_type]:
            raise ValueError(
                "Implementation error: Invalid indices for 'quadrature_permutation' array."
            )

        # Checking type compatibility between _coeffs_dtype and integral
        # dtype.
        if self._coeffs_dtype in [np.float32, np.complex64]:
            assert self._data.dtype == np.float32
        elif self._coeffs_dtype in [np.float64, np.complex128]:
            assert self._data.dtype == np.float64
        else:
            assert False

    def _shim_suffix(self) -> str:
        """Returns the shim entry-point suffix for the real dtype."""
        return "f64" if self._data.dtype == np.float64 else "f32"

    def _create_shim_decls(self) -> str:
        """Emits the ``extern`` declarations of the basix tabulation shim
        entry points used by the on-the-fly custom kernel."""
        suffix = self._shim_suffix()
        dtype_str = dtype_to_C_str(self._data.dtype)
        return (
            f"extern int qugar_register_element_{suffix}"
            "(int, int, int, int, int, int);\n"
            f"extern int qugar_tabulate_{suffix}"
            f"(int, int, const {dtype_str}*, int, int, {dtype_str}*, long);\n"
            f"extern {dtype_str}* qugar_get_scratch_{suffix}(int, long);\n\n"
        )

    @staticmethod
    def _resolve_mixed_component(element, flat_component: int):
        """For a ``basix.ufl._MixedElement``, drill into the sub-element
        that owns ``flat_component`` and return
        ``(sub_element, component_within_sub)``. Non-mixed elements pass
        through unchanged. Recursive for nested mixed elements.
        """
        if not isinstance(element, basix.ufl._MixedElement):
            return element, flat_component
        acc = 0
        for sub in element.sub_elements:
            if flat_component < acc + sub.reference_value_size:
                return _IntegralModifier._resolve_mixed_component(
                    sub, flat_component - acc)
            acc += sub.reference_value_size
        raise ValueError(
            f"flat_component {flat_component} out of range for mixed element "
            f"with reference_value_size {element.reference_value_size}"
        )

    @staticmethod
    def _table_codegen_info(table: FETable) -> dict:
        """Computes the constants needed to repack a basix tabulation
        block into a single FFCx FE table.

        Returns a dict with the core-element ``create_element`` parameters
        and the indices used by the repack: ``didx`` (basix derivative
        index), ``vaxis`` (value-axis index), ``ndofs`` and ``vs`` (basix
        block dims), and ``gdim`` (points dimension).

        For mixed elements (``basix.ufl._MixedElement``), drills into the
        sub-element that owns ``table.component`` so the basix block is
        tabulated for the right sub-element.
        """
        el, local_c = _IntegralModifier._resolve_mixed_component(
            table.element, table.component)
        block_size = getattr(el, "block_size", 1)
        scalar = el.sub_elements[0] if block_size > 1 else el
        be = scalar.basix_element
        params = (
            int(be.family),
            int(be.cell_type),
            be.degree,
            int(be.lagrange_variant),
            int(be.dpc_variant),
            int(be.discontinuous),
        )
        gdim = table.element_dim
        vs = 1 if block_size > 1 else int(el.reference_value_size)
        derivs = list(table.derivatives) if table.derivatives else []
        derivs = (derivs + [0] * gdim)[:gdim]
        return {
            "params": params,
            "gdim": gdim,
            "vs": vs,
            "ndofs": table.funcs,
            "didx": basix.index(*derivs),
            "vaxis": 0 if block_size > 1 else local_c,
            "maxnd": sum(derivs),
        }

    def _create_custom_data_loaders(self) -> str:
        """Creates C functions that load, per quadrature/integrand, the
        custom quadrature *points* and *weights* (and unfitted-boundary
        normals) from the custom coefficients array.

        Unlike the previous design, FE table *values* are NOT read here:
        they are computed on the fly by tabulating the basix elements at
        the loaded points (see ``_create_custom_data_callers``). The
        loader only advances the ``w_custom`` pointer past the per-cell
        points/weights/normals and returns the number of points.

        Returns:
            str: C code for the generated loader functions.
        """

        dtype_str = dtype_to_C_str(self._data.dtype)
        name = self._data.name
        gdim = self._data.tdim
        mixed_dim = self._data.is_mixed_dim
        interior_facet = self._data.integral_type == "interior_facet"
        fdim = gdim - 1  # facet-reference coordinate dimension

        code = ""

        for quad_data, _FE_tables in self._data.quad_data_FE_tables.items():
            quad_name = quad_data.name

            func_name = f"load_points_{name}_Q{quad_name}"
            points_name = f"points_{quad_name}"
            points_side1_name = f"points_side1_{quad_name}"
            points_facet_name = f"points_facet_{quad_name}"
            weights_name = f"weights_{quad_name}"
            normals_name = f"normals_{quad_name}"
            n_pts_name = f"n_pts_Q{quad_name}"

            has_normals = quad_data.unfitted_boundary

            indent = " " * len(f"int {func_name}(")
            code += f"\nint {func_name}(const {dtype_str}* restrict *w_custom"
            code += f",\n{indent}const {dtype_str}* restrict *{points_name}"
            if interior_facet:
                code += f",\n{indent}const {dtype_str}* restrict *{points_side1_name}"
            if mixed_dim:
                code += f",\n{indent}const {dtype_str}* restrict *{points_facet_name}"
            code += f",\n{indent}const {dtype_str}* restrict *{weights_name}"
            if has_normals:
                code += f",\n{indent}const {dtype_str}* restrict *{normals_name}"
            code += ")\n{\n"

            code += f"const int {n_pts_name} = (int) **w_custom;\n*w_custom += 1;\n\n"
            code += f"*{points_name} = *w_custom;\n"
            code += f"*w_custom += {gdim} * {n_pts_name};\n\n"
            if interior_facet:
                code += f"*{points_side1_name} = *w_custom;\n"
                code += f"*w_custom += {gdim} * {n_pts_name};\n\n"
            if mixed_dim:
                code += f"*{points_facet_name} = *w_custom;\n"
                code += f"*w_custom += {fdim} * {n_pts_name};\n\n"
            code += f"*{weights_name} = *w_custom;\n"
            code += f"*w_custom += {n_pts_name};\n\n"
            if has_normals:
                code += f"*{normals_name} = *w_custom;\n"
                code += f"*w_custom += {self._data.tdim} * {n_pts_name};\n\n"

            code += f"return {n_pts_name};\n}}\n\n\n"

        return code

    def _create_custom_data_callers(self, repack_free: bool = False) -> str:
        """Creates the code for calling the functions loading
        the weights arrays and FE tables from extra coefficients.

        This code is intended to be called inside the new generated
        function that uses dynamic (instead of static) data.
        It also declare the needed pointers.

        Returns:
            str: C code for calling dynamic data loaders.
        """

        dtype_str = dtype_to_C_str(self._data.dtype)
        suffix = self._shim_suffix()

        itg_name = self._data.name
        tdim = self._data.tdim
        mixed_dim = self._data.is_mixed_dim

        is_interior_facet = self._data.integral_type == "interior_facet"

        call_code = ""
        for quad_slot, (quad_data, FE_tables) in enumerate(
            self._data.quad_data_FE_tables.items()
        ):
            quad_name = quad_data.name

            has_normals = quad_data.unfitted_boundary

            func_name = f"load_points_{itg_name}_Q{quad_name}"
            points_name = f"points_{quad_name}"
            points_side1_name = f"points_side1_{quad_name}"
            points_facet_name = f"points_facet_{quad_name}"
            weights_name = f"weights_{quad_name}"
            normals_name = f"normals_{quad_name}"
            n_pts_name = f"n_pts_Q{quad_name}"

            # Constant-for-points tables stay statically defined.
            for table in FE_tables:
                if table.is_constant_for_pts():
                    call_code += table.code + "\n"

            # Per-cell points/weights (+side-1, +facet points, +normals) decls.
            call_code += f"const {dtype_str}* restrict {points_name};\n"
            if is_interior_facet:
                call_code += f"const {dtype_str}* restrict {points_side1_name};\n"
            if mixed_dim:
                call_code += f"const {dtype_str}* restrict {points_facet_name};\n"
            call_code += f"const {dtype_str}* restrict {weights_name};\n"
            if has_normals:
                call_code += f"const {dtype_str}* restrict {normals_name};\n"

            # Loader execution (returns the runtime number of points).
            aux = f"const int {n_pts_name} = {func_name}("
            indent = " " * len(aux)
            call_code += f"{aux}&w_custom,\n{indent}&{points_name}"
            if is_interior_facet:
                call_code += f",\n{indent}&{points_side1_name}"
            if mixed_dim:
                call_code += f",\n{indent}&{points_facet_name}"
            call_code += f",\n{indent}&{weights_name}"
            if has_normals:
                call_code += f",\n{indent}&{normals_name}"
            call_code += ");\n\n"

            # On-the-fly tabulation + repack, grouped per element so each
            # element is tabulated once (one basix block feeds all of its
            # component/derivative tables). For interior-facet integrals
            # with perm>1 tables, two basix blocks are produced (one per
            # side) and perm>1 tables get an FE..[2] array of per-side
            # buffers; perm<=1 tables stay single-buffer (their values
            # agree on both sides).
            #
            # All buffers (basix blocks + per-table repack buffers when
            # vs > 1) live in a SINGLE thread-local scratch obtained from
            # the shim. Using the stack via VLAs overflows the thread's
            # stack for higher-degree 3D forms (e.g. P3 hex), so we offset
            # into the scratch instead.
            varying = [t for t in FE_tables if not t.is_constant_for_pts()]

            groups: dict[tuple, list[tuple[FETable, dict]]] = {}
            for table in varying:
                info = self._table_codegen_info(table)
                groups.setdefault(info["params"], []).append((table, info))

            # First pass: gather per-group metadata and the list of scratch
            # size terms.
            group_meta = []
            scratch_terms: list[str] = []
            for gi, (params, items) in enumerate(groups.items()):
                info0 = items[0][1]
                gdim, ndofs, vs = info0["gdim"], info0["ndofs"], info0["vs"]
                maxnd = max(info["maxnd"] for _t, info in items)
                nderiv = math.comb(maxnd + gdim, gdim)
                any_perm = is_interior_facet and any(
                    t.permutations > 1 for t, _ in items)
                blk_size_expr = (
                    f"{nderiv} * {n_pts_name} * {ndofs} * {vs}"
                )
                group_meta.append(
                    (gi, params, items, gdim, ndofs, vs, maxnd, nderiv,
                     any_perm, blk_size_expr)
                )
                scratch_terms.append(blk_size_expr)
                if any_perm:
                    scratch_terms.append(blk_size_expr)
                for table, info in items:
                    if info["vs"] > 1 and not repack_free:
                        scratch_terms.append(f"{n_pts_name} * {table.funcs}")
                        if is_interior_facet and table.permutations > 1:
                            scratch_terms.append(
                                f"{n_pts_name} * {table.funcs}"
                            )

            scratch_var = f"scratch_{quad_name}"
            if scratch_terms:
                total = " + ".join(scratch_terms)
                # Each quadrature gets its own scratch slot so that, in a
                # multi-quadrature kernel, one quadrature's buffer is never
                # reallocated/aliased while another's tables are still in
                # use (see qugar_get_scratch in the shim).
                call_code += (
                    f"{dtype_str}* {scratch_var} = "
                    f"qugar_get_scratch_{suffix}({quad_slot}, (long)({total}));\n\n"
                )

            # Running offset (sum of size expressions already consumed).
            offsets: list[str] = []

            def _cur_offset() -> str:
                return " + ".join(offsets) if offsets else "0"

            for gd in group_meta:
                (gi, params, items, gdim, ndofs, vs, maxnd, nderiv,
                 any_perm, blk_size_expr) = gd
                fam, cell, deg, lv, dv, disc = params
                handle = f"h_{quad_name}_{gi}"
                blk0 = f"block_{quad_name}_{gi}"
                blk1 = f"block_side1_{quad_name}_{gi}"
                stride = f"{ndofs} * {vs}"

                pts_var = (points_facet_name
                           if (mixed_dim and gdim != tdim)
                           else points_name)

                call_code += (
                    f"const int {handle} = qugar_register_element_{suffix}"
                    f"({fam}, {cell}, {deg}, {lv}, {dv}, {disc});\n"
                    f"if ({handle} < 0) return;\n"
                )
                call_code += (
                    f"{dtype_str}* {blk0} = {scratch_var} + ({_cur_offset()});\n"
                )
                offsets.append(blk_size_expr)
                call_code += (
                    f"if (qugar_tabulate_{suffix}({handle}, {maxnd}, {pts_var}, "
                    f"{n_pts_name}, {gdim}, {blk0}, "
                    f"(long)({blk_size_expr})) != 0) return;\n"
                )
                if any_perm:
                    call_code += (
                        f"{dtype_str}* {blk1} = {scratch_var} + "
                        f"({_cur_offset()});\n"
                    )
                    offsets.append(blk_size_expr)
                    call_code += (
                        f"if (qugar_tabulate_{suffix}({handle}, {maxnd}, "
                        f"{points_side1_name}, {n_pts_name}, {gdim}, "
                        f"{blk1}, (long)({blk_size_expr})) != 0) return;\n"
                    )
                call_code += "\n"

                for table, info in items:
                    funcs = table.funcs
                    didx, vaxis = info["didx"], info["vaxis"]
                    two_sided = is_interior_facet and table.permutations > 1

                    if vs == 1 or repack_free:
                        # Point FE.. directly into the basix block (no copy).
                        # The block layout is [deriv][point][dof][vs]; the
                        # kernel accesses FE..[vs * (funcs*iq + dof)], so the
                        # buffer carries the derivative + value-axis offset
                        # `didx * n_pts * (funcs*vs) + vaxis`. For vs == 1 this
                        # reduces to the scalar `didx * n_pts * funcs`.
                        off_in_blk = f"{didx} * {n_pts_name} * {funcs * vs}"
                        if vaxis:
                            off_in_blk += f" + {vaxis}"
                        if two_sided:
                            call_code += (
                                f"const {dtype_str}* restrict "
                                f"{table.name}[2];\n"
                            )
                            call_code += (
                                f"{table.name}[0] = {blk0} + {off_in_blk};\n"
                            )
                            call_code += (
                                f"{table.name}[1] = {blk1} + {off_in_blk};\n\n"
                            )
                        else:
                            call_code += (
                                f"const {dtype_str}* restrict {table.name} = "
                                f"{blk0} + {off_in_blk};\n\n"
                            )
                        continue

                    # vs > 1: strides differ -> a copy into a separate
                    # buffer is unavoidable. The buffer also lives in the
                    # shim scratch.
                    # Default-argument capture freezes the current loop
                    # variable values, avoiding Python's late-binding closure.
                    def _repack_into(
                        buf_name: str, src: str,
                        _n=n_pts_name, _f=funcs, _d=didx, _st=stride,
                        _v=vs, _va=vaxis,
                    ) -> str:
                        r = f"for (int iq = 0; iq < {_n}; ++iq)\n"
                        r += f"  for (int kd = 0; kd < {_f}; ++kd)\n"
                        r += (
                            f"    {buf_name}[iq * {_f} + kd] = {src}"
                            f"[{_d} * {_n} * {_st} + "
                            f"iq * {_st} + kd * {_v} + {_va}];\n"
                        )
                        return r

                    buf_size_expr = f"{n_pts_name} * {funcs}"
                    if two_sided:
                        buf0 = f"{table.name}_buf_0"
                        buf1 = f"{table.name}_buf_1"
                        call_code += (
                            f"{dtype_str}* {buf0} = {scratch_var} + "
                            f"({_cur_offset()});\n"
                        )
                        offsets.append(buf_size_expr)
                        call_code += (
                            f"{dtype_str}* {buf1} = {scratch_var} + "
                            f"({_cur_offset()});\n"
                        )
                        offsets.append(buf_size_expr)
                        call_code += _repack_into(buf0, blk0)
                        call_code += _repack_into(buf1, blk1)
                        call_code += (
                            f"const {dtype_str}* restrict "
                            f"{table.name}[2];\n"
                        )
                        call_code += f"{table.name}[0] = {buf0};\n"
                        call_code += f"{table.name}[1] = {buf1};\n\n"
                    else:
                        buf = f"{table.name}_buf"
                        call_code += (
                            f"{dtype_str}* {buf} = {scratch_var} + "
                            f"({_cur_offset()});\n"
                        )
                        offsets.append(buf_size_expr)
                        call_code += _repack_into(buf, blk0)
                        call_code += (
                            f"const {dtype_str}* restrict {table.name} = "
                            f"{buf};\n\n"
                        )

        return call_code

    def _create_non_custom_quad_function(self) -> str:
        """Creates a copy of the original FFCx function (signature +
        implementation), suffixed ``_original``.

        For forms that contain an unfitted-boundary integral, the loops
        associated to that integral are short-circuited with ``break;``
        so the kernel does nothing when invoked on a non-custom cell.
        The compiler dead-code-eliminates the rest of the loop body.
        """
        if self._body.has_unfitted_boundary():
            return (
                self._body.copy()
                .inline_pre_loop_into_loops()
                .null_unfitted_normals()
                .break_unfitted_loops()
                .render(suffix="_original")
            )
        return self._body.render(suffix="_original")

    def _create_custom_function(self) -> str:
        """Creates a new version of the original function (signature +
        implementation), in which the static versions of weigths and FE
        tables are replaced by values dynamically loaded from the
        coefficients array ``w_custom``.

        The accesses to FE tables are transformed from 4D static-array
        accesses to 1D dynamic-buffer accesses. The upper bounds of the
        quadrature loops are rewritten to use the runtime point count.
        For unfitted boundaries the boundary normal is emitted by FFCx
        (via the :mod:`qugar.dolfinx._ffcx_patches` backend handler); the
        pre-loop band is inlined into every loop so ``iq`` is in scope at
        every ``normals_<quad>`` access site.

        Returns:
            str: New generated code.
        """

        # The inner kernel keeps the STANDARD tabulate_tensor signature -- no
        # extra ``w_custom`` argument. Per-cell points/weights/(normals) are
        # delivered via ``void* custom_data`` (the wrapper sets it). When
        # upstream dolfinx exposes custom_data directly, the wrapper goes
        # away and this kernel is already ABI-ready.
        dtype_str = dtype_to_C_str(self._data.dtype)

        if _use_ast_backend() and self._ffcx_options is not None:
            return self._create_custom_function_ast(dtype_str)

        # Apply the custom-variant transformation pipeline on the
        # structured body. Order matters:
        #   erase_static -> rewrite_table_accesses -> dynamic_loop_bounds
        #   -> inline_pre_loop_into_loops.
        # The normal is lowered structurally by FFCx (via _ffcx_patches),
        # so no substitute_normals step is needed.
        body = (
            self._body.copy()
            .erase_static_declarations()
            .rewrite_table_accesses()
            .dynamic_loop_bounds()
            .inline_pre_loop_into_loops()
        )

        # Recover w_custom from custom_data (interim, while dolfinx still
        # hardcodes nullptr for custom_data and we smuggle through ``w``)
        # and emit the shim register + tabulate + repack prologue at the
        # top of the (now-empty) pre-loop band of the FIRST loop.
        recover = (
            f"const {dtype_str}* restrict w_custom = "
            f"(const {dtype_str}*)custom_data;\n"
        )
        prologue = recover + self._create_custom_data_callers()
        if body.loops:
            body.loops[0].pre_text = prologue + body.loops[0].pre_text

        return body.render(suffix="_custom")

    def _create_custom_function_ast(self, dtype_str: str) -> str:
        """Generate the custom kernel body via the FFCx language backend
        (:class:`QugarIntegralGenerator`) instead of the text transforms.

        The static tables/weights are stripped at the AST level and table
        accesses flattened; the per-cell loader + runtime-tabulation
        prologue (and the constant-for-points table re-declarations) are
        reused verbatim from :meth:`_create_custom_data_callers`. The result
        is equivalent to :meth:`_create_custom_function`.
        """
        assert self._ffcx_options is not None
        domain = next(iter(self._ir.expression.unique_table_types))
        tables = [t for tl in self._data.quad_data_FE_tables.values() for t in tl]

        backend = FFCXBackend(self._ir, self._ffcx_options)
        gen = QugarIntegralGenerator(self._ir, backend, tables, strip_all_tables=True)
        body_c = Formatter(self._ffcx_options["scalar_type"])(  # type: ignore[arg-type]
            gen.generate(domain)
        )

        recover = (
            f"const {dtype_str}* restrict w_custom = "
            f"(const {dtype_str}*)custom_data;\n"
        )
        # The AST kernel accesses tables with the strided form
        # FE..[vs * (funcs*iq + dof)], so the prologue points FE.. into the
        # basix block directly (no per-cell repack copy, even for vs > 1).
        prologue = recover + self._create_custom_data_callers(repack_free=True)

        sig = self._body.signature.replace(
            self._integral_name, self._integral_name + "_custom", 1
        )
        return f"{sig}\n{{\n{prologue}{body_c}\n}}\n\n"

    def _create_new_function(self) -> str:
        """Creates a new function for replacing the original one.

        This function decides at runtime if the static (original)
        version of the function should be called, or the new dynamic
        one. This decision is made based of the additional information
        appended to the coefficients array.

        Returns:
            str: New generated code.
        """

        code_impl = "{\n"

        dtype_str = dtype_to_C_str(self._data.dtype)
        coeffs_offset = self._compute_coeffs_offset()
        code_impl += (
            "const ptrdiff_t w_custom_offset = *((const ptrdiff_t *) " + f"&w[{coeffs_offset}]);\n"
        )

        code_impl += "const bool is_custom = w_custom_offset > 0;\n"
        code_impl += "const bool is_full   = w_custom_offset < 0;\n\n"
        code_impl += "if (is_custom)\n"
        code_impl += "{\n"

        # ``w_custom_offset`` is in real-dtype units (the same unit the
        # packer uses to lay out smuggled data). For complex coefficient
        # forms ``w`` is complex; cast it to the real ``T*`` first, then
        # advance -- this way the same offset semantics work for both
        # real and complex coeffs.
        code_impl += (
            f"  const {dtype_str}* restrict w_custom = "
            f"(const {dtype_str} *)w + w_custom_offset;\n"
        )

        name = self._integral_name
        # Pass the smuggled w_custom region as void* custom_data to the
        # inner kernel -- the inner kernel reads only from custom_data, so
        # when upstream dolfinx exposes custom_data natively this wrapper
        # is deletable and the inner kernel stays unchanged.
        code_impl += (
            f"  tabulate_tensor_integral_{name}_custom(A, w, c, "
            "coordinate_dofs, entity_local_index, quadrature_permutation, "
            "(void*)w_custom);\n"
        )
        code_impl += "}\nelse if (is_full)\n{\n"
        code_impl += (
            f"  tabulate_tensor_integral_{name}_original(A, w, c, "
            "coordinate_dofs, entity_local_index, quadrature_permutation, custom_data);\n"
        )
        code_impl += "}\n}\n"

        return self._body.signature + "\n" + code_impl

    def create_new_code(self) -> str:
        """Creates a modification of the original function's C code by
        creating the posibility of using either the statically defined
        weights and FE tables, or loading them dynamically at certain
        cells (with weights and FE tables that are different for each
        cell or facet).

        Returns:
            str: Newly generated code.
        """

        new_code = ""
        new_code += self._body.before_func
        new_code += self._create_shim_decls()
        new_code += self._create_non_custom_quad_function()
        new_code += self._create_custom_data_loaders()
        new_code += self._create_custom_function()
        new_code += self._create_new_function()
        new_code += self._body.after_func

        return new_code


def generate_code(
    ufl_data: UFLData,
    ir: DataIR,
    ffcx_options: dict[str, int | float | npt.DTypeLike],
) -> tuple[CodeBlocks, list[IntegralData]]:
    """Generates code blocks from FFCx intermediate representation and
    modifies integrals code for accommodating the use of custom
    quadrature rules defined cell by cell (or facet by facet) at
    runtime.

    It also returns a list of integral data containing the required
    information for generating the custom coefficients required at
    runtime by the created integrals.

    Note:
        This function is an alternative to the original FFCx function
        `ffcx.codegeneration.codegeneration.generate_code`.
        Indeed, it first calls that function and then modifies the
        generated code for the integrals.

    Args:
        ufl_data (UFLData): UFL (analysis) data structure holding,
            among others, the UFL form data objects.
        ir (DataIR): FFCx Intermediate Representation that contains
            elements, forms, coordinate mappings, and integrals.
        ffcx_options (dict[str, int | float | npt.DTypeLike]): FFCx
            options for generating the code.

    Returns:
        tuple[CodeBlocks, list[IntegralData]]: First: generated
        blocks of code. Second: integral data containing the required
        information for generating the custom coefficients required at
        runtime by the created integrals.
    """

    if _use_language_backend():
        return _generate_code_language_backend(ufl_data, ir, ffcx_options)

    # FFCx 0.11 changed ``generate_code`` to return ``(CodeBlocks, suffixes)``;
    # qugar emits C only, so the file suffixes are not needed here.
    code_blocks, _suffixes = ffcx.codegeneration.codegeneration.generate_code(ir, ffcx_options)
    code_blocks = _modify_header(code_blocks)

    # In FEniCSx 0.10.0, ffcx emits one code block per (integral, cell_type)
    # pair, so the lists may diverge when an integral spans multiple cell
    # types. qugar currently supports a single cell type per integral, so the
    # lengths must match for the index pairing below to be valid.
    assert len(code_blocks.integrals) == len(ir.integrals), (
        "qugar requires one cell type per integral; multi-cell-type integrals "
        "(e.g. prism/pyramid facets) are not yet supported."
    )

    itg_datas = []
    for i, (header, impl) in enumerate(code_blocks.integrals):
        # Note that the integrals follow the same ordering as the
        # code blocks. See
        # ffcx.codegeneration.codegeneration.generate_code
        itg_ir = ir.integrals[i]
        itg_data = extract_integral_data(ufl_data, ir, itg_ir, ffcx_options, impl)
        itg_datas.append(itg_data)

        itg_mod = _IntegralModifier(impl, itg_ir, itg_data, ffcx_options)
        new_impl = itg_mod.create_new_code()

        code_blocks.integrals[i] = (header, new_impl)

    return code_blocks, itg_datas
