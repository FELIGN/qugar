# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Structured representation of an FFCx-rendered ``tabulate_tensor_*``
function body, used as the input/output of the kernel transformations
performed by :mod:`qugar.dolfinx.codegeneration`.

The previous design ran a sequence of regex passes directly over the
raw C text emitted by FFCx (split the function, erase static decls,
rewrite loop bounds, substitute normals, ...). Each pass implicitly
re-discovered the body structure and was sensitive to formatting details
of the FFCx output.

``KernelBody`` parses the FFCx C **once** into a small structured form
(signature + per-quadrature loops + the pre-/post-loop bands), so every
subsequent transformation operates on the relevant piece directly. The
regex layer remains, but each regex is confined to a single, narrow
substring rather than scanning the whole function body.

Usage pattern: parse once with :meth:`KernelBody.from_ffcx`, then build
the variant kernels with a fluent chain such as::

    (body.copy()
         .erase_static_declarations()
         .rewrite_table_accesses()
         .dynamic_loop_bounds()
         .inline_pre_loop_into_loops()
         .substitute_normals(offsets)
         .render(suffix="_custom"))

Each transformation mutates and returns ``self`` so chains read top to
bottom in transformation order. :meth:`copy` is a shallow clone of the
mutable ``loops`` list (the underlying IR is shared by reference because
it contains unpicklable ``_hashlib.HASH`` objects).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from qugar.dolfinx.parsing_utils import get_pairing_brackets, parse_dtype_C

if TYPE_CHECKING:
    from ffcx.ir.representation import IntegralIR

    from qugar.dolfinx.fe_table import FETable
    from qugar.dolfinx.integral_data import IntegralData
    from qugar.dolfinx.quadrature_data import QuadratureData

# Regexes used by the parser. Confined to this module so the
# transformation methods don't redefine them.
_FUNC_SIG_PATTERN = re.compile(
    r"void\s*tabulate_tensor_integral_(\w+)\s*\([\w|\s|\,|*]*\)"
)
_LOOP_HEADER_PATTERN = re.compile(
    r"for\s*\(\s*int\s+iq\s*=\s*0\s*;\s*iq\s*<\s*(\d+)\s*;\s*\+\+iq\s*\)"
)
_QUAD_NAME_IN_BODY = re.compile(r"_Q(\w\w\w)[\[|_e]")
_QUAD_NAME_VIA_WEIGHTS = re.compile(r"weights_(\w\w\w)")


@dataclass
class Loop:
    """A single quadrature loop pulled out of the kernel body.

    ``pre_text`` carries the text *between this loop and the previous
    one* (or, for ``loops[0]``, between the function-opening ``{`` and
    the first loop's header). ``header`` is the literal
    ``for (int iq = 0; iq < N; ++iq)`` text (so the static upper bound
    ``N`` is still parseable for the runtime-bound rewrite).
    ``body`` is the content of the loop body, excluding its outer
    ``{`` ``}`` braces.
    """

    quad_name: str
    # ``pre_text``: text between the previous loop's closing ``}`` (or
    # the function's opening ``{\n`` for ``loops[0]``) and the start of
    # this loop's ``for``.
    # ``header``: the entire span from ``for (int iq = 0; ...; ++iq)``
    # through and including the loop body's opening ``{`` -- preserves
    # any FFCx-emitted whitespace/newlines between the two.
    # ``body``: content inside the loop body, excluding the outer
    # ``{`` ``}`` braces.
    pre_text: str
    header: str
    body: str


@dataclass
class KernelBody:
    """Structured FFCx kernel body."""

    # File-level context (verbatim text outside the function).
    before_func: str
    after_func: str
    # Function signature (everything up to but excluding the opening
    # brace of the body).
    signature: str
    integral_name: str  # the FFCx-generated hash part
    # Body decomposition.
    loops: list[Loop]
    post_text: str  # text after the last loop's '}' (typically empty)
    # Per-table metadata (carries .name, .element, .derivatives,
    # .component, .funcs, .permutations, ...). Constant-for-points
    # tables are kept; everything else has its access sites rewritten
    # by the on-the-fly path.
    tables: dict[str, "FETable"] = field(default_factory=dict)
    # Quadrature data keyed by quadrature name (matches Loop.quad_name).
    quad_by_name: dict[str, "QuadratureData"] = field(default_factory=dict)
    # Form-level context, useful for transformations.
    integral_type: str = ""
    is_mixed_dim: bool = False
    tdim: int = 0
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    coeffs_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    ir: "IntegralIR | None" = None

    # ---------------------------------------------------------------
    # Parsing
    # ---------------------------------------------------------------

    @classmethod
    def from_ffcx(
        cls,
        code: str,
        itg_ir: "IntegralIR",
        itg_data: "IntegralData",
    ) -> "KernelBody":
        """Parse one FFCx-emitted ``tabulate_tensor_integral_*`` function.

        Side-effects: none. The returned object is independent of the
        input ``code`` string -- consumers may mutate it freely.
        """
        # ---- split the file into (before_func, signature, body, after_func)
        m = _FUNC_SIG_PATTERN.search(code)
        if m is None:
            raise ValueError("FFCx integral function not found in code.")
        before_func = code[: m.start()]
        signature = m.group(0)
        integral_name = m.group(1)

        sub = code[m.end():]
        ob = sub.find("{")
        if ob < 0:
            raise ValueError("Opening '{' of function body not found.")
        # Locate the function's standalone closing '}': it's the LAST `}`
        # in the file that's immediately followed by `\n`. Struct
        # definitions like `ufcx_integral ... = { ... };` have a `}`
        # followed by `;` instead, so they don't match. This matches
        # what the legacy _split_code did.
        rev = sub[::-1]
        rm = rev.find("\n}")  # FIRST "\n}" in reversed = LAST "}\n" in original
        if rm < 0:
            raise ValueError("Closing '}' of function body not found.")
        cb = len(sub) - rm + 1  # one past the trailing '\n' after '}'

        func_impl = sub[ob:cb]
        after_func = sub[cb:]

        # ---- coeffs dtype: parse from the 'w' parameter type in the signature
        wm = re.search(r"const\s+([^\*]+)\*\s+restrict\s+w", signature)
        if wm is None:
            raise ValueError("Could not find 'w' parameter in signature.")
        coeffs_dtype = np.dtype(parse_dtype_C(wm.group(1)))

        # ---- body decomposition: pre_loop / [Loop, ...] / post_loop
        # ``func_impl`` starts with "{\n" and ends past the function's
        # final '}' (plus any trailing newlines included by the boundary
        # search). Split inside at the LAST '}' so the tail captures the
        # close-brace plus whatever whitespace followed it.
        if not func_impl.startswith("{\n"):
            raise ValueError("Function body does not start with '{\\n'.")
        inside = func_impl[2:]  # everything after the opening "{\n"
        close_pos = inside.rfind("}")
        if close_pos < 0:
            raise ValueError("Function body does not end with '}'.")
        inside_body = inside[:close_pos]
        tail = inside[close_pos:]  # '}' + any trailing whitespace

        loops: list[Loop] = []
        ind = 0
        for lm in _LOOP_HEADER_PATTERN.finditer(inside_body):
            pre_text = inside_body[ind:lm.start()]
            after_header = inside_body[lm.end():]
            b_open, b_close = get_pairing_brackets(after_header)
            # Ensure nothing non-whitespace sits between the for-header
            # and the body's '{' (FFCx output puts a newline there).
            pre_brace = after_header[:b_open]
            if pre_brace.strip():
                raise ValueError(
                    f"Unexpected text between loop header and body: "
                    f"{pre_brace!r}"
                )
            # ``header`` keeps everything from ``for (...)`` through the
            # opening ``{`` (preserves FFCx's "\n{" formatting).
            header = inside_body[lm.start():lm.end() + b_open + 1]
            body = after_header[b_open + 1:b_close - 1]
            # Identify the quadrature this loop belongs to.
            quad = _QUAD_NAME_IN_BODY.search(body)
            if quad is None:
                quad = _QUAD_NAME_VIA_WEIGHTS.search(body)
            if quad is None:
                raise ValueError("Quadrature name not found in loop body.")
            loops.append(
                Loop(
                    quad_name=quad.group(1),
                    pre_text=pre_text,
                    header=header,
                    body=body,
                )
            )
            # Advance to the position right after the loop's closing '}'.
            ind = lm.end() + b_close

        post_text = inside_body[ind:] + tail

        # ---- per-table metadata (mirror integral_data.extract_integral_data)
        tables: dict[str, "FETable"] = {}
        for table_list in itg_data.quad_data_FE_tables.values():
            for t in table_list:
                tables[t.name] = t

        # ---- quadrature lookup by name (used by transformations)
        quad_by_name: dict[str, "QuadratureData"] = {
            qd.name: qd for qd in itg_data.quad_data_FE_tables.keys()
        }

        # ---- form-level context
        integral_type = itg_data.integral_type
        is_mixed_dim = itg_data.is_mixed_dim
        tdim = itg_data.tdim
        dtype = np.dtype(itg_data.dtype)

        return cls(
            before_func=before_func,
            after_func=after_func,
            signature=signature,
            integral_name=integral_name,
            loops=loops,
            post_text=post_text,
            tables=tables,
            quad_by_name=quad_by_name,
            integral_type=integral_type,
            is_mixed_dim=is_mixed_dim,
            tdim=tdim,
            dtype=dtype,
            coeffs_dtype=coeffs_dtype,
            ir=itg_ir,
        )

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------

    def render_body(self) -> str:
        """Render only the function body (``{ ... }``) back to C."""
        parts = ["{\n"]
        for loop in self.loops:
            parts.append(loop.pre_text)
            parts.append(loop.header)  # includes the body-opening '{'
            parts.append(loop.body)
            parts.append("}")
        parts.append(self.post_text)
        # post_text already carries the function's trailing '}' (or
        # '\n}') as captured by the parse, so don't add another here.
        return "".join(parts)

    def render(self, suffix: str = "") -> str:
        """Render the full function (signature + body) with an optional
        name suffix, e.g. ``"_original"`` or ``"_custom"``."""
        sig = self.signature
        if suffix:
            sig = sig.replace(
                self.integral_name, self.integral_name + suffix, 1
            )
        return sig + self.render_body() + "\n\n"

    # ---------------------------------------------------------------
    # Transformations (mutate self, return self for chaining)
    # ---------------------------------------------------------------

    def copy(self) -> "KernelBody":
        """Fork the body so transformations don't leak into the parent.

        Only the mutable pieces are cloned (loops; everything else is
        either an immutable string/scalar or a read-only reference like
        ``tables`` / ``quad_by_name`` / ``ir`` -- the latter contains
        non-deepcopyable objects such as basix hash handles).
        """
        return replace(self, loops=[replace(L) for L in self.loops])

    def has_unfitted_boundary(self) -> bool:
        """Whether any quadrature in the integral targets an unfitted
        boundary (``ds_bdry_unf``)."""
        return any(
            qd.unfitted_boundary for qd in self.quad_by_name.values()
        )

    def inline_pre_loop_into_loops(self) -> "KernelBody":
        """Move the pre-loop band into every loop's body and clear it.

        FFCx puts cell-affine setup (Jacobian for linear simplices, etc.)
        before the first quadrature loop. Downstream transformations that
        rewrite per-point quantities (normals substitution on the custom
        variant; ``break;`` short-circuiting on the original variant) need
        this setup *inside* each loop's body. Subsequent loops' inter-loop
        whitespace in ``Loop.pre_text`` is left untouched.
        """
        if not self.loops:
            return self
        cnt_vars = self.loops[0].pre_text
        self.loops[0].pre_text = ""
        if cnt_vars:
            for loop in self.loops:
                loop.body = cnt_vars + loop.body
        return self

    def break_unfitted_loops(self) -> "KernelBody":
        """Prepend ``break;`` to every loop body whose quadrature has
        ``unfitted_boundary=True``.

        Used by the ``_original`` variant of the kernel: when dolfinx
        invokes it for a non-custom cell, ``unfitted_boundary`` loops
        make no sense and the ``break;`` lets the compiler dead-code-
        eliminate them entirely.
        """
        for loop in self.loops:
            qd = self.quad_by_name.get(loop.quad_name)
            if qd is not None and qd.unfitted_boundary:
                loop.body = "break;\n" + loop.body
        return self

    def null_unfitted_normals(self) -> "KernelBody":
        """Replace every ``normals_<quad>[...]`` access belonging to an
        unfitted-boundary quadrature with ``0.0``, across all loop bodies.

        This is needed for the ``_original`` kernel variant. FFCx (via the
        backend patch in :mod:`qugar.dolfinx._ffcx_patches`) emits
        ``normals_<quad>[tdim * iq + i]`` accesses unconditionally —
        including in the ``_original`` kernel where the array is never
        declared. Even though the unfitted loops are dead-code-eliminated
        by ``break_unfitted_loops``, the C compiler still type-checks the
        unreachable body. Replacing the accesses with ``0.0`` makes the
        original kernel compile cleanly.

        Must be called after :meth:`inline_pre_loop_into_loops` (the
        normals expression is in the inlined pre-loop text, which ends up
        in every loop's body).
        """
        unfitted_names = {
            name for name, qd in self.quad_by_name.items() if qd.unfitted_boundary
        }
        if not unfitted_names:
            return self
        for loop in self.loops:
            for qname in unfitted_names:
                loop.body = re.sub(
                    rf"normals_{qname}\[[^\]]*\]", "0.0", loop.body
                )
        return self

    def erase_static_declarations(self) -> "KernelBody":
        """Strip FFCx's ``// Quadrature rules`` weights block and the
        ``static const ... FE..._Q...[][][][] = {...};`` table
        declarations from the pre-loop band of the body.

        Used by the ``_custom`` variant: those values are loaded
        dynamically (weights / table buffers from the shim) instead.
        """
        if not self.loops:
            return self
        text = self.loops[0].pre_text
        qm = re.search(r"// Quadrature rules", text)
        if qm is None:
            return self
        # End of last `static const ... FE...;` declaration.
        last_end = None
        for fm in re.finditer(r"static\s+const\s+\w+\s+FE[^;]*;\s*\n", text):
            last_end = fm.end()
        if last_end is None:
            return self
        self.loops[0].pre_text = text[: qm.start()] + text[last_end:]
        return self

    def rewrite_table_accesses(self) -> "KernelBody":
        """Rewrite ``FE..[perm][entity][iq][dof]`` (4-D FFCx accesses) to
        the qugar 1-D form used by the dynamically loaded buffers.

        ``constant_for_pts`` tables keep their 4-D static-array access
        (their declarations are re-emitted by the caller-side codegen).
        """

        bracket = r"\[\s*(\w+\[\d+\]|[^\]]*)\s*\]\s*"
        fe_re = re.compile(r"FE([\w|\_]+)\s*" + bracket * 4)
        has_perms = self.integral_type == "interior_facet"

        def transform(text: str) -> str:
            out: list[str] = []
            pos = 0
            for m in fe_re.finditer(text):
                table = self.tables.get(f"FE{m.group(1)}")
                if table is None:
                    continue  # leave unknown accesses alone (shouldn't happen)
                if table.is_constant_for_pts():
                    out.append(text[pos: m.end()])
                else:
                    indices = tuple(m.group(i) for i in range(2, 6))
                    out.append(text[pos: m.start()])
                    out.append(table.create_new_access_code(indices, has_perms))
                pos = m.end()
            out.append(text[pos:])
            return "".join(out)

        # Apply to the pre-loop band and every loop body. Loop headers
        # never contain FE accesses; post_text is just the function's
        # closing '}' so it doesn't either.
        if self.loops:
            self.loops[0].pre_text = transform(self.loops[0].pre_text)
        for loop in self.loops:
            loop.body = transform(loop.body)
        return self

    def dynamic_loop_bounds(self) -> "KernelBody":
        """Rewrite each loop header from FFCx's compile-time upper bound
        ``for (int iq = 0; iq < N; ++iq)`` to qugar's runtime form
        ``for (int iq = 0; iq < n_pts_Q<quad_name>; ++iq)``.
        """
        for loop in self.loops:
            loop.header = re.sub(
                r"for\s*\(\s*int\s+iq\s*=\s*0\s*;\s*iq\s*<\s*\d+\s*;\s*\+\+iq\s*\)",
                f"for (int iq = 0; iq < n_pts_Q{loop.quad_name}; ++iq)",
                loop.header,
                count=1,
            )
        return self

    def substitute_normals(self, normal_constant_offsets: list[int]) -> "KernelBody":
        """For every loop whose quadrature has ``unfitted_boundary=True``,
        replace references to the fake-normal-constant slots
        ``c[ct_offset+i]`` with ``normals_<quad>[tdim * iq + i]``.

        ``normal_constant_offsets`` is the list of base offsets returned
        by :meth:`_IntegralModifier._get_constant_normal_offsets`.
        """
        if not normal_constant_offsets:
            return self
        for loop in self.loops:
            qd = self.quad_by_name.get(loop.quad_name)
            if qd is None or not qd.unfitted_boundary:
                continue
            block = loop.body
            for ct_off in normal_constant_offsets:
                for i in range(self.tdim):
                    pat = rf"([*+\-/%\s(])c\[{int(ct_off) + i}\]"
                    idx = f"{self.tdim} * iq" + (f" + {i}" if i > 0 else "")
                    repl_to = f"normals_{loop.quad_name}[{idx}]"
                    block = re.sub(
                        pat,
                        lambda m, r=repl_to: f"{m.group(1)}{r}",
                        block,
                    )
            loop.body = block
        return self
