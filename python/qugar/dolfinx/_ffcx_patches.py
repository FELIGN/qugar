# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""FFCx backend hooks for unfitted-boundary normals.

qugar represents the per-point boundary normal as a custom UFL geometric
terminal (:class:`qugar.dolfinx.boundary.UnfittedReferenceNormal`). FFCx
does not know how to lower an unknown terminal, so this module teaches its
code-generation backend to emit a structural access into the per-cell
``normals_<quad>`` array that qugar smuggles in through ``custom_data``.

This replaces the previous approach of rewriting the generated C text with
a regular expression: instead of post-processing ``c[k]`` constant
accesses, the terminal is lowered to ``normals_<quad>[tdim * iq + i]``
directly by FFCx, through the same type-dispatched handler tables FFCx
uses for its own geometric quantities (``FFCXBackendAccess.call_lookup``
and ``FFCXBackendDefinitions.handler_lookup``).

The patch couples qugar to FFCx backend internals, but only to stable
class-level dispatch dictionaries — far less brittle than text surgery on
emitted C.
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import ffcx.codegeneration.access as _access
import ffcx.codegeneration.definitions as _definitions
import ffcx.codegeneration.integral_generator as _integral_generator
import ffcx.codegeneration.lnodes as L
import ufl

from qugar.dolfinx.boundary import UnfittedReferenceNormal


def _unfitted_normal_access(self, mt, tabledata, quadrature_rule):
    """FFCx access handler for :class:`UnfittedReferenceNormal`.

    Emits ``normals_<quad>[tdim * iq + component]``, i.e. an access into
    the per-cell normals array threaded in via ``custom_data``. The array
    name matches qugar's quadrature name (``QuadratureRule.id()``), which
    is the same identifier FFCx uses for its own ``weights_<quad>`` /
    ``points_<quad>`` arrays for this integral.
    """
    domain = ufl.domain.extract_unique_domain(mt.terminal)
    tdim = domain.topological_dimension()
    component = mt.component[0]
    iq = self.symbols.quadrature_loop_index

    # The terminal is cellwise-constant, so FFCx lowers it in the
    # piecewise partition, which calls the access backend with
    # ``quadrature_rule=None``. Recover the rule recorded by the patched
    # ``generate_piecewise_partition`` (see :func:`_patch_ffcx_backend`).
    rule = quadrature_rule if quadrature_rule is not None else self._qugar_current_rule
    if rule is None:
        raise RuntimeError("unfitted normal lowered outside a quadrature context")

    # ``QuadratureRule.id()`` needs the hash to have been computed.
    hash(rule)
    quad_name = rule.id()

    normals = L.Symbol(f"normals_{quad_name}", dtype=L.DataType.REAL)
    return normals[tdim * iq + component]


def _patch_ffcx_backend() -> None:
    """Register the unfitted-normal handlers on the FFCx backend.

    Three small, idempotent patches against stable FFCx internals:

    * ``FFCXBackendAccess.__init__`` / ``FFCXBackendDefinitions.__init__``
      are wrapped so every freshly built backend gets our terminal added
      to its dispatch table.
    * ``IntegralGenerator.generate_piecewise_partition`` is wrapped to
      record the current quadrature rule on the access backend, since the
      piecewise partition otherwise passes ``quadrature_rule=None`` down
      to the access handler.
    """
    access_cls = _access.FFCXBackendAccess
    defs_cls = _definitions.FFCXBackendDefinitions
    gen_cls = _integral_generator.IntegralGenerator

    if getattr(access_cls, "_qugar_patched", False):
        return

    _orig_access_init = access_cls.__init__
    _orig_defs_init = defs_cls.__init__
    _orig_piecewise = gen_cls.generate_piecewise_partition

    def access_init(self, *args, **kwargs):
        _orig_access_init(self, *args, **kwargs)
        self._qugar_current_rule = None
        # Bind the handler as a method of this backend instance so it can
        # reach ``self.symbols``.
        self.call_lookup[UnfittedReferenceNormal] = _unfitted_normal_access.__get__(self)

    def defs_init(self, *args, **kwargs):
        _orig_defs_init(self, *args, **kwargs)
        # No definition needed: the value is read directly at the access
        # site (same as FFCx's own ReferenceNormal).
        self.handler_lookup[UnfittedReferenceNormal] = self.pass_through

    def generate_piecewise_partition(self, quadrature_rule, domain):
        self.backend.access._qugar_current_rule = quadrature_rule
        try:
            return _orig_piecewise(self, quadrature_rule, domain)
        finally:
            self.backend.access._qugar_current_rule = None

    access_cls.__init__ = access_init
    defs_cls.__init__ = defs_init
    gen_cls.generate_piecewise_partition = generate_piecewise_partition
    access_cls._qugar_patched = True  # type: ignore[attr-defined]


def apply_patches() -> None:
    """Apply every FFCx backend patch needed by qugar."""
    _patch_ffcx_backend()
