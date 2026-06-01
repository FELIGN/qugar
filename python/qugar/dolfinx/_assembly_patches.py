# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Transparent unfitted assembly for stock DOLFINx.

This module monkeypatches DOLFINx so that its *own* assemblers and
high-level solvers work on unfitted (qugar) meshes without the caller
having to use any qugar-specific form or solver class. It is applied
automatically when :mod:`qugar.dolfinx` is imported.

Two patches are enough because of how DOLFINx 0.10.0 is structured:

1. **Form compilation** — every form is compiled through
   ``dolfinx.fem.forms.form`` (aliased ``_create_form`` in
   ``dolfinx.fem.petsc``). We wrap it so that a UFL form whose integration
   domain is an unfitted (qugar) domain is routed to
   :func:`qugar.dolfinx.form_custom` (producing a
   :class:`qugar.dolfinx.CustomForm` that JITs the runtime-quadrature
   kernel and carries the custom-coefficient packer); every other form
   falls through to the original ``form``.

2. **Coefficient packing** — every assembler
   (``assemble_scalar`` / ``assemble_vector`` / ``assemble_matrix`` /
   ``apply_lifting``) and the high-level ``LinearProblem`` /
   ``NonlinearProblem`` call ``pack_coefficients(form)`` when no explicit
   ``coeffs`` are supplied. We wrap ``pack_coefficients`` so that a
   :class:`CustomForm` packs its custom coefficients (runtime quadrature
   smuggled alongside the standard ones); every other form falls through
   to the original packer.

Together these make a stock ``dolfinx.fem.petsc.LinearProblem`` (or a
direct ``dolfinx.fem.assemble_*`` call) assemble unfitted forms correctly.

Note:
    Scope of the form patch is single forms and flat sequences of forms,
    which covers every qugar demo and test. Block / nested
    (``Sequence[Sequence[...]]``) systems that *mix* fitted and unfitted
    forms are not handled specially yet: the patch routes the whole call
    to ``form_custom`` as soon as any leaf is unfitted. Purely fitted
    forms are always delegated to the original ``form``, so non-unfitted
    DOLFINx usage is unaffected beyond a cheap per-form domain check.

The patches are idempotent and guarded; the ``petsc`` rebinds are skipped
when ``petsc4py`` is unavailable.
"""

import collections.abc

from qugar.utils import has_FEniCSx, has_PETSc

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx.fem
import dolfinx.fem.assemble as _assemble_module
import dolfinx.fem.forms as _forms_module
import ufl
from dolfinx import default_scalar_type

from qugar.dolfinx.forms import CustomForm, form_custom


def _ufl_form_is_unfitted(ufl_form: ufl.Form) -> bool:
    """Returns whether a single UFL form integrates over an unfitted
    (qugar) domain.

    An unfitted domain is tagged with an ``unf_domain`` attribute on its
    UFL domain object (see
    :meth:`qugar.mesh.unfitted_domain_abc.UnfittedDomainABC.__init__`).
    """
    try:
        domains = ufl_form.ufl_domains()
    except Exception:
        return False
    return any(getattr(domain, "unf_domain", None) is not None for domain in domains)


def _tree_is_unfitted(form) -> bool:
    """Returns whether any leaf UFL form in ``form`` (a UFL form or an
    arbitrarily nested sequence of UFL forms) is unfitted."""
    if isinstance(form, ufl.Form):
        return _ufl_form_is_unfitted(form)
    if isinstance(form, collections.abc.Iterable):
        return any(_tree_is_unfitted(sub_form) for sub_form in form)
    return False


def _patch_form() -> None:
    """Route unfitted UFL forms through ``form_custom`` while leaving the
    fitted path untouched.

    Rebinds the wrapper in every namespace that holds a reference to the
    original ``form``: its definition module (``dolfinx.fem.forms``), the
    ``dolfinx.fem`` re-export, and (when available) the ``_create_form``
    alias inside ``dolfinx.fem.petsc``.
    """
    original_form = _forms_module.form
    if getattr(original_form, "_qugar_patched", False):
        return

    def form(
        form,
        dtype=default_scalar_type,
        form_compiler_options=None,
        jit_options=None,
        entity_maps=None,
    ):
        if _tree_is_unfitted(form):
            return form_custom(
                form,
                dtype=dtype,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
        return original_form(
            form,
            dtype=dtype,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

    form.__doc__ = original_form.__doc__
    form.__name__ = original_form.__name__
    form._qugar_patched = True  # type: ignore[attr-defined]
    form._qugar_original = original_form  # type: ignore[attr-defined]

    _forms_module.form = form
    dolfinx.fem.form = form

    if has_PETSc:
        import dolfinx.fem.petsc as _petsc_module

        # ``petsc.py`` did ``from dolfinx.fem.forms import form as
        # _create_form`` at its own import time, so its bound reference
        # must be replaced explicitly.
        _petsc_module._create_form = form


def _patch_pack_coefficients() -> None:
    """Dispatch ``pack_coefficients`` to a ``CustomForm``'s own packer.

    Rebinds the wrapper in every namespace that holds a reference: its
    definition module (``dolfinx.fem.assemble`` — used by the in-module
    ``assemble_vector`` / ``assemble_matrix`` / ``apply_lifting``), the
    ``dolfinx.fem`` re-export, and (when available)
    ``dolfinx.fem.petsc``.
    """
    original_pack = _assemble_module.pack_coefficients
    if getattr(original_pack, "_qugar_patched", False):
        return

    def pack_coefficients(form):
        def _pack(sub_form):
            if isinstance(sub_form, CustomForm):
                return sub_form.pack_coefficients()
            if sub_form is None:
                return {}
            if isinstance(sub_form, collections.abc.Sequence):
                return [_pack(f) for f in sub_form]
            # Plain DOLFINx form: defer to the original single-form packer.
            return original_pack(sub_form)

        return _pack(form)

    pack_coefficients.__doc__ = original_pack.__doc__
    pack_coefficients.__name__ = original_pack.__name__
    pack_coefficients._qugar_patched = True  # type: ignore[attr-defined]
    pack_coefficients._qugar_original = original_pack  # type: ignore[attr-defined]

    _assemble_module.pack_coefficients = pack_coefficients
    dolfinx.fem.pack_coefficients = pack_coefficients

    if has_PETSc:
        import dolfinx.fem.petsc as _petsc_module

        _petsc_module.pack_coefficients = pack_coefficients


def apply_patches() -> None:
    """Apply every patch needed for transparent unfitted assembly."""
    _patch_form()
    _patch_pack_coefficients()
