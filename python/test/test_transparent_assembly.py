# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for qugar's transparent unfitted-assembly patches.

Importing ``qugar.dolfinx`` monkeypatches DOLFINx so that the *stock*
``dolfinx.fem.form`` and ``dolfinx.fem.assemble_*`` functions work on
unfitted meshes without the caller using any qugar-specific form/solver
class (see ``qugar.dolfinx._assembly_patches``). These tests pin that
behaviour:

* a UFL form on an unfitted mesh compiled with the stock
  ``dolfinx.fem.form`` becomes a ``CustomForm``;
* stock ``assemble_scalar`` / ``assemble_vector`` / ``assemble_matrix``
  with no explicit ``coeffs=`` produce the same result as the explicit
  ``form_custom(...).pack_coefficients()`` path; and
* a form on a plain (fitted) mesh is left untouched (not a
  ``CustomForm``).
"""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from mpi4py import MPI

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import (  # type: ignore
    check_vals,
    create_mock_unfitted_mesh,
    dtypes,
)

from qugar.dolfinx import CustomForm, form_custom

_N = 4
_NNZ = 0.3
_MAX_QUAD = 3


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_stock_form_becomes_custom(dim, simplex_cell, dtype):
    """``dolfinx.fem.form`` on an unfitted mesh returns a ``CustomForm``,
    while on a fitted mesh it does not."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)
    assert isinstance(dolfinx.fem.form(a, dtype=dtype), CustomForm)

    # A plain fitted mesh must fall through to the original ``form``. Match
    # the unfitted mesh's geometry dtype so the form's scalar type is
    # compatible with the mesh.
    fitted = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, _N, _N, dtype=unf.geometry.x.dtype
    )
    Vf = dolfinx.fem.functionspace(fitted, ("Lagrange", 1))
    uf, vf = ufl.TrialFunction(Vf), ufl.TestFunction(Vf)
    af = ufl.inner(ufl.grad(uf), ufl.grad(vf)) * ufl.dx
    assert not isinstance(dolfinx.fem.form(af, dtype=dtype), CustomForm)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
def test_stock_assembly_matches_explicit(dim, simplex_cell, dtype):
    """Stock ``assemble_scalar`` / ``assemble_vector`` / ``assemble_matrix``
    (no explicit ``coeffs``) match the explicit ``form_custom`` +
    ``pack_coefficients`` path on an unfitted mesh."""
    unf = create_mock_unfitted_mesh(dim, _N, simplex_cell, _NNZ, _MAX_QUAD, dtype)
    V = dolfinx.fem.functionspace(unf, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    coeff = dolfinx.fem.Function(V, dtype=dtype)
    coeff.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore

    scalar_ufl = coeff * ufl.dx(domain=unf)
    vector_ufl = ufl.inner(coeff, v) * ufl.dx(domain=unf)
    matrix_ufl = coeff * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=unf)

    # Stock path: compile with the patched dolfinx.fem.form, assemble with
    # no explicit coeffs (the patched pack_coefficients kicks in).
    s_stock = dolfinx.fem.assemble_scalar(dolfinx.fem.form(scalar_ufl, dtype=dtype))
    b_stock = dolfinx.fem.assemble_vector(dolfinx.fem.form(vector_ufl, dtype=dtype))
    A_stock = dolfinx.fem.assemble_matrix(dolfinx.fem.form(matrix_ufl, dtype=dtype))

    # Explicit qugar path.
    sf = form_custom(scalar_ufl, dtype=dtype)
    bf = form_custom(vector_ufl, dtype=dtype)
    Af = form_custom(matrix_ufl, dtype=dtype)
    s_ref = dolfinx.fem.assemble_scalar(sf, coeffs=sf.pack_coefficients())
    b_ref = dolfinx.fem.assemble_vector(bf, coeffs=bf.pack_coefficients())
    A_ref = dolfinx.fem.assemble_matrix(Af, coeffs=Af.pack_coefficients())

    check_vals(np.asarray(s_stock), np.asarray(s_ref), dtype=dtype)
    check_vals(b_stock.array, b_ref.array, dtype=dtype)
    check_vals(A_stock.to_dense(), A_ref.to_dense(), dtype=dtype)
