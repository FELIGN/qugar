# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for custom Python integrals on unfitted boundaries."""

import os

import qugar.utils

if not qugar.utils.has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import clean_cache, create_mock_unfitted_mesh, dtypes  # type: ignore

from qugar.dolfinx import CustomForm, dsu, dsu_normal, form_custom

dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("max_quad_sets", [3])
@pytest.mark.parametrize("nnz", [0.3])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplex_cell", [True, False])
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("N", [4])
def test_unfitted_normal(N, dim, p, simplex_cell, dtype, nnz, max_quad_sets):
    """Test for integral on unfitted custom boundaries.

    This test actually does not check correctness. The only thing it
    checks is that the code does not blow up when a assemblying a
    quantity integrated along the normal that contains custom normals.

    This is mainly because so far there is no easy way of checking if
    the computations are correct until we get a way of defininig custom
    quadrature points and normals.

    Args:
        N (int): Number of cells per direction in the mes.
        dim (int): Domain's dimension (either 2D or 3D).
        dx (ufl.Measure): Measure to be used (dx for cell integrals or
            ds for boundary integrals).
        p (int): Base degree to used in finite element spaces.
        simplex_cell (bool): If ``True`` simplex cells (triangles or
            tetrahedra) are used. Otherwise, quadrilaterals or
            hexahedra.
        dtype (type[np.float32 | np.float64]): `numpy` type to be used for
            scalars
        nnz (float): Ratio of entities with custom quadratures respect
            to the total number of entities. It is a value in the range
            [0.0, 1.0].
        max_quad_sets (int): Maximum number of repetitions of the
            standard quadrature to be used for each custom entity
            quadrature. For each custom entity a random number is
            generated between 1 and `max_quad_sets`.
    """

    unf_mesh = create_mock_unfitted_mesh(dim, N, simplex_cell, nnz, max_quad_sets, dtype)

    bdry_tag = 0
    unf_bdry_cells = np.sort(unf_mesh.get_cut_cells())
    cell_tags = dolfinx.mesh.meshtags(
        unf_mesh, dim, unf_bdry_cells, np.full_like(unf_bdry_cells, bdry_tag)
    )

    V0 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p))
    V1 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", p + 1))

    n = dsu_normal(unf_mesh)
    c = dolfinx.fem.Constant(unf_mesh, np.ones(dim, dtype=dtype))
    c2 = dolfinx.fem.Constant(unf_mesh, dtype(1.0))

    v = ufl.TestFunction(V0)

    e = dolfinx.fem.Function(V1, dtype=dtype)
    e.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)  # type: ignore

    ufl_form_0 = (
        ufl.dot(c, n)
        * e  # type: ignore
        * v
        * dsu(domain=unf_mesh, subdomain_data=cell_tags, subdomain_id=bdry_tag)
    )
    ufl_form_1 = (
        e
        * v  # type: ignore
        * dsu(domain=unf_mesh, subdomain_data=cell_tags, subdomain_id=bdry_tag)
    )

    ufl_form_2 = (
        c2 * v * ufl.dx(domain=unf_mesh, subdomain_data=cell_tags, subdomain_id=bdry_tag)  # type: ignore
    )

    ufl_form = ufl_form_0 + ufl_form_1 + ufl_form_2  # type: ignore

    custom_form = form_custom(ufl_form, dtype=dtype)
    assert isinstance(custom_form, CustomForm)

    custom_coeffs = custom_form.pack_coefficients()

    dolfinx.fem.assemble_vector(custom_form, coeffs=custom_coeffs)


if __name__ == "__main__":
    clean_cache()
    test_unfitted_normal(
        N=8, dim=2, p=1, simplex_cell=True, dtype=np.float32, nnz=0.0, max_quad_sets=3
    )
