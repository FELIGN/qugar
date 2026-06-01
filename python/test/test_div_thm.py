# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Tests for quadrature point generation and assemblers."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import Callable

from mpi4py import MPI

import dolfinx.fem as fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from utils import (
    dtypes,  # type: ignore
    get_Gauss_quad_degree,  # type: ignore
)

import qugar.cpp
import qugar.impl
from qugar.dolfinx import CustomForm, dsu, dsu_normal, form_custom
from qugar.impl import ImplicitFunc
from qugar.mesh import UnfittedCartMesh, create_unfitted_impl_Cartesian_mesh


def create_vector_func(dlf_mesh: dolfinx.mesh.Mesh) -> ufl.Coefficient:
    """
    Create a vector function for testing the divergence theorem.

    Args:
        dlf_mesh (dolfinx.mesh.Mesh): The input mesh for which the UFL function is created.

    Returns:
        ufl.Coefficient: The generated UFL vector field as a coefficient.
    """
    x = ufl.SpatialCoordinate(dlf_mesh)
    if dlf_mesh.geometry.dim == 2:
        return ufl.as_vector([ufl.sin(x[0]) * ufl.cos(x[1]), ufl.cos(x[0]) * ufl.sin(x[1])])  # type: ignore
    else:
        return ufl.as_vector(
            [
                ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.exp(x[2]),  # type: ignore
                ufl.cos(x[0]) * ufl.sin(x[1]) * ufl.exp(x[2]),  # type: ignore
                ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.exp(x[2]),  # type: ignore
            ]
        )


def create_div_thm_volume_ufl_form(domain: UnfittedCartMesh, n_quad_pts: int, use_tags: bool):
    """
    Creates a UFL form representing the volume integral of the divergence theorem
    for a given unfitted domain.

    Args:
        domain (UnfittedImplDomain): The unfitted domain for which the UFL form is created.
        n_quad_pts (int): The number of quadrature points to be used for integration of
            cut cells.
        use_tags (bool): Flag to indicate whether to use tags for cut and full cells.
            If True, the function will create meshtags for the cut and full cells.
            If False, it will use the default meshtags.
            By setting this to False, the right management of default integrals
            eveywhere (in the presence of unfitted domains and boundaries) is tested.

    Returns:
        ufl.Form: The UFL form representing the volume integral of the divergence theorem.
    """

    quad_degree = get_Gauss_quad_degree(n_quad_pts)

    if use_tags:
        full_tag = 1
        cut_tag = 0
        cell_tags = domain.create_cell_meshtags(cut_tag=cut_tag, full_tag=full_tag)

        dx_ = ufl.dx(
            domain=domain,
            subdomain_data=cell_tags,
        )
        dx = dx_(subdomain_id=cut_tag, degree=quad_degree) + dx_(full_tag)
    else:
        dx = ufl.dx(
            domain=domain,
            degree=quad_degree,
        )

    # Note: if no specific number of quadrature points is set for the cut cells,
    # it would be enough to use a single tag for both cut and full cells.
    # and invoke dx_ only once for that tag.

    func = create_vector_func(domain)
    div_func = ufl.div(func)

    ufl_form_vol = div_func * dx
    return ufl_form_vol


def create_div_thm_surface_ufl_form(domain: UnfittedCartMesh, n_quad_pts: int, use_tags: bool):
    """
    Creates a UFL form representing the surface integral of the divergence theorem
    for a given unfitted domain.

    Args:
        domain (UnfittedImplDomain): The unfitted domain for which the UFL form is created.
        n_quad_pts (int): The number of quadrature points to be used for integration of
            cut cells and facets.
        use_tags (bool): Flag to indicate whether to use tags for cut and full facets.
            If True, the function will create subdomain data and tags for the cut and full facets.
            If False, it will use the default subdomain data and tags.
            By setting this to False, the right management of default integrals
            eveywhere (in the presence of unfitted domains and boundaries) is tested.

    Returns:
        ufl.Form: The UFL form representing the surface integral of the divergence theorem.
    """

    quad_degree = get_Gauss_quad_degree(n_quad_pts)

    if use_tags:
        cut_tag = 0
        full_tag = 1
        cell_tags = domain.create_cell_meshtags(cut_tag=cut_tag)
        facet_tags = domain.create_facet_tags(cut_tag=cut_tag, full_tag=full_tag, ext_integral=True)

        ds_unf = dsu(
            domain=domain,
            subdomain_data=cell_tags,
            subdomain_id=cut_tag,
            degree=quad_degree,
        )

        ds_ = ufl.ds(domain=domain, subdomain_data=facet_tags)
        ds = ds_(cut_tag, degree=quad_degree) + ds_(full_tag)
    else:
        ds_unf = dsu(
            domain=domain,
            degree=quad_degree,
        )

        ds = ufl.ds(domain=domain, degree=quad_degree)

    # Note: if no specific number of quadrature points is set for the cut facets,
    # it would be enough to use a single tag for both cut and full facets.
    # and invoke ds_ only once for that tag.

    bound_normal = dsu_normal(domain)
    facet_normal = ufl.FacetNormal(domain)
    func = create_vector_func(domain)

    ufl_form_srf = ufl.inner(func, bound_normal) * ds_unf + ufl.inner(func, facet_normal) * ds
    return ufl_form_srf


def check_div_thm(
    dom_func: ImplicitFunc,
    n_cells_dir: int,
    n_quad_pts: int,
    exclude_empty_cells: bool = True,
    use_tags: bool = True,
    dtype: type[np.float32 | np.float64] = np.float64,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Check the divergence theorem for a given unfitted implicit domain.

    Args:
        dom_func (ImplicitFunc): The implicit function defining the unfitted domain.
        n_cells_dir (int): Number of cells in each direction.
        n_quad_pts (int): Number of quadrature points for cut cells and facets.
        exclude_empty_cells (bool, optional): Flag to exclude empty cells in the
            unfitted mesh. Defaults to True.
        use_tags (bool): Flag to indicate whether to use tags for cut and full cells and facets.
            If True, the function will create meshtags for the cut and full cells and facets.
            If False, it will use the default meshtags.
            By setting this to False, the right management of default integrals
            eveywhere (in the presence of unfitted domains and boundaries) is tested.
        use_tags (bool): Flag to indicate whether to use tags for cut and full cells and facets.
            If True, the function will create meshtags for the cut and full cells and facets.
            If False, it will use the default meshtags.
            By setting this to False, the right management of default integrals
            eveywhere (in the presence of unfitted domains and boundaries) is tested.
        dtype (type[np.float32 | np.float64], optional): Data type for computations.
            Defaults to np.float64.
        rtol (float, optional): Relative tolerance for comparison. Defaults to 1e-5.
        atol (float, optional): Absolute tolerance for comparison. Defaults to 1e-8.

    Raises:
        AssertionError: If the volume integral is not close to the surface integral
            within the specified tolerances.
    """

    dim = dom_func.dim
    comm = MPI.COMM_WORLD
    n_cells = [n_cells_dir] * dim
    xmin = np.zeros(dim, dtype)
    xmax = np.ones(dim, dtype)
    unf_mesh = create_unfitted_impl_Cartesian_mesh(
        comm,
        dom_func,
        n_cells,
        xmin,
        xmax,
        exclude_empty_cells=exclude_empty_cells,
    )

    ufl_form_vol = create_div_thm_volume_ufl_form(unf_mesh, n_quad_pts, use_tags)
    ufl_form_srf = create_div_thm_surface_ufl_form(unf_mesh, n_quad_pts, use_tags)

    form_vol = form_custom(ufl_form_vol, dtype=dtype)
    assert isinstance(form_vol, CustomForm)

    form_srf = form_custom(ufl_form_srf, dtype=dtype)
    assert isinstance(form_srf, CustomForm)

    vol_integral = fem.assemble_scalar(form_vol, coeffs=form_vol.pack_coefficients())
    srf_integral = fem.assemble_scalar(form_srf, coeffs=form_srf.pack_coefficients())

    assert np.isclose(vol_integral, srf_integral, atol=atol, rtol=rtol)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("exclude_empty_cells", [True])
@pytest.mark.parametrize("use_tags", [True])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_disk(
    n_cells: int,
    n_quad_pts: int,
    exclude_empty_cells: bool,
    use_tags: bool,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the divergence theorem for a 2D disk unfitted domain.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        exclude_empty_cells (bool, optional): Flag to exclude empty cells in the
            unfitted mesh. Defaults to True.
        use_tags (bool): Flag to indicate whether to use tags for cut and full cells and facets.
            If True, the function will create meshtags for the cut and full cells and facets.
            If False, it will use the default meshtags.
            By setting this to False, the right management of default integrals
            eveywhere (in the presence of unfitted domains and boundaries) is tested.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function
            should be used.
    """
    radius = 0.55
    center = np.array([0.51, 0.45], dtype=dtype)

    func = qugar.impl.create_disk(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    check_div_thm(func, n_cells, n_quad_pts, exclude_empty_cells, use_tags, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("exclude_empty_cells", [True])
@pytest.mark.parametrize("use_tags", [True])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_sphere(
    n_cells: int,
    n_quad_pts: int,
    exclude_empty_cells: bool,
    use_tags: bool,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the divergence theorem for a 3D sphere unfitted domain.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        exclude_empty_cells (bool, optional): Flag to exclude empty cells in the
            unfitted mesh. Defaults to True.
        use_tags (bool): Flag to indicate whether to use tags for cut and full cells and facets.
            If True, the function will create meshtags for the cut and full cells and facets.
            If False, it will use the default meshtags.
            By setting this to False, the right management of default integrals
            eveywhere (in the presence of unfitted domains and boundaries) is tested.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
        negative (bool): Flag to indicate whether the negative of the implicit function
            should be used.
    """
    radius = 0.8
    center = np.array([0.5, 0.45, 0.35], dtype=dtype)

    func = qugar.impl.create_sphere(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)

    rtol = 1.0e-4
    atol = 1.0e-8
    check_div_thm(func, n_cells, n_quad_pts, exclude_empty_cells, use_tags, dtype, rtol, atol)



# Only one TPMS functor is exercised here: Schoen has the same code path
# as the rest but the cheapest implicit-function root finding. The other
# variants (Schoen_IWP, Schoen_FRD, Fischer_Koch_S, Schwarz_Diamond,
# Schwarz_Primitive) test the same divergence-theorem assembly.
tpms_impls = [qugar.impl.create_Schoen]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n_cells", [11])
@pytest.mark.parametrize("n_quad_pts", [8])
@pytest.mark.parametrize("impl_functor", tpms_impls)
@pytest.mark.parametrize("exclude_empty_cells", [True])
@pytest.mark.parametrize("use_tags", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("negative", [False, True])
def test_tpms(
    dim: int,
    n_cells: int,
    n_quad_pts: int,
    impl_functor: Callable,
    exclude_empty_cells: bool,
    use_tags: bool,
    dtype: type[np.float32 | np.float64],
    negative: bool,
):
    """
    Test the divergence theorem for a series of TPMS in 2D and 3.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        dim (int): Dimension. Either 2D or 3D.
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
            and facets in the test.
        impl_functor (Callable): Implicit function functor to use for the TPMS.
        exclude_empty_cells (bool, optional): Flag to exclude empty cells in the
            unfitted mesh. Defaults to True.
        use_tags (bool): Flag to indicate whether to use tags for cut and full cells and facets.
            If True, the function will create meshtags for the cut and full cells and facets.
            If False, it will use the default meshtags.
            By setting this to False, the right management of default integrals
            eveywhere (in the presence of unfitted domains and boundaries) is tested.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        negative (bool): Flag to indicate whether the negative of the implicit function
            should be used.
    """

    periods = np.ones(dim, dtype=dtype)
    impl_func = impl_functor(periods)
    if negative:
        impl_func = qugar.impl.create_negative(impl_func)

    rtol = 1e-4 if dim == 2 else 1e-3
    atol = 1e-8 if dim == 2 else 2.0e-2
    check_div_thm(
        impl_func, n_cells, n_quad_pts, exclude_empty_cells, use_tags, dtype, rtol=rtol, atol=atol
    )


if __name__ == "__main__":
    # test_disk(8, 6, False, True, np.float64, True, False)
    # test_cylinder(8, 6, np.float64, True, False)
    # test_tpms(3, 12, 8, False, np.float32, False)
    test_tpms(3, 12, 8, qugar.impl.create_Schoen, False, False, np.float32, False)
    # test_ellipsoid(8, 5, True, True, np.float32, True, False)  # , use_bzr=False)
