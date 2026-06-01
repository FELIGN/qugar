# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

(
    """Tests for quadrature point generation and assemblers - computes the volume """
    """ and boundary area of domains."""
)

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")


from typing import Optional

from mpi4py import MPI

import dolfinx.fem
import numpy as np
import pytest
import ufl
from utils import (
    dtypes,  # type: ignore
    get_Gauss_quad_degree,  # type: ignore
)

import qugar.cpp
import qugar.impl
from qugar.dolfinx import CustomForm, dsu, form_custom
from qugar.impl import ImplicitFunc
from qugar.mesh import UnfittedDomain, create_unfitted_impl_Cartesian_mesh


def compute_volume(
    unf_mesh: UnfittedDomain,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
) -> np.float32 | np.float64:
    """
    Computes the volume of a given unfitted domain.

    Args:
        mesh (UnfittedMesh): The unfitted mesh for which the volume is to be computed.
        dtype (type[np.float32 | np.float64]): The data type to be used for computations.

    Returns:
        np.float32 | np.float64: The computed volume of the domain.
    """

    cut_tag = 0
    full_tag = 1
    cell_tags = unf_mesh.create_cell_meshtags(cut_tag=cut_tag, full_tag=full_tag)

    one = dolfinx.fem.Constant(unf_mesh, dtype(1.0))

    quad_degree = get_Gauss_quad_degree(n_quad_pts)
    dx = ufl.dx(domain=unf_mesh, subdomain_data=cell_tags)
    dx_cut = dx(
        subdomain_id=cut_tag,
        degree=quad_degree,
    )
    ufl_form = one * (dx(full_tag) + dx_cut)

    custom_form = form_custom(ufl_form, dtype=dtype)
    assert isinstance(custom_form, CustomForm)

    custom_coeffs = custom_form.pack_coefficients()

    return dolfinx.fem.assemble_scalar(custom_form, coeffs=custom_coeffs)


def compute_boundary_area(
    unf_mesh: UnfittedDomain,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
) -> np.float32 | np.float64:
    """
    Computes the boundary area of a given unfitted mesh.

    Args:
        unf_mesh (UnfittedMesh): The unfiffted mesh for which the volume is to be computed.
        dtype (type[np.float32 | np.float64]): The data type to be used for computations.

    Returns:
        np.float32 | np.float64: The computed area of the domain's boundary.
    """

    bdry_tag = 0
    cell_tags = unf_mesh.create_cell_meshtags(cut_tag=bdry_tag)

    one = dolfinx.fem.Constant(unf_mesh, dtype(1.0))

    quad_degree = get_Gauss_quad_degree(n_quad_pts)
    ds = dsu(
        domain=unf_mesh,
        subdomain_data=cell_tags,
        subdomain_id=bdry_tag,
        degree=quad_degree,
    )
    ufl_form = one * ds

    custom_form = form_custom(ufl_form, dtype=dtype)
    assert isinstance(custom_form, CustomForm)

    custom_coeffs = custom_form.pack_coefficients()

    return dolfinx.fem.assemble_scalar(custom_form, coeffs=custom_coeffs)


def volume_test_and_area_test(
    func: ImplicitFunc,
    n_cells_dir: int,
    n_quad_pts: int,
    exact_volume: np.float32 | np.float64,
    exact_area: Optional[np.float32 | np.float64] = None,
    dtype: type[np.float32 | np.float64] = np.float64,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """
    Tests the volume and area of an implicit function's domain by comparing computed
    against exact values.

    Args:
        func (ImplicitFunc): The implicit function defining the domain.
        n_cells_dir (int): Number of cells in each direction for the Cartesian mesh.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        exact_volume (np.float32 | np.float64): The exact volume to compare against.
        exact_area (Optional[np.float32 | np.float64], optional): The exact area to compare against.
            Defaults to None. If None, the area is not tested.
        dtype (type[np.float32 | np.float64], optional): Data type for computations.
            Defaults to np.float64.
        rtol (floating): Relative tolerance to check volume and error area.
            See `numpy.isclose` for further details.
        atol (floating): Absolute tolerance to check volume and error area.
            See `numpy.isclose` for further details.

    Raises:
        AssertionError: If the computed volume does not match the exact volume.
        AssertionError: If the computed area does not match the exact area (if provided).
    """
    dim = func.dim
    comm = MPI.COMM_WORLD
    n_cells = [n_cells_dir] * dim
    xmin = np.zeros(dim, dtype)
    xmax = np.ones(dim, dtype)
    domain = create_unfitted_impl_Cartesian_mesh(comm, func, n_cells, xmin, xmax)

    if exact_volume is not None:
        volume = compute_volume(domain, n_quad_pts, dtype)
        assert np.isclose(volume, exact_volume, rtol=rtol, atol=atol)

    if exact_area is not None:
        area = compute_boundary_area(domain, n_quad_pts, dtype)
        assert np.isclose(area, exact_area, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_disk(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 2D disk.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """
    radius = 0.4
    center = np.array([0.5, 0.5], dtype=dtype)

    exact_volume = dtype(np.pi * radius**2)
    exact_area = dtype(2.0 * np.pi * radius)

    func = qugar.impl.create_disk(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, exact_area, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_sphere(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 3D sphere.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """
    radius = 0.4
    center = np.array([0.5, 0.5, 0.5], dtype=dtype)

    exact_volume = dtype(4.0 / 3.0 * np.pi * radius**3)
    exact_area = dtype(4.0 * np.pi * radius**2)

    func = qugar.impl.create_sphere(radius=radius, center=center, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    rtol = 1e-4
    atol = 1e-8
    volume_test_and_area_test(
        func, n_cells, n_quad_pts, exact_volume, exact_area, dtype, rtol, atol
    )


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_ellipse(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 2D ellipse.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.5, 0.5], dtype=dtype)
    x_axis = np.array([1.0, 1.0], dtype=dtype)
    ref_system = qugar.cpp.create_ref_system(center, x_axis)
    semi_axes = np.array([0.4, 0.2], dtype=dtype)

    exact_volume = dtype(np.pi * np.prod(semi_axes))

    func = qugar.impl.create_ellipse(semi_axes=semi_axes, ref_system=ref_system, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, None, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_ellipsoid(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 3D ellipsoid.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.5, 0.5, 0.5], dtype=dtype)
    x_axis = np.array([1.0, 1.0, 1.0], dtype=dtype)
    y_axis = np.array([-1.0, 1.0, 1.0], dtype=dtype)
    ref_system = qugar.cpp.create_ref_system(center, x_axis, y_axis)
    semi_axes = np.array([0.4, 0.2, 0.3], dtype=dtype)

    exact_volume = dtype(4.0 / 3.0 * np.pi * np.prod(semi_axes))

    func = qugar.impl.create_ellipsoid(semi_axes=semi_axes, ref_system=ref_system, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    rtol = 1e-4
    atol = 1e-8
    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, None, dtype, rtol, atol)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [5])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_cylinder(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 3D cylinder.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    origin = np.array([0.55, 0.45, 0.47], dtype=dtype)
    axis = np.array([0.0, 0.0, 1.0], dtype=dtype)
    radius = 0.4

    exact_volume = dtype(np.pi * radius**2)
    exact_area = dtype(2.0 * np.pi * radius)

    func = qugar.impl.create_cylinder(radius=radius, origin=origin, axis=axis, use_bzr=use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, exact_area, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [6])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_annulus(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 2D annulus.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.55, 0.47], dtype=dtype)
    outer_radius = 0.4
    inner_radius = 0.2

    exact_volume = dtype(np.pi * (outer_radius**2 - inner_radius**2))
    exact_area = dtype(2.0 * np.pi * (outer_radius + inner_radius))

    func = qugar.impl.create_annulus(inner_radius, outer_radius, center, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, exact_area, dtype)


@pytest.mark.parametrize("n_cells", [12])
@pytest.mark.parametrize("n_quad_pts", [8])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_torus(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 3D torus.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    center = np.array([0.55, 0.47, 0.51], dtype=dtype)
    axis = np.array([1.0, 0.9, 0.8], dtype=dtype)
    major_radius = 0.35
    minor_radius = 0.15

    exact_volume = dtype(2.0 * np.pi * major_radius * np.pi * minor_radius**2)
    exact_area = dtype(2.0 * np.pi * major_radius * 2.0 * np.pi * minor_radius)

    func = qugar.impl.create_torus(major_radius, minor_radius, center, axis, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    rtol = 1e-3
    atol = 1e-8
    volume_test_and_area_test(
        func, n_cells, n_quad_pts, exact_volume, exact_area, dtype, rtol, atol
    )


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [6])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_line(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 2D domain whose boundary is a straigth line.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    a = 0.2
    origin = np.array([0.5, 0.5], dtype=dtype)
    normal = np.array([1.0, a], dtype=dtype)

    exact_volume = dtype(0.5)
    exact_area = dtype(np.sqrt(1.0 + a**2))

    func = qugar.impl.create_line(origin, normal, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, exact_area, dtype)


@pytest.mark.parametrize("n_cells", [8])
@pytest.mark.parametrize("n_quad_pts", [6])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("use_bzr", [True, False])
@pytest.mark.parametrize("negative", [False, True])
def test_plane(
    n_cells: int,
    n_quad_pts: int,
    dtype: type[np.float32 | np.float64],
    use_bzr: bool,
    negative: bool,
):
    """
    Test the volume and area computation of a 3D domain whose boundary is a plane.

    Note:
        Asserts raise if the computed volume or area do not match the exact values
        (up to a tolerance).

    Args:
        n_cells (int): Number of cells per direction in the Cartesian mesh to use in the test.
        n_quad_pts (int): Number of quadrature points per direction to use for the cut cells
           in the test.
        dtype (type): Data type to use for numerical operations (np.float32 or np.float64).
        use_bzr (bool): Flag to indicate whether to use Bezier representation for the implicit
            function.
    """

    a = 0.2
    origin = np.array([0.5, 0.5, 0.5], dtype=dtype)
    normal = np.array([1.0, a, 0.0], dtype=dtype)

    exact_volume = dtype(0.5)
    exact_area = dtype(np.sqrt(1.0 + a**2))

    func = qugar.impl.create_plane(origin, normal, use_bzr)
    if negative:
        func = qugar.impl.create_negative(func)
        exact_volume = dtype(1.0) - exact_volume

    volume_test_and_area_test(func, n_cells, n_quad_pts, exact_volume, exact_area, dtype)


if __name__ == "__main__":
    # test_disk(8, 5, np.float32, True, False)
    test_sphere(8, 5, np.float32, False, False)
    # test_ellipse(8, 5, np.float32, True)
    # test_ellipsoid(8, 5, np.float32, True, False)
    # test_annulus(8, 6, np.float32, True, False)
    # test_torus(12, 8, np.float32, False, False)
    # test_line(8, 2, np.float64, True, False)
    # test_plane(8, 2, np.float64, True, False)
