# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

import qugar.cpp
import qugar.impl
import qugar.utils
from qugar.cpp import ImplicitFunc_2D, ImplicitFunc_3D


class ImplicitFunc:
    """Class for storing an implicit function that describes a domain."""

    def __init__(self, cpp_object: ImplicitFunc_2D | ImplicitFunc_3D) -> None:
        """Constructor.

        Args:
            cpp_object (ImplicitFunc_2D | ImplicitFunc_3D):
                Already generated implicit function binary object.
        """

        self._cpp_object = cpp_object

    @property
    def cpp_object(self) -> ImplicitFunc_2D | ImplicitFunc_3D:
        """Returns the C++ object.

        Returns:
            ImplicitFunc_2D | ImplicitFunc_3D: Underlying function C++ object.
        """
        return self._cpp_object

    @property
    def dim(self) -> int:
        """Gets the dimension of the function's domain.

        Returns:
            int: Function's dimension.
        """
        return self._cpp_object.dim

    def eval(
        self, points: npt.NDArray[np.float32 | np.float64]
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluates the implicit function at the given points.

        Args:
            points (npt.NDArray[np.float32 | np.float64]):
                Points to evaluate the implicit function.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Evaluated values of the implicit function.
        """
        assert points.shape[1] == self.dim, (
            "Invalid points dimension. It must match the function's dimension"
        )

        points_ = points.astype(np.float64) if points.dtype == np.float32 else points
        vals = self._cpp_object.eval(points_)
        return vals.astype(np.float32) if points.dtype == np.float32 else vals

    def grad(
        self, points: npt.NDArray[np.float32 | np.float64]
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluates the implicit function gradient at the given points.

        Args:
            points (npt.NDArray[np.float32 | np.float64]):
                Points to evaluate the implicit function.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Evaluated gradients
            values of the implicit function.
        """
        assert points.shape[1] == self.dim, (
            "Invalid points dimension. It must match the function's dimension"
        )

        points_ = points.astype(np.float64) if points.dtype == np.float32 else points
        vals = self._cpp_object.grad(points_)
        return vals.astype(np.float32) if points.dtype == np.float32 else vals


def create_disk(
    radius: float | np.float32 | np.float64,
    center: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates a 2D disk with the given radius and optional center.

    Args:
        radius (float | np.float32 | np.float64): The radius of the disk.
        center (Optional[npt.NDArray[np.float32 | np.float64]], optional):
            The center of the disk. Defaults to None. If not provided,
            the radius is centered at the origin.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to False.

    Returns:
        ImplicitFunc: The implicit function representing the disk.

    Raises:
        AssertionError: If the radius is not positive.
        AssertionError: If the origin array, if provided, does not have exactly 2 coordinates.
    """

    radius = np.float64(radius)
    assert radius > 0.0, "Invalid radius. It must be positive"

    if center is None:
        cpp_obj = qugar.cpp.create_disk(radius, use_bzr)
    else:
        assert center.size == 2, "Invalid center dimension. It must have 2 coordinates"
        cpp_obj = qugar.cpp.create_disk(radius, center.astype(np.float64), use_bzr)
    return ImplicitFunc(cpp_obj)


def create_sphere(
    radius: float | np.float32 | np.float64,
    center: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates a 3D sphere with the given radius and optional center.

    Args:
        radius (float | np.float32 | np.float64): The radius of the sphere.
        center (Optional[npt.NDArray[np.float32 | np.float64]], optional):
            The center of the sphere. Defaults to None. If not provided,
            the radius is centered at the origin.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: The implicit function representing the sphere.

    Raises:
        AssertionError: If the radius is not positive.
        AssertionError: If the origin array, if provided, does not have exactly 3 coordinates.
    """

    radius = np.float64(radius)
    assert radius > 0.0, "Invalid radius. It must be positive"

    if center is None:
        cpp_obj = qugar.cpp.create_sphere(radius, use_bzr)
    else:
        assert center.size == 3, "Invalid center dimension. It must have 3 coordinates"
        cpp_obj = qugar.cpp.create_sphere(radius, center.astype(np.float64), use_bzr)
    return ImplicitFunc(cpp_obj)


def create_ellipse(
    semi_axes: npt.NDArray[np.float32 | np.float64],
    ref_system: Optional[qugar.cpp.RefSystem_2D] = None,
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates a 2D ellipse with the given semi-axes and optional reference system.

    Args:
        semi_axes (Optional[npt.NDArray[np.float32 | np.float64]], optional):
            The semi-axes of the ellipse. It must have two entries.
        ref_system (Optional[qugar.cpp.RefSystem], optional): The reference system of the ellipse.
            Defaults to None. If not provided, the ellipse is aligned with the Cartesian system.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: The implicit function representing the ellipse.

    Raises:
        AssertionError: If any of the semi_axes is not positive or it does
            not have exactly 2 coordinates.
    """

    assert semi_axes.size == 2, "Invalid semi-axes dimension. It must have 2 coordinates"
    assert np.all(semi_axes > 0.0), "Invalid semi-axes. They must be all positive"
    semi_axes = semi_axes.astype(np.float64)

    if ref_system is None:
        cpp_obj = qugar.cpp.create_ellipse(semi_axes.astype(np.float64), use_bzr)
    else:
        cpp_obj = qugar.cpp.create_ellipse(semi_axes.astype(np.float64), ref_system, use_bzr)
    return ImplicitFunc(cpp_obj)


def create_ellipsoid(
    semi_axes: npt.NDArray[np.float32 | np.float64],
    ref_system: Optional[qugar.cpp.RefSystem_3D] = None,
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates a 3D ellipsoid with the given semi-axes and optional reference system.

    Args:
        semi_axes (Optional[npt.NDArray[np.float32 | np.float64]], optional):
            The semi-axes of the ellipsoid. It must have three entries.
        ref_system (Optional[qugar.cpp.RefSystem], optional): The reference system of the ellipsoid.
            Defaults to None. If not provided, the ellipsoid is aligned with the Cartesian system.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: The implicit function representing the ellipsoid.

    Raises:
        AssertionError: If any of the semi_axes is not positive or it does
            not have exactly 3 coordinates.
    """

    assert semi_axes.size == 3, "Invalid semi-axes dimension. It must have 3 coordinates"
    semi_axes = semi_axes.astype(np.float64)
    semi_axes = semi_axes.astype(np.float64)

    if ref_system is None:
        cpp_obj = qugar.cpp.create_ellipsoid(semi_axes.astype(np.float64), use_bzr)
    else:
        cpp_obj = qugar.cpp.create_ellipsoid(semi_axes.astype(np.float64), ref_system, use_bzr)
    return ImplicitFunc(cpp_obj)


def create_cylinder(
    radius: float | np.float32 | np.float64,
    origin: npt.NDArray[np.float32 | np.float64],
    axis: Optional[npt.NDArray[np.float32 | np.float64]],
    use_bzr: bool = True,
) -> ImplicitFunc:
    """Creates a 3D cylinder with the given radius, origin, and axis.

    Args:
        radius (float | np.float32 | np.float64): Cylider's radius.
        origin (npt.NDArray[np.float32 | np.float64]): Cylinder's origin
            (a point in the revolution axis).
        axis (Optional[npt.NDArray[np.float32 | np.float64]]):
            Cylinder's axis. If not provided, the cylinder is aligned with the z-axis.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: _description_

    Raises:
        AssertionError: If the radius is not positive.
        AssertionError: If the origin array, if provided, does not have exactly 3 coordinates.
    """

    assert origin.size == 3, "Invalid origin dimension. It must have 3 coordinates"
    origin = origin.astype(np.float64)

    assert radius > 0.0, "Invalid radius. It must be positive"
    radius = np.float64(radius)

    if axis is None:
        cpp_obj = qugar.cpp.create_cylinder(radius, origin, use_bzr)
    else:
        cpp_obj = qugar.cpp.create_cylinder(radius, origin, axis.astype(np.float64), use_bzr)
    return ImplicitFunc(cpp_obj)


def create_annulus(
    inner_radius: float | np.float32 | np.float64,
    outer_radius: float | np.float32 | np.float64,
    center: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates a 2D annulus (a ring-shaped object) with the specified inner and outer radii,
    and optional center.

    Args:
        inner_radius (float | np.float32 | np.float64): The inner radius of the annulus.
            Must be positive.
        outer_radius (float | np.float32 | np.float64): The outer radius of the annulus.
            Must be positive.
        center (Optional[npt.NDArray[np.float32 | np.float64]], optional): The center coordinates
            of the annulus. Must be a 2-dimensional array. Defaults to None.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: An implicit function representing the created annulus.

    Raises:
        AssertionError: If the inner or outer radius is not positive.
        AssertionError: If the center array, if provided, does not have exactly 2 coordinates.
    """
    inner_radius = np.float64(inner_radius)
    assert inner_radius > 0.0, "Invalid inner radius. It must be positive"

    outer_radius = np.float64(outer_radius)
    assert outer_radius > 0.0, "Invalid outer radius. It must be positive"

    if center is None:
        cpp_obj = qugar.cpp.create_annulus(inner_radius, outer_radius, use_bzr)
    else:
        assert center.size == 2, "Invalid center dimension. It must have 2 coordinates"
        cpp_obj = qugar.cpp.create_annulus(
            inner_radius, outer_radius, center.astype(np.float64), use_bzr
        )
    return ImplicitFunc(cpp_obj)


def create_torus(
    major_radius: float | np.float32 | np.float64,
    minor_radius: float | np.float32 | np.float64,
    center: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    axis: Optional[npt.NDArray[np.float32 | np.float64]] = None,
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates a 3D torus (a ring-shaped object) with the specified major and outer radii,
    and optional center and axis

    Args:
        major_radius (float | np.float32 | np.float64): The major radius of the annulus.
            Must be positive.
        minor_radius (float | np.float32 | np.float64): The outer radius of the annulus.
            Must be positive.
        center (Optional[npt.NDArray[np.float32 | np.float64]], optional): The center coordinates
            of the annulus. Must be a 3-dimensional array. Defaults to None.
        axis (Optional[npt.NDArray[np.float32 | np.float64]], optional): The axis coordinates
            of the torus (perpendicular to the torus plane). Must be a 3-dimensional array.
            Defaults to None.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: An implicit function representing the created annulus.

    Raises:
        AssertionError: If the major or outer radius are not positive.
        AssertionError: If the center array, if provided, does not have exactly 3 coordinates.
        AssertionError: If the axis array, if provided, does not have exactly 3 coordinates.
    """
    major_radius = np.float64(major_radius)
    assert major_radius > 0.0, "Invalid major radius. It must be positive"

    minor_radius = np.float64(minor_radius)
    assert minor_radius > 0.0, "Invalid outer radius. It must be positive"

    if center is None:
        if axis is None:
            return ImplicitFunc(qugar.cpp.create_torus(major_radius, minor_radius, use_bzr))
        else:
            center = np.zeros(3, dtype=np.float64)
            return ImplicitFunc(
                qugar.cpp.create_torus(
                    major_radius, minor_radius, center, axis.astype(np.float64), use_bzr
                )
            )
    else:
        center = center.astype(np.float64)
        if axis is None:
            return ImplicitFunc(qugar.cpp.create_torus(major_radius, minor_radius, center, use_bzr))
        else:
            return ImplicitFunc(
                qugar.cpp.create_torus(
                    major_radius, minor_radius, center, axis.astype(np.float64), use_bzr
                )
            )


def create_constant(
    value: float | np.float32 | np.float64,
    dim: int,
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates a constant implicit function with the specified value and dimension.

    Args:
        value (float | np.float32 | np.float64): The constant value for the implicit function.
        dim (int): The dimension of the implicit function. Must be 2 or 3.
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: An instance of ImplicitFunc representing the constant function.

    Raises:
        AssertionError: If the dimension is not 2 or 3.
    """
    assert dim == 2 or dim == 3, "Invalid dimension. It must be 2 or 3"

    value = np.float64(value)

    if dim == 2:
        cpp_obj = qugar.cpp.create_constant_2D(value, use_bzr)
    else:
        cpp_obj = qugar.cpp.create_constant_3D(value, use_bzr)

    return ImplicitFunc(cpp_obj)


def create_plane(
    origin: Optional[npt.NDArray[np.float32 | np.float64]],
    normal: Optional[npt.NDArray[np.float32 | np.float64]],
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates an implicit plane function.

    Args:
        origin (Optional[npt.NDArray[np.float32 | np.float64]]): The origin point of the plane.
            If None, defaults to [0.0, 0.0, 0.0].
        normal (Optional[npt.NDArray[np.float32 | np.float64]]): The normal vector of the plane.
          If None, defaults to [1.0, 0.0, 0.0].
        use_bzr (bool): A flag indicating whether to use BZR. Defaults to True.

    Returns:
        ImplicitFunc: An implicit function representing the plane.

    Raises:
        AssertionError: If the origin array, if provided, does not have exactly 3 coordinates.
        AssertionError: If the normal array, if provided, does not have exactly 3 coordinates.

    """
    if origin is None:
        if normal is None:
            return ImplicitFunc(qugar.cpp.create_plane(use_bzr))
        else:
            origin = np.zeros(3, dtype=np.float64)
            assert normal.size == 3, "Invalid center dimension. It must have 3 coordinates"
            return ImplicitFunc(qugar.cpp.create_plane(origin, normal.astype(np.float64), use_bzr))
    else:
        assert origin.size == 3, "Invalid center dimension. It must have 3 coordinates"
        if normal is None:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            assert normal.size == 3, "Invalid center dimension. It must have 3 coordinates"
        return ImplicitFunc(
            qugar.cpp.create_plane(origin.astype(np.float64), normal.astype(np.float64), use_bzr)
        )


def create_line(
    origin: Optional[npt.NDArray[np.float32 | np.float64]],
    normal: Optional[npt.NDArray[np.float32 | np.float64]],
    use_bzr: bool = True,
) -> ImplicitFunc:
    """
    Creates an implicit line function.

    Args:
        origin (Optional[npt.NDArray[np.float32 | np.float64]]): The origin point of the line.
            If None, defaults to [0.0, 0.0].
        normal (Optional[npt.NDArray[np.float32 | np.float64]]): The normal vector of the line.
          If None, defaults to [1.0, 0.0].
        use_bzr (bool, optional): Flag to use Bezier representation. Defaults to True.

    Returns:
        ImplicitFunc: An implicit function representing the line.

    Raises:
        AssertionError: If the origin array, if provided, does not have exactly 2 coordinates.
        AssertionError: If the normal array, if provided, does not have exactly 2 coordinates.

    """
    if origin is None:
        if normal is None:
            return ImplicitFunc(qugar.cpp.create_line(use_bzr))
        else:
            origin = np.zeros(2, dtype=np.float64)
            assert normal.size == 2, "Invalid center dimension. It must have 2 coordinates"
            return ImplicitFunc(qugar.cpp.create_line(origin, normal.astype(np.float64), use_bzr))
    else:
        assert origin.size == 2, "Invalid center dimension. It must have 2 coordinates"
        if normal is None:
            normal = np.array([1.0, 0.0], dtype=np.float64)
        else:
            assert normal.size == 2, "Invalid center dimension. It must have 2 coordinates"
        return ImplicitFunc(
            qugar.cpp.create_line(origin.astype(np.float64), normal.astype(np.float64), use_bzr)
        )


def create_dim_linear(
    coefs: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    affine_trans: Optional[qugar.cpp.AffineTransf_2D | qugar.cpp.AffineTransf_2D] = None,
) -> ImplicitFunc:
    """
    Creates a dimensional linear implicit function.

    Args:
        coefs (list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64]):
            Coefficients for the linear function. Must be a list or numpy array of floats.
            4 coefficients in 2D and 8 coefficients in 3D.
        affine_trans (Optional[qugar.cpp.AffineTransf_2D | qugar.cpp.AffineTransf_2D]):
            Optional affine transformation to be applied. If provided, it must be an instance
            of `qugar.cpp.AffineTransf_2D` or `qugar.cpp.AffineTransf_2D`.

    Returns:
        ImplicitFunc: An instance of `ImplicitFunc` representing the created dimensional
            linear function.

    Raises:
        AssertionError: If the size of `coefs` is not 4 or 8, or if the size of `coefs` does
            not match the expected dimension based on the optionally provided affine transformation.
    """
    if isinstance(coefs, list):
        coefs = np.array(coefs)
    coefs = coefs.astype(np.float64)

    assert coefs.size == 4 or coefs.size == 8, (
        "Invalid coefficients dimension. It must have 4 (for 2D) or 8 (for 3D) coordinates"
    )

    if affine_trans is None:
        cpp_obj = qugar.cpp.create_dim_linear(coefs)
    else:
        dim = 2 if isinstance(affine_trans, qugar.cpp.AffineTransf_2D) else 3

        assert coefs.size == 2**dim, (
            "Invalid coefficients dimension. "
            f"For {dim}D cases (according to the provided affine transformation) the number "
            f"of coefficients must be {2**dim}."
        )

        cpp_obj = qugar.cpp.create_dim_linear(coefs, affine_trans)

    return ImplicitFunc(cpp_obj)


def create_box(
    affine_trans: Optional[qugar.cpp.AffineTransf_2D | qugar.cpp.AffineTransf_2D] = None,
    dim: Optional[int] = None,
) -> ImplicitFunc:
    """
    Creates a box (square in 2D or cube in 3D) with an optional affine transformation.

    A box, before transforming it affinely, is a domain [-1,1]^dim.

    Args:
        affine_trans (Optional[qugar.cpp.AffineTransf_2D | qugar.cpp.AffineTransf_2D]):
            An optional affine transformation to apply to the box. If None, a default box
            is created.
        dim (Optional[int]):
            The dimension of the box. Must be either 2 or 3. If affine_trans is provided,
            this must match the dimension of the affine transformation. If None, defaults to 2.

    Returns:
        ImplicitFunc: An implicit function representing the created box.

    Raises:
        AssertionError: If the dimension is not 2 or 3, or if the dimension does not match the
            affine transformation.
    """
    if affine_trans is None:
        if dim is None:
            dim = 2
        assert dim == 2 or dim == 3, "Invalid dimension. It must be 2 or 3"

        if dim == 2:
            cpp_obj = qugar.cpp.create_square_2D()
        else:
            cpp_obj = qugar.cpp.create_square_3D()

    else:
        assert dim == (2 if isinstance(affine_trans, qugar.cpp.AffineTransf_2D) else 3), (
            "Non-matching affine transformation and dimension"
        )

        if dim == 2:
            cpp_obj = qugar.cpp.create_square_2D(affine_trans)
        else:
            cpp_obj = qugar.cpp.create_square_3D(affine_trans)

    return ImplicitFunc(cpp_obj)


def create_negative(func: ImplicitFunc) -> ImplicitFunc:
    """
    Creates a new implicit function that represents the negative of the input function.

    Args:
        func (ImplicitFunc): The input function.

    Returns:
        ImplicitFunc: A new implicit function representing the negative of the input function.
    """
    return ImplicitFunc(qugar.cpp.create_negative(func.cpp_object))


def create_functions_addition(lhs_func: ImplicitFunc, rhs_func: ImplicitFunc) -> ImplicitFunc:
    """
    Creates a new implicit function that represents the addition of two given implicit functions.

    Args:
        lhs_func (ImplicitFunc): The left-hand side implicit function.
        rhs_func (ImplicitFunc): The right-hand side implicit function.

    Returns:
        ImplicitFunc: A new implicit function representing the addition of the two input functions.
    """
    return ImplicitFunc(
        qugar.cpp.create_functions_addition(lhs_func.cpp_object, rhs_func.cpp_object)
    )


def create_functions_subtraction(lhs_func: ImplicitFunc, rhs_func: ImplicitFunc) -> ImplicitFunc:
    """
    Creates a new implicit function that represents the subtraction of two given implicit functions.

    Args:
        lhs_func (ImplicitFunc): The left-hand side implicit function.
        rhs_func (ImplicitFunc): The right-hand side implicit function.

    Returns:
        ImplicitFunc: A new implicit function representing the subtraction of the two input
            functions.
    """
    return ImplicitFunc(
        qugar.cpp.create_functions_subtraction(lhs_func.cpp_object, rhs_func.cpp_object)
    )


def create_functions_intersection(funcs: list[ImplicitFunc]) -> ImplicitFunc:
    r"""
    Creates a new implicit function whose domain is the intersection of the negative regions
    of the given functions.

    The resulting domain is the subregion where *all* the input functions are simultaneously
    negative, i.e., :math:`\Omega = \{\mathbf{x}\,|\,\phi_i(\mathbf{x})\leq 0\ \forall i\}`.
    This maps onto Algoim's multi-polynomial quadrature (a domain defined by several Bernstein
    polynomials) and can be used, e.g., to carve several disjoint holes out of a domain by
    intersecting negated disks.

    Args:
        funcs (list[ImplicitFunc]): The functions whose negative regions are intersected.
            All of them must be Bezier polynomials (created with ``use_bzr=True``) and share
            the same dimension. At least one function is required.

    Returns:
        ImplicitFunc: A new implicit function representing the intersection domain.

    Raises:
        AssertionError: If ``funcs`` is empty or the functions do not share the same dimension.
        ValueError: If any of the functions is not a Bezier polynomial.
    """
    assert len(funcs) > 0, "At least one function is required."

    dim = funcs[0].dim
    assert all(func.dim == dim for func in funcs), "All functions must share the same dimension."

    cpp_objects = [func.cpp_object for func in funcs]
    return ImplicitFunc(qugar.cpp.create_functions_intersection(cpp_objects))


def create_affinely_transformed_functions(
    func: ImplicitFunc, affine_transf: qugar.cpp.AffineTransf_2D | qugar.cpp.AffineTransf_3D
) -> ImplicitFunc:
    """
    Creates a new implicit function that is an affine transformation of the given function.

    Note that the polynomial nature of `func` (if it is the case) will not be preserved.

    Args:
        func (ImplicitFunc): The original implicit function to be transformed.
        affine_transf (qugar.cpp.AffineTransf_2D | qugar.cpp.AffineTransf_3D):
            The affine transformation to apply.

    Returns:
        ImplicitFunc: A new implicit function that represents the affine transformation
            of the original function.
    """
    return ImplicitFunc(qugar.cpp.create_affinely_transformed(func.cpp_object, affine_transf))


def _process_tpms_periods(
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    if isinstance(periods, list):
        periods = np.array(periods)
    periods = periods.astype(np.float64)

    assert periods.size == 2 or periods.size == 3, "Invalid dimension. It must be 2 or 3"
    return periods


def _create_tpms(
    func: Callable[
        [npt.NDArray[np.float32 | np.float64], Optional[float]], ImplicitFunc_2D | ImplicitFunc_3D
    ],
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    z: Optional[float | np.float32 | np.float64] = None,
) -> ImplicitFunc:
    """Creates a TPMS implicit function object.

    This function generates a TPMS implicit function based on the provided periods
    and (optionally) z-coordinate.

    Args:
        func (Callable[[npt.NDArray[np.float32 | np.float64], Optional[float]], ImplicitFunc_2D | ImplicitFunc_3D]):
            The function to create the TPMS implicit function.
        periods (list[float  |  np.float32  |  np.float64] | npt.NDArray[np.float32  |  np.float64]):
            The periods for the TPMS function. It can be a list of floats or a NumPy array.
        z (Optional[float  |  np.float32  |  np.float64], optional):
            The constant z-coordinate for 2D cases. 2D cases are
            generated by setting the z-coordinate to a constant value
            and apply the 3D expression.
            If the periods are 3D, and a not None value is provided an
            exception is raised. Defaults to None.

    Returns:
        ImplicitFunc: An implicit function object representing the TPMS function.
    """

    periods = _process_tpms_periods(periods)

    assert periods.size == 2 or z is None, "No z-coordinate can be set in 3D case."

    return ImplicitFunc(func(periods) if z is None else func(periods, z))  # type: ignore


def create_Schoen(
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    z: Optional[float | np.float32 | np.float64] = None,
) -> ImplicitFunc:
    """
    Creates a Schoen gyroid implicit function object.

    This function generates a Schoen gyroid implicit function based on the provided periods.
    The periods can be a list of floats or a NumPy array of float32 or float64 values.
    The dimension of the periods must be either 2 or 3.

    Args:
        periods (list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64]):
            The periods for the Schoen function. It can be a list of floats or a NumPy array.
        z (Optional[float | np.float32 | np.float64], optional):
            The constant z-coordinate for 2D cases. 2D cases are
            generated by setting the z-coordinate to a constant value
            and apply the 3D expression.
            If the periods are 3D, an exception is raised.
            Defaults to None.


    Returns:
        ImplicitFunc: An implicit function object representing the Schoen function.

    Raises:
        AssertionError: If the dimension of the periods is not 2 or 3.
    """
    return _create_tpms(qugar.cpp.create_Schoen, periods, z)


def create_Schoen_IWP(
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    z: Optional[float | np.float32 | np.float64] = None,
) -> ImplicitFunc:
    """
    Creates a Schoen-IWP gyroid implicit function object.

    This function generates a Schoen-IWP gyroid implicit function based on the provided periods.
    The periods can be a list of floats or a NumPy array of float32 or float64 values.
    The dimension of the periods must be either 2 or 3.

    Args:
        periods (list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64]):
            The periods for the Schoen-IWP function. It can be a list of floats or a NumPy array.
        z (Optional[float | np.float32 | np.float64], optional):
            The constant z-coordinate for 2D cases. 2D cases are
            generated by setting the z-coordinate to a constant value
            and apply the 3D expression.
            If the periods are 3D, an exception is raised.
            Defaults to None.

    Returns:
        ImplicitFunc: An implicit function object representing the Schoen-IWP function.

    Raises:
        AssertionError: If the dimension of the periods is not 2 or 3.
    """
    return _create_tpms(qugar.cpp.create_SchoenIWP, periods, z)


def create_Schoen_FRD(
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    z: Optional[float | np.float32 | np.float64] = None,
) -> ImplicitFunc:
    """
    Creates a Schoen-FRD gyroid implicit function object.

    This function generates a Schoen-FRD gyroid implicit function based on the provided periods.
    The periods can be a list of floats or a NumPy array of float32 or float64 values.
    The dimension of the periods must be either 2 or 3.

    Args:
        periods (list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64]):
            The periods for the Schoen-FRD function. It can be a list of floats or a NumPy array.
        z (Optional[float | np.float32 | np.float64], optional):
            The constant z-coordinate for 2D cases. 2D cases are
            generated by setting the z-coordinate to a constant value
            and apply the 3D expression.
            If the periods are 3D, an exception is raised.
            Defaults to None.

    Returns:
        ImplicitFunc: An implicit function object representing the Schoen-FRD function.

    Raises:
        AssertionError: If the dimension of the periods is not 2 or 3.
    """
    return _create_tpms(qugar.cpp.create_SchoenFRD, periods, z)


def create_Fischer_Koch_S(
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    z: Optional[float | np.float32 | np.float64] = None,
) -> ImplicitFunc:
    """
    Creates a Fischer-Koch-S gyroid implicit function object.

    This function generates a Fischer-Koch-S gyroid implicit function based on the provided periods.
    The periods can be a list of floats or a NumPy array of float32 or float64 values.
    The dimension of the periods must be either 2 or 3.

    Args:
        periods (list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64]):
            The periods for the Fischer-Koch-S function. It can be a list of floats or
            a NumPy array.
        z (Optional[float | np.float32 | np.float64], optional):
            The constant z-coordinate for 2D cases. 2D cases are
            generated by setting the z-coordinate to a constant value
            and apply the 3D expression.
            If the periods are 3D, an exception is raised.
            Defaults to None.

    Returns:
        ImplicitFunc: An implicit function object representing the Fischer-Koch-S function.

    Raises:
        AssertionError: If the dimension of the periods is not 2 or 3.
    """
    return _create_tpms(qugar.cpp.create_FischerKochS, periods, z)


def create_Schwarz_Primitive(
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    z: Optional[float | np.float32 | np.float64] = None,
) -> ImplicitFunc:
    """
    Creates a Schwarz Primitive gyroid implicit function object.

    This function generates a Schwarz Primitive gyroid implicit function based on the
    provided periods. The periods can be a list of floats or a NumPy array of float32 or
    float64 values. The dimension of the periods must be either 2 or 3.

    Args:
        periods (list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64]):
            The periods for the Schwarz Primitive function. It can be a list of floats or
            a NumPy array.
        z (Optional[float | np.float32 | np.float64], optional):
            The constant z-coordinate for 2D cases. 2D cases are
            generated by setting the z-coordinate to a constant value
            and apply the 3D expression.
            If the periods are 3D, an exception is raised.
            Defaults to None.

    Returns:
        ImplicitFunc: An implicit function object representing the Schwarz Primitive function.

    Raises:
        AssertionError: If the dimension of the periods is not 2 or 3.
    """
    return _create_tpms(qugar.cpp.create_SchwarzPrimitive, periods, z)


def create_Schwarz_Diamond(
    periods: list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64],
    z: Optional[float | np.float32 | np.float64] = None,
) -> ImplicitFunc:
    """
    Creates a Schwarz Diamond gyroid implicit function object.

    This function generates a Schwarz Diamond gyroid implicit function based on the
    provided periods. The periods can be a list of floats or a NumPy array of float32 or
    float64 values. The dimension of the periods must be either 2 or 3.

    Args:
        periods (list[float | np.float32 | np.float64] | npt.NDArray[np.float32 | np.float64]):
            The periods for the Schwarz Diamond function. It can be a list of floats or
            a NumPy array.
        z (Optional[float | np.float32 | np.float64], optional):
            The constant z-coordinate for 2D cases. 2D cases are
            generated by setting the z-coordinate to a constant value
            and apply the 3D expression.
            If the periods are 3D, an exception is raised.
            Defaults to None.

    Returns:
        ImplicitFunc: An implicit function object representing the Schwarz Diamond function.

    Raises:
        AssertionError: If the dimension of the periods is not 2 or 3.
    """
    return _create_tpms(qugar.cpp.create_SchwarzDiamond, periods, z)
