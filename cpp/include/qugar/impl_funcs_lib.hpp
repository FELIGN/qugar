// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_FUNCS_LIB_HPP
#define QUGAR_IMPL_FUNCS_LIB_HPP

//! @file impl_funcs_lib.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of a few implicit functions ready to be consumed by Algoim.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/affine_transf.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_funcs_lib_macros.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <algoim/interval.hpp>

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

namespace qugar::impl {
//! @brief Forward declaration of the tensor-product Bezier class.
template<int dim, int range> class BezierTP;
}// namespace qugar::impl

//! Namespace for defining implicit function examples.
//! These function are ready to be consumed by Algoim.
namespace qugar::impl::funcs {

namespace _impl {
  //! @brief Computes the exponent of a value.
  //!
  //! @tparam T Input and output's type.
  //! @param base Base of the exponent.
  //! @param exponent Exponent of the base.
  //! @return Computed value
  // NOLINTNEXTLINE (misc-no-recursion)
  template<class T> constexpr T pow(const T base, unsigned const exponent) noexcept
  {
    return (exponent == 0) ? 1 : (base * pow(base, exponent - 1));
  }

  template<typename V> struct IsAlgoimVector;

  template<typename T, int dim> struct IsAlgoimVector<Vector<T, dim>>
  {
    using value = std::integral_constant<bool, true>;
  };

  template<int dim> struct IsAlgoimVector<Point<dim>>
  {
    using value = std::integral_constant<bool, false>;
  };

  template<typename V, int new_dim> struct NewVector;

  template<typename T, int dim, int new_dim> struct NewVector<Vector<T, dim>, new_dim>
  {
    using type = Vector<T, new_dim>;
  };

  template<int dim, int new_dim> struct NewVector<Point<dim>, new_dim>
  {
    using type = Point<new_dim>;
  };

  template<typename V, int new_dim> using NewVector_t = typename NewVector<V, new_dim>::type;


  template<typename V> struct VectorDim;

  template<typename T, int dim_> struct VectorDim<Vector<T, dim_>>
  {
    static const int dim = dim_;
  };

  template<int dim_> struct VectorDim<Point<dim_>>
  {
    static const int dim = dim_;
  };

  template<typename V> using Hessian_t = typename NewVector<V, (VectorDim<V>::dim * (VectorDim<V>::dim + 1)) / 2>::type;

  template<typename V> struct VectorType;

  template<typename T, int dim> struct VectorType<Vector<T, dim>>
  {
    using type = T;
  };

  template<int dim> struct VectorType<Point<dim>>
  {
    using type = real;
  };

  template<typename V> using VectorType_t = typename VectorType<V>::type;


}// namespace _impl


//! @brief Function containing an affine transformation.
//!
//! @tparam dim Parametric dimension.
template<int dim> class FuncWithAffineTransf : public ImplicitFunc<dim>
{
public:
  //! @brief Default constructor.
  //! Creates and stores an identity transformation.
  FuncWithAffineTransf();

  //! @brief Constructs a new class storing the given @p transf.
  //!
  //! @param transf Transformation to store.
  explicit FuncWithAffineTransf(const AffineTransf<dim> &transf);

protected:
  //! @brief Stored affine transformation.
  AffineTransf<dim> transf_;
};


//! @brief Dimension independent square function in domain [-1,1]^dim.
//!
//! @tparam dim Parametric dimension.
//!
//! @warning So far, only implemented for the 2D case.
template<int dim> class Square : public FuncWithAffineTransf<dim>
{
public:
  //! @brief Constructs a new square aligned with the Cartesian axes.
  Square() = default;

  //! @brief Constructs a new square object transformed according to @p transf.
  //!
  //! @param transf Affine transformation applied to the square. It may rotate
  //! the axes, translate the square, and/or scale it (iso or anisotropically).
  explicit Square(const AffineTransf<dim> &transf);

  declare_impl_func_virtual_interface;

  //! @brief Returns the sign of the given value.
  //!
  //! @tparam T Type of the value.
  //! @param val Value to be tested.
  //! @return +1 is @p val is positive, -1 if it is negative, 0 otherwise.
  template<typename T> static int sgn(const T &val);
};

//! @brief dim-linear function.
//!
//! @tparam dim Parametric dimension.
template<int dim> class DimLinear : public FuncWithAffineTransf<dim>
{
public:
  //! Number of coefficients.
  static const int num_coeffs = _impl::pow(2, dim);

  //! @brief Constructs a new dim-linear function from its coefficients.
  //!
  //! @param coefs Function coefficients.
  explicit DimLinear(const std::array<real, num_coeffs> &coefs);

  //! @brief Constructs a new dim-linear function from its coefficients.
  //!
  //! @param coefs Coefficients defining the function (stored in lexicographical ordering).
  //! @param transf Affine transformation applied to the dim-linear function.
  //! It may rotate the axes, translate the square, and/or scale it (iso or anisotropically).
  DimLinear(const std::array<real, num_coeffs> &coefs, const AffineTransf<dim> &transf);

  declare_impl_func_virtual_interface;

private:
  //! Coefficients of the dim-linear expression.
  std::array<real, num_coeffs> coefs_;
};


//! @brief Creates a new implicit function that is just a base function to which
//! an affine transformation is applied.
//!
//! @tparam dim Function's parametric direction.
template<int dim> class TransformedFunction : public FuncWithAffineTransf<dim>
{
public:
  //! @brief Constructs a new transformed function.
  //!
  //! @param base_func Base function to be transformed.
  //! @param transf Affine transformation to apply.
  TransformedFunction(const std::shared_ptr<const ImplicitFunc<dim>> &base_func, const AffineTransf<dim> &transf);

  declare_impl_func_virtual_interface;

private:
  //! Base function to be transformed.
  std::shared_ptr<const ImplicitFunc<dim>> base_func_;
};

//! @brief This function computes the negative of a given function.
//!
//! @tparam dim Parametric dimension.
template<int dim> class Negative : public ImplicitFunc<dim>
{
public:
  //! @brief Constructor.
  //!
  //! @param func Function whose negative is computed.
  explicit Negative(const std::shared_ptr<const ImplicitFunc<dim>> &func);

  declare_impl_func_virtual_interface;

private:
  //! Function whose negative is computed.
  std::shared_ptr<const ImplicitFunc<dim>> func_;
};

//! @brief This function adds two functions together.
//!
//! @tparam dim Parametric dimension.
template<int dim> class AddFunctions : public ImplicitFunc<dim>
{
public:
  //! @brief Constructor.
  //!
  //! @param lhs Left-hand-side operand.
  //! @param rhs Right-hand-side operand.
  explicit AddFunctions(const std::shared_ptr<const ImplicitFunc<dim>> &lhs,
    const std::shared_ptr<const ImplicitFunc<dim>> &rhs);

  declare_impl_func_virtual_interface;

private:
  //! Left-hand-side operand.
  std::shared_ptr<const ImplicitFunc<dim>> lhs_;
  //! Right-hand-side operand.
  std::shared_ptr<const ImplicitFunc<dim>> rhs_;
};

//! @brief This function subtracts two functions.
//!
//! @tparam dim Parametric dimension.
template<int dim> class SubtractFunctions : public ImplicitFunc<dim>
{
public:
  //! @brief Constructor.
  //!
  //! @param lhs Left-hand-side operand.
  //! @param rhs Right-hand-side operand.
  explicit SubtractFunctions(const std::shared_ptr<const ImplicitFunc<dim>> &lhs,
    const std::shared_ptr<const ImplicitFunc<dim>> &rhs);

  declare_impl_func_virtual_interface;

private:
  //! Left-hand-side operand.
  std::shared_ptr<const ImplicitFunc<dim>> lhs_;
  //! Right-hand-side operand.
  std::shared_ptr<const ImplicitFunc<dim>> rhs_;
};


//! @brief Domain implicitly defined by the intersection of the negative regions of
//! several Bezier polynomials.
//!
//! The represented domain is the subregion where <em>all</em> the stored Bezier
//! polynomials are simultaneously negative, i.e.,
//! @f$\Omega = \{\mathbf{x}\,|\,\phi_i(\mathbf{x})\leq 0\ \forall i\}@f$.
//!
//! As a scalar implicit function it evaluates to @f$\max_i \phi_i@f$, so that its
//! negative region coincides with the intersection above. This representation maps
//! directly onto Algoim's multi-polynomial quadrature (a domain defined by several
//! Bernstein polynomials), and it is the mechanism used, e.g., to carve several
//! disjoint holes out of a domain.
//!
//! @note Interval (Taylor-model) evaluation is not supported: cell signs are computed
//! component-wise (per polynomial) by the classification routines, which special-case
//! this type.
//!
//! @tparam dim Parametric dimension.
template<int dim> class BeziersIntersection : public ImplicitFunc<dim>
{
public:
  //! Type of the stored Bezier polynomials.
  using BezierPtr = std::shared_ptr<const BezierTP<dim, 1>>;

  //! @brief Constructor.
  //!
  //! @param beziers Bezier polynomials whose negative regions are intersected.
  //! It must contain at least one polynomial.
  explicit BeziersIntersection(const std::vector<BezierPtr> &beziers);

  declare_impl_func_virtual_interface;

public:
  //! @brief Gets the Bezier polynomials defining the domain.
  //! @return Stored Bezier polynomials.
  [[nodiscard]] const std::vector<BezierPtr> &get_beziers() const;

private:
  //! @brief Gets the polynomial attaining the maximum value at @p point (the active constraint).
  //!
  //! The gradient and Hessian of the intersection at a point coincide with those of the
  //! polynomial attaining the maximum there.
  //!
  //! @param point Point at which the active polynomial is queried.
  //! @return Reference to the active Bezier polynomial.
  [[nodiscard]] const BezierTP<dim, 1> &get_active_bezier(const Point<dim> &point) const;

  //! Bezier polynomials whose negative regions define the domain.
  std::vector<BezierPtr> beziers_;
};


}// namespace qugar::impl::funcs


#endif// QUGAR_IMPL_FUNCS_LIB_HPP