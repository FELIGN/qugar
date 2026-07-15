// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_funcs_lib.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of a few implicit functions ready to be consumed by Algoim.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_funcs_lib.hpp>

#include <qugar/affine_transf.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_funcs_lib_macros.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/types.hpp>
#include <qugar/vector.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace qugar::impl::funcs {


template<int dim> FuncWithAffineTransf<dim>::FuncWithAffineTransf() : FuncWithAffineTransf(AffineTransf<dim>()) {}

template<int dim>
FuncWithAffineTransf<dim>::FuncWithAffineTransf(const AffineTransf<dim> &transf) : transf_(transf.inverse())
{}


template<int dim> Square<dim>::Square(const AffineTransf<dim> &transf) : FuncWithAffineTransf<dim>(transf) {}

implement_impl_func_transf(Square);

template<int dim> template<typename T> T Square<dim>::eval_(const Point<dim, T> &point) const
{
  Vector<real, dim> diff;

  for (int dir = 0; dir < dim; ++dir) {
    if constexpr (std::is_same_v<T, real>) {
      diff(dir) = std::abs(point(dir)) - numbers::one;
    } else {
      diff(dir) = std::abs(T(point(dir)).alpha) - numbers::one;
    }
  }

  if constexpr (std::is_same_v<T, real>) {
    return max(diff);
  } else {
    const int ind = argmax(diff);
    return point(ind) * sgn(point(ind)) - numbers::one;
  }
}

template<int dim> template<typename T> Vector<T, dim> Square<dim>::grad_(const Point<dim, T> &point) const
{
  Vector<real, dim> diff{};
  for (int dir = 0; dir < dim; ++dir) {
    if constexpr (std::is_same_v<T, real>) {
      diff(dir) = std::abs(point(dir)) - numbers::one;
    } else {

      diff(dir) = std::abs(T(point(dir)).alpha) - numbers::one;
    }
  }

  const int ind = argmax(diff);
  return set_component<T, dim>(numbers::zero, ind, T(sgn(point(ind))));
}

template<int dim> template<typename T> auto Square<dim>::hessian_(const Vector<T, dim> & /*point*/) const -> Hessian<T>
{
  // Not implemented.
  assert(false);
  return Hessian<T>{};
}

template<int dim> template<typename T> int Square<dim>::sgn(const T &val)
{
  if constexpr (std::is_arithmetic_v<T>) {
    return (T(numbers::zero) < val) - (val < T(numbers::zero));
  } else {
    return val.sign();
  }
}

template<int dim>
DimLinear<dim>::DimLinear(const std::array<real, num_coeffs> &coefs) : DimLinear<dim>(coefs, AffineTransf<dim>())
{}

template<int dim>
DimLinear<dim>::DimLinear(const std::array<real, num_coeffs> &coefs, const AffineTransf<dim> &transf)
  : FuncWithAffineTransf<dim>(transf), coefs_(coefs)
{}

implement_impl_func_transf(DimLinear);

template<int dim> template<typename T> T DimLinear<dim>::eval_(const Point<dim, T> &point) const
{
  const auto &cfs = this->coefs_;
  const auto one = numbers::one;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  if constexpr (dim == 2) {
    return (cfs[0] * (one - point(0)) + cfs[1] * point(0)) * (one - point(1))
           + (cfs[2] * (one - point(0)) + cfs[3] * point(0)) * point(1);
  } else {// if constexpr (dim == 3)
    return ((cfs[0] * (one - point(0)) + cfs[1] * point(0)) * (one - point(1))
             + (cfs[2] * (one - point(0)) + cfs[3] * point(0)) * point(1))
             * (one - point(2))
           + ((cfs[4] * (one - point(0)) + cfs[5] * point(0)) * (one - point(1))
               + (cfs[6] * (one - point(0)) + cfs[7] * point(0)) * point(1))
               * point(2);
  }
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}


template<int dim> template<typename T> Vector<T, dim> DimLinear<dim>::grad_(const Point<dim, T> &point) const
{
  const auto &cfs = this->coefs_;
  const auto one = numbers::one;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  if constexpr (dim == 2) {
    return Vector<T, dim>{
      (cfs[1] - cfs[0]) * (one - point(1)) + (cfs[3] - cfs[2]) * point(1),
      (cfs[2] - cfs[0]) * (one - point(0)) + (cfs[3] - cfs[1]) * point(0),
    };
  } else {// if constexpr (dim == 3)
    return Vector<T, dim>{
      ((cfs[1] - cfs[0]) * (one - point(1)) + (cfs[3] - cfs[2]) * point(1)) * (one - point(2))
        + ((cfs[5] - cfs[4]) * (one - point(1)) + (cfs[7] - cfs[6]) * point(1)) * point(2),

      ((cfs[2] - cfs[0]) * (one - point(0)) + (cfs[3] - cfs[1]) * point(0)) * (one - point(2))
        + ((cfs[6] - cfs[4]) * (one - point(0)) + (cfs[7] - cfs[5]) * point(0)) * point(2),

      ((cfs[4] - cfs[0]) * (one - point(0)) + (cfs[5] - cfs[1]) * point(0)) * (one - point(1))
        + ((cfs[6] - cfs[2]) * (one - point(0)) + (cfs[7] - cfs[3]) * point(0)) * point(1),
    };
  }
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}

// NOLINTNEXTLINE (readability-function-cognitive-complexity)
template<int dim> template<typename T> auto DimLinear<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  const auto &cfs = this->coefs_;
  const auto zero = numbers::zero;
  const auto one = numbers::one;

  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  if constexpr (dim == 2) {
    return Hessian<T>{ zero, cfs[0] + cfs[3] - cfs[1] - cfs[2], zero };
  } else {
    return Hessian<T>{ zero,
      (cfs[0] + cfs[3] - cfs[1] - cfs[2]) * (one - point(2)) + (cfs[4] + cfs[7] - cfs[5] - cfs[6]) * point(2),
      (cfs[0] + cfs[5] - cfs[1] - cfs[4]) * (one - point(1)) + (cfs[2] + cfs[7] - cfs[3] - cfs[6]) * point(1),
      zero,
      ((cfs[0] + cfs[7] - cfs[2] - cfs[5]) * (one - point(0)) + (cfs[1] + cfs[6] - cfs[3] - cfs[4]) * point(0)),
      zero };
  }
  // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)
}


template<int dim>
TransformedFunction<dim>::TransformedFunction(const std::shared_ptr<const ImplicitFunc<dim>> &base_func,
  const AffineTransf<dim> &transf)
  : FuncWithAffineTransf<dim>(transf), base_func_(base_func)
{
  assert(this->base_func_ != nullptr);
}

implement_impl_func_transf(TransformedFunction);

template<int dim> template<typename T> T TransformedFunction<dim>::eval_(const Point<dim, T> &point) const
{
  return this->base_func_->operator()(point);
}

template<int dim> template<typename T> Vector<T, dim> TransformedFunction<dim>::grad_(const Point<dim, T> &point) const
{
  return this->base_func_->grad(point);
}

template<int dim>
template<typename T>
auto TransformedFunction<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  return this->base_func_->hessian(point);
}

template<int dim> Negative<dim>::Negative(const std::shared_ptr<const ImplicitFunc<dim>> &func) : func_(func)
{
  assert(func_ != nullptr);
}

implement_impl_func(Negative);

template<int dim> template<typename T> T Negative<dim>::eval_(const Point<dim, T> &point) const
{
  return -func_->operator()(point);
}

template<int dim> template<typename T> Vector<T, dim> Negative<dim>::grad_(const Point<dim, T> &point) const
{
  return -func_->grad(point);
}

template<int dim> template<typename T> auto Negative<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  return -func_->hessian(point);
}

template<int dim>
AddFunctions<dim>::AddFunctions(const std::shared_ptr<const ImplicitFunc<dim>> &lhs,
  const std::shared_ptr<const ImplicitFunc<dim>> &rhs)
  : lhs_(lhs), rhs_(rhs)
{
  assert(this->lhs_ != nullptr);
  assert(this->rhs_ != nullptr);
}

implement_impl_func(AddFunctions);

template<int dim> template<typename T> T AddFunctions<dim>::eval_(const Point<dim, T> &point) const
{
  return this->lhs_->operator()(point) + this->rhs_->operator()(point);
}

template<int dim> template<typename T> Vector<T, dim> AddFunctions<dim>::grad_(const Point<dim, T> &point) const
{
  return this->lhs_->grad(point) + this->rhs_->grad(point);
}

template<int dim> template<typename T> auto AddFunctions<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  return this->lhs_->hessian(point) + this->rhs_->hessian(point);
}

template<int dim>
SubtractFunctions<dim>::SubtractFunctions(const std::shared_ptr<const ImplicitFunc<dim>> &lhs,
  const std::shared_ptr<const ImplicitFunc<dim>> &rhs)
  : lhs_(lhs), rhs_(rhs)
{
  assert(this->lhs_ != nullptr);
  assert(this->rhs_ != nullptr);
}

implement_impl_func(SubtractFunctions);

template<int dim> template<typename T> T SubtractFunctions<dim>::eval_(const Point<dim, T> &point) const
{
  return this->lhs_->operator()(point) - this->rhs_->operator()(point);
}

template<int dim> template<typename T> Vector<T, dim> SubtractFunctions<dim>::grad_(const Point<dim, T> &point) const
{
  return this->lhs_->grad(point) - this->rhs_->grad(point);
}

template<int dim>
template<typename T>
auto SubtractFunctions<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  return this->lhs_->hessian(point) - this->rhs_->hessian(point);
}

template<int dim>
BeziersIntersection<dim>::BeziersIntersection(const std::vector<BezierPtr> &beziers) : beziers_(beziers)
{
  assert(!beziers_.empty());
  assert(std::ranges::none_of(beziers_, [](const BezierPtr &bzr) { return bzr == nullptr; }));
}

implement_impl_func(BeziersIntersection);

template<int dim>
auto BeziersIntersection<dim>::get_active_bezier(const Point<dim> &point) const -> const BezierTP<dim, 1> &
{
  auto best = beziers_.cbegin();
  real best_val = (**best)(point);
  for (auto it = std::next(beziers_.cbegin()); it != beziers_.cend(); ++it) {
    const real val = (**it)(point);
    if (val > best_val) {
      best_val = val;
      best = it;
    }
  }
  return **best;
}

template<int dim> template<typename T> T BeziersIntersection<dim>::eval_(const Point<dim, T> &point) const
{
  if constexpr (std::is_same_v<T, real>) {
    real value = (*beziers_.front())(point);
    for (auto it = std::next(beziers_.cbegin()); it != beziers_.cend(); ++it) {
      value = std::max(value, (**it)(point));
    }
    return value;
  } else {
    throw std::logic_error("Interval evaluation is not supported for BeziersIntersection.");
  }
}

template<int dim> template<typename T> Vector<T, dim> BeziersIntersection<dim>::grad_(const Point<dim, T> &point) const
{
  if constexpr (std::is_same_v<T, real>) {
    return this->get_active_bezier(point).grad(point);
  } else {
    throw std::logic_error("Interval gradient is not supported for BeziersIntersection.");
  }
}

template<int dim>
template<typename T>
auto BeziersIntersection<dim>::hessian_(const Point<dim, T> &point) const -> Hessian<T>
{
  return this->get_active_bezier(point).hessian(point);
}

template<int dim> auto BeziersIntersection<dim>::get_beziers() const -> const std::vector<BezierPtr> &
{
  return this->beziers_;
}

// Instantiations

template class Square<2>;
template class Square<3>;

template class DimLinear<2>;
template class DimLinear<3>;

template class TransformedFunction<2>;
template class TransformedFunction<3>;

template class Negative<2>;
template class Negative<3>;

template class AddFunctions<2>;
template class AddFunctions<3>;

template class SubtractFunctions<2>;
template class SubtractFunctions<3>;

template class BeziersIntersection<2>;
template class BeziersIntersection<3>;

template class FuncWithAffineTransf<2>;
template class FuncWithAffineTransf<3>;


}// namespace qugar::impl::funcs
