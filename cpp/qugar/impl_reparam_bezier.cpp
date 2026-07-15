// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_reparam_bezier.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of tools for creating Bezier reparameterizations.
//!
//! The implementation in this file partially derives from the Algoim library.
//! Its license is included in the Algoim_LICENSE file in the root directory of this project.
//!
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_reparam_bezier.hpp>

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_reparam_mesh.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/tensor_index_tp.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/bernstein.hpp>
#include <algoim/booluarray.hpp>
#include <algoim/polyset.hpp>
#include <algoim/quadrature_multipoly.hpp>
#include <algoim/xarray.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace qugar::impl {
namespace {

  namespace alg = algoim;

  //! @brief dim-dimensional reparameterization of an range-dimensional Bezier(s) polynomial(s)
  //! in the unit hypercube.
  //! This is just a modification of the class ImplicitPolyQuadrature (quadrature_multipoly.hpp)
  //! for generating reparameterizations.
  //!
  //! @tparam dim Parametric dimension of the current sub-domain.
  //! @tparam range Parametric dimension of the final domain.
  //! @tparam S Flag indicating if the reparameterization must
  //!         be performed only for the levelset surface (true), i.e.,
  //!         the manifold where the any of the polynomials is equal to 0,
  //!         or the subregion volume (false) between those manifolds
  //!         where all the polynomials are negative.
  template<int dim, int range, bool S> struct ImplicitPolyReparam
  {
    //! Sub parametric dimension.
    static const int sub_dim = dim - 1;
    //! Parametric dimension of the reparameterization.
    static const int reparam_dim = S ? range - 1 : range;

    //! Integral type.
    enum IntegralType : std::uint8_t { inner, outer_single, outer_aggregate };

    //! Type of the parent, either a higher-dimension class instance or void (when dim==range).
    using Parent = std::conditional_t<dim == range, void, ImplicitPolyReparam<dim + 1, range, S>>;

    //! Type of the reparameterization mesh (when dim==range) or void pointer.
    using Reparam = std::conditional_t<dim == range, ImplReparamMesh<reparam_dim, range> &, void *>;


    //! Given dim-dimensional polynomials
    alg::PolySet<dim, ALGOIM_M> phi_;
    //! Elimination axis/height direction; dir_k_=dim if there are no interfaces, dir_k_=-1 if the domain is empty.
    int dir_k_{ -1 };
    //! Reparameterization order.
    int order_{ 2 };
    //! Whether (2nd kind) Chebyshev nodes are used in the quadrature (if true), or equally spaced nodes (if false).
    bool use_Chebyshev_{ false };
    //! Base polynomials corresponding to removal of axis dir_k_
    ImplicitPolyReparam<sub_dim, range, S> base_;
    //! If quad method is auto chosen, indicates whether TS is applied
    bool auto_apply_TS_{ false };
    //! Whether an inner integral, or outer of two kinds
    IntegralType type_;
    //! Stores other base cases, besides dir_k_, when in aggregate mode.
    std::array<std::tuple<int, ImplicitPolyReparam<sub_dim, range, S>>, sub_dim> base_other_;
    //! Parent class instance (or void when dim==range).
    Parent *parent_;
    //! Reference intervals in the unit hypercube along the current height direction.
    std::unordered_map<TensorIndexTP<sub_dim>, RootsIntervals<sub_dim>> ref_intervals_;
    //! Reparameterization element (if dim < range, void).
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-const-or-ref-data-members)
    Reparam reparam_;
    //! Default tolerance to be used in certain calculations.
    Tolerance tol_;

    //! Map of reparameterization cells (cell tensor index to flat index).
    std::conditional_t<dim == range, std::unordered_map<TensorIndexTP<range>, int>, void *> cells_indices_map_;

    //! Default constructor.
    ImplicitPolyReparam() : ImplicitPolyReparam(2, nullptr) {}

    //! Constructor sets to an uninitialised state.
    explicit ImplicitPolyReparam(const int order, Reparam reparam)
      : phi_(), order_(order), use_Chebyshev_(ImplReparamMesh<range, range>::use_Chebyshev(order)),
        base_(order, nullptr), type_(IntegralType::outer_single), base_other_(), parent_(nullptr), ref_intervals_(),
        reparam_(reparam)
    {
      assert(1 < order);
      if constexpr (dim == range) {
        assert(reparam_.get_order() == order);
      }
    }

    //! @brief Assuming phi has been instantiated,
    //! determine elimination axis and build base
    //! @param outer
    //! @param auto_apply_TS
    // NOLINTNEXTLINE (readability-function-cognitive-complexity)
    void build(const bool outer, const bool auto_apply_TS)
    {
      this->type_ = outer ? outer_single : inner;
      this->auto_apply_TS_ = auto_apply_TS;

      // If phi is empty, apply a tensor-product Gaussian quadrature
      if (this->phi_.count() == 0) {
        this->dir_k_ = dim;
        this->auto_apply_TS_ = false;
        return;
      }

      if constexpr (dim == 1) {
        // If in one dimension, there is only one choice of height direction and
        // the recursive process halts
        this->dir_k_ = 0;
        return;
      } else {
        // Compute score; penalise any directions which likely contain vertical tangents
        Vector<bool, dim> has_disc;
        Vector<real, dim> score = alg::detail::score_estimate(this->phi_, has_disc);
        assert(max(abs(score)) > 0);
        score /= 2 * max(abs(score));
        for (int i = 0; i < dim; ++i) {
          if (!has_disc(i)) {
            score(i) += 1.0;
          }
        }

        // Choose height direction and form base polynomials; if tanh-sinh is being used at this
        // level, suggest the same all the way down; moreover, suggest tanh-sinh if a non-empty
        // discriminant mask has been found
        this->dir_k_ = argmax(score);
        alg::detail::eliminate_axis(this->phi_, this->dir_k_, this->base_.phi_);
        this->base_.parent_ = this;
        this->base_.order_ = this->order_;
        this->base_.build(false, this->auto_apply_TS_ || has_disc(this->dir_k_));

        // If this is the outer integral, and surface quadrature schemes are required, apply
        // the dimension-aggregated scheme when necessary
        if (outer && has_disc(this->dir_k_)) {
          this->type_ = outer_aggregate;
          for (int i = 0; i < dim; ++i) {
            if (i != this->dir_k_) {
              const auto ind = i < this->dir_k_ ? i : i - 1;
              auto &[kother, base] = at(this->base_other_, ind);
              base = ImplicitPolyReparam<sub_dim, range, S>(this->order_, nullptr);
              kother = i;
              alg::detail::eliminate_axis(this->phi_, kother, base.phi_);
              // In aggregate mode, triggered by non-empty discriminant mask,
              // base integrals always have T-S suggested
              base.parent_ = this;
              base.order_ = this->order_;
              base.build(false, true);
            }
          }
        }
      }
    }


    //! @brief Clears recursively the reference intervals and associated points.
    //! It clears the intervals for the current dimension and higher ones.
    void clear_ref_intervals()
    {
      this->ref_intervals_.clear();
      if constexpr (dim < range) {
        this->parent_->clear_ref_intervals();
      }
    }

    //! @brief Computes the roots of the polynomial @p poly_id at point @p point,
    //! along the direction k.
    //!
    //! @param point Point at which the roots are computed.
    //! @param poly_id Id of the polynomial.
    //! @param Computed roots.
    void compute_roots(const Point<sub_dim> &point, const int poly_id, std::vector<real> &roots)
    {
      roots.clear();

      const auto &pol = this->phi_.poly(static_cast<std::size_t>(poly_id));
      const auto &mask = this->phi_.mask(static_cast<std::size_t>(poly_id));
      const int order = pol.ext(this->dir_k_);

      // Ignore phi if its mask is void everywhere above the base point
      if (!alg::detail::lineIntersectsMask(mask, point, this->dir_k_)) {
        return;
      }

      // Restrict polynomial to axis-aligned line and compute its roots
      real *pline{ nullptr };
      real *roots_i_data{ nullptr };
      // NOLINTNEXTLINE (misc-const-correctness)
      alg::algoim_spark_alloc(real, &pline, order, &roots_i_data, order - 1);

      const std::span<real> roots_i(roots_i_data, static_cast<std::size_t>(order) - 1);

      alg::bernstein::collapseAlongAxis(pol, point, this->dir_k_, pline);
      const int rcount = alg::bernstein::bernsteinUnitIntervalRealRoots(pline, order, roots_i_data);

      // Add all real roots in [0,1] which are also within masked region of phi
      for (int j = 0; j < rcount; ++j) {
        const auto root_j = at(roots_i, j);
        const auto x = add_component(point, this->dir_k_, root_j);
        if (alg::detail::pointWithinMask(mask, x)) {
          roots.push_back(root_j);
        }
      }
    }

    //! @brief Checks if a point is inside the implicit domain described by polynomials.
    //! We consider that the point is inside if at that point all the polynomials are negative.
    //!
    //! @param point Point to be checked.
    //! @return True if points inside, false otherwise.
    bool check_point_inside(const Point<dim> &point)
    {
      for (std::size_t i = 0; i < this->phi_.count(); ++i) {
        const auto poly = this->phi_.poly(i);
        if (0.0 < alg::bernstein::evalBernsteinPoly(poly, point)) {
          return false;
        }
      }
      return true;
    }

    //! @brief Computes all the intervals (between consecutive roots)
    //! at point @p point along direction dir_k_.
    //!
    //! @param point Point at which intervals are computed.
    //! @param Computed intervals.
    void compute_all_intervals(const Point<sub_dim> &point, RootsIntervals<sub_dim> &intervals)
    {
      intervals.clear();
      intervals.point = point;
      // auto &roots = intervals.roots;
      // auto &func_ids = intervals.func_ids;

      intervals.add_root(0.0, -1);
      intervals.add_root(1.0, -1);

      static thread_local std::vector<real> roots;
      // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
      roots.reserve(6);
      for (int i = 0; i < static_cast<int>(this->phi_.count()); ++i) {
        this->compute_roots(point, i, roots);
        for (const auto root : roots) {
          intervals.add_root(root, i);
        }
      }

      // In rare cases, degenerate segments can be found, filter out with a tolerance
      intervals.adjust_roots(tol_, 0.0, 1.0);

      const auto n_int = intervals.get_num_roots() - 1;
      assert(0 < n_int);

      intervals.active_intervals.resize(static_cast<std::size_t>(n_int));

      for (int i = 0; i < n_int; ++i) {
        const real x_0 = at(intervals.roots, i);
        const real x_1 = at(intervals.roots, i + 1);
        at(intervals.active_intervals, i) = !tol_.equal(x_0, x_1);
      }
    }

    //! @brief Checks if the intervals with the given @p elem_tid_base,
    //! and the ones higher dimension and same base @p elem_tid_base, are inactive.
    //!
    //! @param elem_tid_base Element tensor-index of the intervals to be queried.
    //! @return True if all the intervals are inactive, false otherwise.
    [[nodiscard]] bool check_all_ref_intervals_inactive(const TensorIndexTP<sub_dim> &elem_tid_base) const
    {
      const auto &intervals = this->ref_intervals_.at(elem_tid_base);
      const auto &active_intervals = intervals.active_intervals;

      const auto n_int = static_cast<int>(active_intervals.size());
      for (int i = 0; i < n_int; ++i) {
        if (at(active_intervals, i)) {
          if constexpr (dim < range) {
            const TensorIndexTP<dim> elem_tid(add_component(elem_tid_base, this->dir_k_, i));
            if (!this->parent_->check_all_ref_intervals_inactive(elem_tid)) {
              return false;
            }
          } else {// (dim == range)
            return false;
          }
        }
      }

      return true;
    }

    //! @brief Computes recursively the reference intervals for the current
    //! dimension and the ones above.
    //!
    //! @param point_base Point at which the intervals are computed.
    //! @param elem_tid_base Element tensor index.
    void compute_ref_intervals(const Point<sub_dim> &point_base, const TensorIndexTP<sub_dim> &elem_tid_base)
    {
      if constexpr (dim == 1) {
        this->parent_->clear_ref_intervals();
      }

      static thread_local RootsIntervals<sub_dim> intervals;
      this->compute_all_intervals(point_base, intervals);

      // Loop over segments of divided interval
      const auto n_int = static_cast<int>(intervals.active_intervals.size());
      for (int i = 0; i < n_int; ++i) {
        if (!at(intervals.active_intervals, i)) {
          continue;
        }

        const real x0 = at(intervals.roots, i);
        const real x1 = at(intervals.roots, i + 1);

        const auto x = add_component(point_base, this->dir_k_, numbers::half * (x0 + x1));
        const TensorIndexTP<dim> elem_tid(add_component(elem_tid_base, this->dir_k_, i));

        if constexpr (dim < range) {
          this->parent_->compute_ref_intervals(x, elem_tid);
          at(intervals.active_intervals, i) = !this->parent_->check_all_ref_intervals_inactive(elem_tid);
        } else// (dim == range)
        {
          at(intervals.active_intervals, i) = this->check_point_inside(x);
        }
      }// i

      if constexpr (dim == range && S) {
        std::vector<bool> new_active_intervals(static_cast<std::size_t>(n_int), false);
        for (int i = 0; i < (n_int - 1); ++i) {
          at(new_active_intervals, i) = at(intervals.active_intervals, i) != at(intervals.active_intervals, i + 1);
        }
        intervals.active_intervals = new_active_intervals;
      }

      this->ref_intervals_.emplace(elem_tid_base, intervals);
    }

    //! @brief Computes all the intervals (between consecutive roots)
    //! at point @p point along direction dir_k_, taking as reference the provided
    //! @p ref_intervals.
    //!
    //! This method allows to deal with degenerate intervals by comparing
    //! the computed ones with the ones obtained at a point without degeneracies.
    //!
    //! @param ref_intervals Reference intervals.
    //! @param point Point at which new intervals are computed.
    //! @param Computed intervals.
    //!
    //! @warning This method is not bulletproof, and it may fail in corner cases.
    // NOLINTNEXTLINE (readability-function-cognitive-complexity)
    void compute_similar_intervals(const RootsIntervals<sub_dim> &ref_intervals,
      const Point<sub_dim> &point,
      RootsIntervals<sub_dim> &intervals)
    {
      static_assert(1 < dim, "Invalid dimension.");

      intervals.clear();
      intervals.point = point;

      const auto n_roots = ref_intervals.get_num_roots();
      if (n_roots == 2) {
        intervals = ref_intervals;
        return;
      }

      const real x0{ 0.0 };
      const real x1{ 1.0 };

      static thread_local std::vector<real> new_roots_i;
      // NOLINTNEXTLINE (cppcoreguidelines-avoid-magic-numbers)
      new_roots_i.reserve(6);

      for (int i = 0; i < static_cast<int>(this->phi_.count()); ++i) {
        std::vector<real> roots_i;

        for (int j = 0; j < n_roots; ++j) {
          if (i == at(ref_intervals.func_ids, j)) {
            roots_i.push_back(at(ref_intervals.roots, j));
          }
        }

        if (roots_i.empty()) {
          continue;
        }

        const auto n_i = roots_i.size();

        this->compute_roots(point, i, new_roots_i);

        // Filtering out roots near x0 and x1.
        const auto it = std::remove_if(new_roots_i.begin(), new_roots_i.end(), [x0, x1, this](const auto &root) {
          return tol_.equal(root, x0) || tol_.equal(root, x1);
        });
        new_roots_i.erase(it, new_roots_i.end());
        const auto n_i_new = new_roots_i.size();

        if (n_i_new < n_i) {
          // We have to add roots in x0 and/or x1.
          const auto &poly = this->phi_.poly(static_cast<std::size_t>(i));
          const bool root_0 = tol_.equal(
            std::abs(alg::bernstein::evalBernsteinPoly(poly, add_component(point, this->dir_k_, x0))), numbers::zero);
          const bool root_1 = tol_.equal(
            std::abs(alg::bernstein::evalBernsteinPoly(poly, add_component(point, this->dir_k_, x1))), numbers::zero);

          if (root_0 ^ root_1) {
            new_roots_i.insert(new_roots_i.end(), n_i - n_i_new, root_0 ? x0 : x1);
          } else if (root_0 && root_1) {
            if ((n_i_new + 2) == n_i) {
              new_roots_i.push_back(x0);
              new_roots_i.push_back(x1);
            } else if ((n_i_new + 1) == n_i) {
              // We decide if inserting x0 and x1 based on the function signs.
              const auto ref_intervals_pt = ref_intervals.point;
              auto x = add_component(ref_intervals_pt, this->dir_k_, numbers::half * (x0 + roots_i.front()));
              const auto sign = alg::bernstein::evalBernsteinPoly(poly, x) > 0;

              std::ranges::sort(new_roots_i);
              const real xmid = new_roots_i.empty() ? x1 : new_roots_i.front();
              x = add_component(point, this->dir_k_, numbers::half * (x0 + xmid));
              const auto new_sign = alg::bernstein::evalBernsteinPoly(poly, x) > 0;

              new_roots_i.push_back(sign == new_sign ? x1 : x0);
            }
          }
        }

        if (new_roots_i.size() != n_i)// First backup strategy.
        {
          constexpr int n_pts = 6;
          const real t0 = std::log(numbers::near_eps);
          const real t1 = std::log(0.001);
          const real dt = (t1 - t0) / real(n_pts - 1);

          std::vector<real> ts;
          ts.reserve(n_pts + 3);
          for (int j = 0; j < n_pts; ++j) {
            ts.push_back(std::exp(t0 + (j * dt)));
          }
          // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
          ts.push_back(0.005);
          ts.push_back(0.01);
          ts.push_back(0.05);
          ts.push_back(0.1);
          // NOLINTEND (cppcoreguidelines-avoid-magic-numbers)

          const Point<sub_dim> &old_point = ref_intervals.point;
          Point<sub_dim> new_point = point;
          for (const auto &val : ts) {

            const int dir = std::max(this->base_.dir_k_, dim - 2);
            new_point(dir) = (numbers::one - val) * point(dir) + val * old_point(dir);

            this->compute_roots(new_point, i, new_roots_i);
            if (new_roots_i.size() == n_i) {
              break;
            }
          }

          if (new_roots_i.size() != n_i) {
            // Instead of interpolating along one direction, we interpolate along all directions.
            for (const auto &val : ts) {
              new_point = ((numbers::one - val) * point) + (val * old_point);

              this->compute_roots(new_point, i, new_roots_i);
              if (new_roots_i.size() == n_i) {
                break;
              }
            }
          }
        }

        if (new_roots_i.size() != n_i) {// Last backup strategy.
          new_roots_i = roots_i;
        }

        for (const auto root : new_roots_i) {
          intervals.add_root(root, i);
        }

      }// i

      intervals.add_root(x0, -1);
      intervals.add_root(x1, -1);
      intervals.active_intervals = ref_intervals.active_intervals;
      assert(intervals.get_num_roots() == ref_intervals.get_num_roots());

      intervals.adjust_roots(tol_, x0, x1);
    }

    void insert_point_reparam(const Point<range> &perm_point,
      const TensorIndexTP<range> &cell_tid,
      const TensorIndexTP<reparam_dim> &pt_tid)
    {
      int cell_id{ 0 };

      const auto it = this->cells_indices_map_.find(cell_tid);
      if (it != this->cells_indices_map_.cend()) {
        cell_id = it->second;
      } else {
        cell_id = this->reparam_.allocate_cells(1);
        this->cells_indices_map_.emplace(cell_tid, cell_id);
      }

      const auto pt_id = static_cast<int>(pt_tid.flat(TensorSizeTP<reparam_dim>(this->order_)));

      // Permutation needed due to different ordering in QUGaR and Algoim.
      this->reparam_.insert_cell_point(permute_vector_directions(perm_point), cell_id, pt_id);
    }

    //! @brief Computes all the intervals (between consecutive roots)
    //! at @p point along direction e0.
    //! In the 1D, it just computes the intervals without any reference.
    //! For higher dimensions, the stored reference intervals are used.
    //!
    //! @param point Point at which new intervals are computed.
    //! @param elem_tid Element tensor index (for selecting the reference intervals).
    //! @param Computed intervals.
    void compute_intervals(const Point<sub_dim> &point,
      const TensorIndexTP<sub_dim> &elem_tid,
      RootsIntervals<sub_dim> &intervals)
    {
      const auto &ref_int = this->ref_intervals_.at(elem_tid);
      if constexpr (dim == 1) {
        intervals = ref_int;
      } else {// if (1 < dim)
        compute_similar_intervals(ref_int, point, intervals);
      }
    }

    //! @brief Generates the reparameterization point by point in a dimensional
    //! recursive way.
    //!
    //! @param point_base Point of lower dimension.
    //! @param elem_tid_base Lower dimension element tensor index.
    //! @param pt_tid_base Lower dimension point tensor index.
    void process(const Point<sub_dim> &point_base,
      // NOLINTNEXTLINE (bugprone-easily-swappable-parameters)
      const TensorIndexTP<sub_dim> &elem_tid_base,
      const TensorIndexTP<sub_dim> &pt_tid_base)
    {
      static thread_local RootsIntervals<sub_dim> intervals;
      compute_intervals(point_base, elem_tid_base, intervals);

      // Loop over segments of divided interval
      const auto n_int = static_cast<int>(intervals.active_intervals.size());
      for (int i = 0; i < n_int; ++i) {
        if (!at(intervals.active_intervals, i)) {
          continue;
        }

        const real x0 = at(intervals.roots, i);
        const real x1 = at(intervals.roots, i + 1);

        const TensorIndexTP<dim> cell_tid{ add_component(elem_tid_base, dir_k_, i) };

        if constexpr (dim == range && S) {
          const auto point = add_component(point_base, dir_k_, x1);
          this->insert_point_reparam(point, cell_tid, pt_tid_base);
        } else {
          for (int j = 0; j < this->order_; ++j) {
            const auto coord = x0 + ((x1 - x0) * this->generate_point_in_01(j));
            const auto point = add_component(point_base, dir_k_, coord);

            const TensorIndexTP<dim> pt_tid{ add_component(pt_tid_base, dir_k_, j) };
            if constexpr (dim == range) {
              this->insert_point_reparam(point, cell_tid, pt_tid);
            } else {
              this->parent_->process(point, cell_tid, pt_tid);
            }
          }
        }
      }// i
    }

    //! @brief Generates a point in the 1D range [0, 1] based on the given index.
    //!
    //! This function generates a point in the range [0, 1] for a given index `ind`.
    //! The method of generation depends on whether Chebyshev nodes are being used.
    //!
    //! if use_Chebyshev_ is true, the point is generated using the modified Chebyshev nodes.
    //! Otherwise, the point is generated using equally spaced nodes.
    //!
    //! @param ind The index for which to generate the point. Must be in the range [0, order_).
    //! @return A real number representing the generated point in the range [0, 1].
    //! @pre The index `ind` must be in the range [0, order_).
    real generate_point_in_01(const int ind)
    {
      assert(0 <= ind && ind < this->order_);
      if (this->use_Chebyshev_) {
        return alg::bernstein::modifiedChebyshevNode(ind, this->order_);
      } else {
        return real(ind) / real(this->order_ - 1);
      }
    }

    //! @brief Generates the reparameterization for a (sub-dimensional) domain
    //! that is not intersected by the polynomials.
    void reparam_tensor_product()
    {
      if constexpr (dim == range) {
        if constexpr (!S) {
          constexpr bool wirebasket = false;
          const BoundBox<range> domain_01;
          this->reparam_.add_full_cell(domain_01, wirebasket);
        }
      } else {
        this->parent_->compute_ref_intervals(Point<dim>(numbers::half), TensorIndexTP<dim>(0));

        const TensorIndexTP<dim> cell_tid(0);
        Point<dim> x;

        for (const auto &pt_tid : TensorIndexRangeTP<dim>(this->order_)) {
          for (int dir = 0; dir < dim; ++dir) {
            x(dir) = this->generate_point_in_01(pt_tid(dir));
          }

          this->parent_->process(x, cell_tid, pt_tid);
        }
      }
    }

    //! @brief Triggers the reparameterization of the implicit domain
    //! at the current dimension.
    void reparam()
    {
      if (this->dir_k_ == dim) {
        this->reparam_tensor_product();
      } else if (0 <= this->dir_k_ && this->dir_k_ < dim) {
        if constexpr (dim == 1) {
          const Point<0> x;
          const TensorIndexTP<0> pt_id;
          const TensorIndexTP<0> elem_id;
          this->compute_ref_intervals(x, pt_id);
          this->process(x, elem_id, pt_id);
        } else {// if constexpr (1 < dim)
          // Recursive call until reaching dim == 1 or tensor product case.
          this->base_.reparam();
        }
      }
    }

    //! @brief Orients the cells of the reparameterization positively.
    //!
    //! Positive orientation is such that the determinant of the Jacobian
    //! is positive for codimension 0 cells, and the normal is oriented
    //! according to the one of the polynomial levelset for codimension 1 cells.
    //!
    //! @param polys Vector of polynomials defining the domain.
    //! @param cell_ids List of cells to orient.
    void orient_cells(const std::vector<std::reference_wrapper<const ImplicitFunc<dim>>> &polys,
      const std::vector<int> &cell_ids)
    {
      assert(!polys.empty());

      if constexpr (S) {
        this->reparam_.orient_levelset_cells_positively(polys, cell_ids);
      } else {
        this->reparam_.orient_cells_positively(cell_ids);
      }
    }


    //! @brief Given a list of polynomials, generates a class instance wrapped.
    //!
    //! @param polys Vector of polynomials to be evaluated.
    //! @param order Order of the reparameterization (number of points
    //!        per direction in each reparameterization cell).
    //! @return Class instance wrapped in a shared pointer.
    static std::shared_ptr<ImplicitPolyReparam<range, range, S>> create(
      const std::vector<std::reference_wrapper<const BezierTP<range, 1>>> &polys,
      ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam)
    {
      assert(!polys.empty());

      const auto order = reparam.get_order();
      auto quad = std::make_shared<ImplicitPolyReparam<range, range, S>>(order, reparam);

      for (const auto &poly : polys) {
        const ::algoim::xarray<real, range> &xarray = poly.get().get_xarray();
        const auto mask = alg::detail::nonzeroMask(xarray, alg::booluarray<range, ALGOIM_M>(true));
        if (alg::detail::maskEmpty(mask)) {
          if (0.0 < alg::bernstein::evalBernsteinPoly<range>(xarray, Point<range>(numbers::half))) {
            quad = std::make_shared<ImplicitPolyReparam<range, range, S>>(order, reparam);// Inactive domain.
            break;
          }
        } else {
          quad->phi_.push_back(xarray, mask);
        }
      }
      quad->build(true, false);
      quad->reparam();

      return quad;
    }


    //! @brief Reparameterizes the domain defined implicitly by a list of polynomials.
    //! The interior of the domain is the subregion where all the polynomials are
    //! negative at the same time.
    //!
    //! @param polys Vector of Bezier polynomials definining the domain.
    //! @param Reparamterization to be filled.
    static void reparameterize(const std::vector<std::reference_wrapper<const BezierTP<range, 1>>> &polys,
      const BoundBox<dim> &domain,
      ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam)
    {
      static const int param_dim = S ? dim - 1 : dim;
      const auto n_cells_est = param_dim == 2 ? 3 : 13;// Estimation
      reparam.reserve_cells(n_cells_est);

      const auto n_cells_0 = static_cast<int>(reparam.get_num_cells());
      const auto n_pts_0 = static_cast<int>(reparam.get_num_points());
      const auto manager = create(polys, reparam);
      const auto n_cells_1 = static_cast<int>(reparam.get_num_cells());
      const auto n_pts_1 = static_cast<int>(reparam.get_num_points());

      const auto rng_cells = std::ranges::iota_view<int, int>{ n_cells_0, n_cells_1 };
      const std::vector<int> cell_ids(rng_cells.begin(), rng_cells.end());

      const auto rng_pts = std::ranges::iota_view<int, int>{ n_pts_0, n_pts_1 };
      const std::vector<int> point_ids(rng_pts.begin(), rng_pts.end());

      const std::vector<std::reference_wrapper<const ImplicitFunc<range>>> funcs(polys.cbegin(), polys.cend());
      manager->orient_cells(funcs, cell_ids);

      const Tolerance tol(1.0e4 * numbers::eps);
      const BoundBox<dim> domain_01;

      const std::vector<std::reference_wrapper<const ImplicitFunc<dim>>> funcs_vec{ polys.cbegin(), polys.cend() };
      reparam.generate_wirebasket(funcs_vec, cell_ids, domain_01, tol);

      reparam.scale_points(point_ids, domain_01, domain);
    }
  };

  template<int range, bool S> struct ImplicitPolyReparam<0, range, S>
  {
    explicit ImplicitPolyReparam(const int /*order*/, void * /*reparam*/) {}
  };


}// namespace


template<int dim, bool S>
std::shared_ptr<ImplReparamMesh<S ? dim - 1 : dim, dim>>
  reparam_Bezier(const BezierTP<dim, 1> &bzr, const BoundBox<dim> &domain, const int order)
{
  const auto reparam = std::make_shared<ImplReparamMesh<S ? dim - 1 : dim, dim>>(order);
  reparam_Bezier<dim, S>(bzr, domain, *reparam);
  return reparam;
}

template<int dim, bool S>
void reparam_Bezier(const BezierTP<dim, 1> &bzr,
  const BoundBox<dim> &domain,
  ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam)
{
  const std::vector<std::reference_wrapper<const BezierTP<dim, 1>>> polys = { bzr };
  ImplicitPolyReparam<dim, dim, S>::reparameterize(polys, domain, reparam);
}

template<int dim, bool S>
void reparam_Beziers(const std::vector<std::shared_ptr<const BezierTP<dim, 1>>> &bzrs,
  const BoundBox<dim> &domain,
  ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam)
{
  assert(!bzrs.empty());

  std::vector<std::reference_wrapper<const BezierTP<dim, 1>>> polys;
  polys.reserve(bzrs.size());
  for (const auto &bzr : bzrs) {
    polys.emplace_back(*bzr);
  }
  ImplicitPolyReparam<dim, dim, S>::reparameterize(polys, domain, reparam);
}


// Instantations

template std::shared_ptr<ImplReparamMesh<2, 2>>
  reparam_Bezier<2, false>(const BezierTP<2, 1> &, const BoundBox<2> &, const int);
template std::shared_ptr<ImplReparamMesh<1, 2>>
  reparam_Bezier<2, true>(const BezierTP<2, 1> &, const BoundBox<2> &, const int);
template std::shared_ptr<ImplReparamMesh<3, 3>>
  reparam_Bezier<3, false>(const BezierTP<3, 1> &, const BoundBox<3> &, const int);
template std::shared_ptr<ImplReparamMesh<2, 3>>
  reparam_Bezier<3, true>(const BezierTP<3, 1> &, const BoundBox<3> &, const int);

template void reparam_Bezier<2, false>(const BezierTP<2, 1> &, const BoundBox<2> &, ImplReparamMesh<2, 2> &);
template void reparam_Bezier<2, true>(const BezierTP<2, 1> &, const BoundBox<2> &, ImplReparamMesh<1, 2> &);
template void reparam_Bezier<3, false>(const BezierTP<3, 1> &, const BoundBox<3> &, ImplReparamMesh<3, 3> &);
template void reparam_Bezier<3, true>(const BezierTP<3, 1> &, const BoundBox<3> &, ImplReparamMesh<2, 3> &);

template void reparam_Beziers<2, false>(const std::vector<std::shared_ptr<const BezierTP<2, 1>>> &,
  const BoundBox<2> &,
  ImplReparamMesh<2, 2> &);
template void reparam_Beziers<2, true>(const std::vector<std::shared_ptr<const BezierTP<2, 1>>> &,
  const BoundBox<2> &,
  ImplReparamMesh<1, 2> &);
template void reparam_Beziers<3, false>(const std::vector<std::shared_ptr<const BezierTP<3, 1>>> &,
  const BoundBox<3> &,
  ImplReparamMesh<3, 3> &);
template void reparam_Beziers<3, true>(const std::vector<std::shared_ptr<const BezierTP<3, 1>>> &,
  const BoundBox<3> &,
  ImplReparamMesh<2, 3> &);

}// namespace qugar::impl