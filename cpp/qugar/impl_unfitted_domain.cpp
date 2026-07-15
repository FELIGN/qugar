// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_unfitted_domain.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of UnfittedImplDomain class.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_unfitted_domain.hpp>

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/cart_grid_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_funcs_lib.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/unfitted_domain.hpp>
#include <qugar/unfitted_domain_binary_part.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/booluarray.hpp>
#include <algoim/hyperrectangle.hpp>
#include <algoim/interval.hpp>
#include <algoim/polyset.hpp>
#include <algoim/quadrature_general.hpp>
#include <algoim/quadrature_multipoly.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qugar::impl {


namespace alg = ::algoim;

namespace {

  enum class ImmersedCellStatusTmp : std::uint8_t {
    cut,
    full,
    empty,
    full_with_unf_bdry,
  };


  //! @brief Checks if the subgrid intersects with the target cells.
  //!
  //! This function determines whether any of the target cells are within the range
  //! of the given subgrid by checking if at least one cell ID from the target cells
  //! exists within the subgrid's range.
  //!
  //! @tparam dim The dimensionality of the grid.
  //! @param subgrid The subgrid to check for intersection with target cells.
  //! @param target_cells A vector of cell indices to test for intersection.
  //! @return `true` if any target cell is within the subgrid's range; `false` otherwise.
  template<int dim> bool intersect(const SubCartGridTP<dim> &subgrid, const std::vector<std::int64_t> &target_cells)
  {
    const auto &range = subgrid.get_range();
    const auto &size = subgrid.get_grid()->get_num_cells_dir();
    return std::any_of(target_cells.cbegin(), target_cells.cend(), [&range, &size](const std::int64_t cell_id) {
      return range.is_in_range(cell_id, size);
    });
  }

  //! @brief Checks if the subgrid intersects with the target cells.
  //!
  //! This function determines whether any of the target cells are within the range
  //! of the given subgrid by checking if at least one cell ID from the target cells
  //! exists within the subgrid's range.
  //!
  //! @p target_cells is an optional vector of cell indices to test for intersection.
  //! If `target_cells` is not provided, this function returns `true` (it assumes that
  //! all the cells are considered.)
  //!
  //! @tparam dim The dimensionality of the grid.
  //! @param subgrid The subgrid to check for intersection with target cells.
  //! @param target_cells An optional vector of cell indices to test for intersection.
  //! @return `true` if any target cell is within the subgrid's range, or no target cells are defined;
  //! `false` otherwise.
  template<int dim>
  bool intersect(const SubCartGridTP<dim> &subgrid, const std::optional<std::vector<std::int64_t>> &target_cells)
  {
    if (target_cells.has_value()) {
      return intersect(subgrid, target_cells.value());
    } else {
      return true;
    }
  }

  //! Inserts cell IDs from a subgrid into a container.
  //!
  //! @param subgrid The subgrid from which to extract cell IDs.
  //! @param target_cells Optional vector of target cell IDs. If provided, only these cells
  //!                     within the subgrid's range will be inserted into the container.
  //! @param container The container to store the cell IDs.
  //!
  //! If `target_cells` is provided, this function inserts the cell IDs from `target_cells`
  //! that are within the range of `subgrid` into `container`. If `target_cells` is not provided,
  //! all cell IDs from the `subgrid` are inserted into `container`.
  template<int dim>
  void insert_cells(UnfittedBinarySpacePart<dim> &tree,
    const std::optional<std::vector<std::int64_t>> &target_cells,
    const FuncSign sign)
  {
    // TODO: Implement this function for target cells.
    static_cast<void>(target_cells);
    assert(!target_cells.has_value());

    const auto status = sign == FuncSign::positive ? ImmersedCellStatus::empty : ImmersedCellStatus::full;

    tree.set_status(status);
  }


  //! @brief Sets up the interval arithmetic (Taylor model) for a box.
  //!
  //! @tparam dim Parametric dimension.
  //! @param domain Box over which the interval is defined.
  //! @return Interval-valued point spanning @p domain.
  template<int dim> Vector<alg::Interval<dim>, dim> create_interval_point(const BoundBox<dim> &domain)
  {
    Vector<alg::Interval<dim>, dim> xint;
    const auto mid_pt = domain.mid_point();
    for (int dir = 0; dir < dim; ++dir) {
      const auto beta = set_component<real, dim>(numbers::zero, dir, numbers::one);
      xint(dir) = alg::Interval<dim>(mid_pt(dir), beta);
      alg::Interval<dim>::delta(dir) = numbers::half * domain.length(dir);
    }
    return xint;
  }

  template<int dim>
  std::function<FuncSign(const BoundBox<dim> &)> create_compute_sign_function_multi(
    const funcs::BeziersIntersection<dim> &multi)
  {
    return [&multi](const BoundBox<dim> &domain) {
      const auto xint = create_interval_point<dim>(domain);

      // The domain is the intersection of the negative regions of every polynomial:
      // - if any polynomial is uniformly non-negative over the box, the box lies outside;
      // - if every polynomial is uniformly negative, the box lies inside;
      // - otherwise, the sign is undetermined.
      bool all_negative = true;
      for (const auto &bzr : multi.get_beziers()) {
        const auto res = (*bzr)(xint);
        if (res.uniformSign()) {
          if (!(res.alpha < 0.0)) {
            return FuncSign::positive;
          }
        } else {
          all_negative = false;
        }
      }
      return all_negative ? FuncSign::negative : FuncSign::undetermined;
    };
  }

  template<int dim>
  std::function<FuncSign(const BoundBox<dim> &)> create_compute_sign_function(const ImplicitFunc<dim> &phi)
  {
    if (const auto *multi = dynamic_cast<const funcs::BeziersIntersection<dim> *>(&phi); multi != nullptr) {
      return create_compute_sign_function_multi<dim>(*multi);
    }

    return [&phi](const BoundBox<dim> &domain) {
      const auto xint = create_interval_point<dim>(domain);

      const auto res = phi(xint);
      if (res.uniformSign()) {
        if (res.alpha < 0.0) {
          return FuncSign::negative;
        } else {
          return FuncSign::positive;
        }
      } else {
        return FuncSign::undetermined;
      }
    };
  }

  template<int dim> using QuadRule = typename ::algoim::QuadratureRule<dim>;

  template<int dim>
  ImmersedCellStatusTmp classify_cell_from_quad(const QuadRule<dim> &quad, const BoundBox<dim> &domain)
  {
    const auto cell_vol = quad.sumWeights();

    const Tolerance default_tol;
    if (default_tol.equal(cell_vol, numbers::zero)) {
      // It is an empty cell but with an unfitted boundary (on the cell's boundary).
      // We consider that the unfitted boundary belongs to a neighbor cell.
      return ImmersedCellStatusTmp::empty;
    } else if (default_tol.equal(cell_vol, domain.volume())) {
      // Full cell but with an unfitted boundary (on a cell's boundary).
      return ImmersedCellStatusTmp::full_with_unf_bdry;
    } else {
      // Cut cell with an unfitted boundary
      return ImmersedCellStatusTmp::cut;
    }
  }


  template<int dim>
  ImmersedCellStatusTmp classify_cell_general(const ImplicitFunc<dim> &phi, const BoundBox<dim> &domain)
  {
    constexpr int n_pts_dir{ 1 };
    const auto alg_domain = domain.to_hyperrectangle();
    const auto quad = ::algoim::quadGen(phi, alg_domain, -1, 0, n_pts_dir);
    return classify_cell_from_quad(quad, domain);
  }

  template<int dim> ImmersedCellStatusTmp classify_cell_Bezier(const BezierTP<dim> &bezier, const BoundBox<dim> &domain)
  {
    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);

    switch (bezier.sign()) {
    case FuncSign::positive:
      return ImmersedCellStatusTmp::empty;
    case FuncSign::negative:
      return ImmersedCellStatusTmp::full;
    default:
      break;
    }

    constexpr auto quad_strategy = alg::QuadStrategy::AlwaysGL;
    constexpr int n_pts_dir{ 1 };

    alg::ImplicitPolyQuadrature<dim> ipquad(bzr_domain.get_xarray());

    // Creates a quadrature for the negative part of the function.
    QuadRule<dim> quad;
    ipquad.integrate(
      quad_strategy, n_pts_dir, [&bzr_domain, &quad](const Vector<real, dim> &perm_point, const real weight) {
        const auto point = permute_vector_directions(perm_point);
        if (bzr_domain(point) < numbers::zero) {
          quad.nodes.emplace_back(point, weight);
        }
      });

    const BoundBox<dim> domain_0_1(numbers::zero, numbers::one);
    return classify_cell_from_quad(quad, domain_0_1);
  }

  //! @brief Rescales the polynomials of an intersection domain to a cell domain.
  template<int dim>
  std::vector<std::shared_ptr<BezierTP<dim, 1>>> rescale_beziers(const funcs::BeziersIntersection<dim> &multi,
    const BoundBox<dim> &domain)
  {
    std::vector<std::shared_ptr<BezierTP<dim, 1>>> bzrs;
    bzrs.reserve(multi.get_beziers().size());
    for (const auto &bzr : multi.get_beziers()) {
      auto bzr_domain = std::make_shared<BezierTP<dim, 1>>(*bzr);
      bzr_domain->rescale_domain(domain);
      bzrs.push_back(std::move(bzr_domain));
    }
    return bzrs;
  }

  template<int dim>
  ImmersedCellStatusTmp classify_cell_MultiBezier(const funcs::BeziersIntersection<dim> &multi,
    const BoundBox<dim> &domain)
  {
    const auto bzrs = rescale_beziers<dim>(multi, domain);

    // If any polynomial is uniformly positive, the intersection is empty in this cell.
    if (std::ranges::any_of(bzrs, [](const auto &bzr) { return bzr->sign() == FuncSign::positive; })) {
      return ImmersedCellStatusTmp::empty;
    }
    // If every polynomial is uniformly negative, the whole cell is inside the domain.
    if (std::ranges::all_of(bzrs, [](const auto &bzr) { return bzr->sign() == FuncSign::negative; })) {
      return ImmersedCellStatusTmp::full;
    }

    constexpr auto quad_strategy = alg::QuadStrategy::AlwaysGL;
    constexpr int n_pts_dir{ 1 };

    alg::ImplicitPolyQuadrature<dim> ipquad;
    for (const auto &bzr : bzrs) {
      const auto mask = alg::detail::nonzeroMask(bzr->get_xarray(), alg::booluarray<dim, ALGOIM_M>(true));
      if (!alg::detail::maskEmpty(mask)) {
        ipquad.phi.push_back(bzr->get_xarray(), mask);
      }
    }
    ipquad.build(true, false);

    // Creates a quadrature for the region where all the polynomials are negative.
    QuadRule<dim> quad;
    ipquad.integrate(
      quad_strategy, n_pts_dir, [&bzrs, &quad](const Vector<real, dim> &perm_point, const real weight) {
        const auto point = permute_vector_directions(perm_point);
        const bool inside =
          std::ranges::all_of(bzrs, [&point](const auto &bzr) { return (*bzr)(point) < numbers::zero; });
        if (inside) {
          quad.nodes.emplace_back(point, weight);
        }
      });

    const BoundBox<dim> domain_0_1(numbers::zero, numbers::one);
    return classify_cell_from_quad(quad, domain_0_1);
  }

  template<int dim> ImmersedCellStatusTmp classify_cell(const ImplicitFunc<dim> &phi, const BoundBox<dim> &domain)
  {
    if (const auto *multi = dynamic_cast<const funcs::BeziersIntersection<dim> *>(&phi); multi != nullptr) {
      return classify_cell_MultiBezier(*multi, domain);
    } else if (is_bezier(phi)) {
      const auto &bezier = dynamic_cast<const BezierTP<dim> &>(phi);
      return classify_cell_Bezier(bezier, domain);
    } else {
      return classify_cell_general(phi, domain);
    }
  }


  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  template<int dim>
  ImmersedFacetStatus classify_facet_full_unfitted_boundary(const ImmersedFacetStatus facet_status,
    const alg::HyperRectangle<real, dim> &domain,
    const int local_facet_id,
    const real cell_facet_vol,
    const int n_pts)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const real domain_facet_vol = prod(remove_component(domain.extent(), const_dir));

    const Tolerance tol;
    const Tolerance rel_tol{ tol.value() * 1.0e3 };
    const bool is_full_facet = tol.equal_rel(cell_facet_vol, domain_facet_vol, rel_tol);

    if (is_full_facet) {
      switch (facet_status) {
      case ImmersedFacetStatus::unf_bdry:
      case ImmersedFacetStatus::ext_bdry: {
        if (n_pts == 1) {
          return facet_status == ImmersedFacetStatus::unf_bdry ? ImmersedFacetStatus::full_unf_bdry
                                                               : ImmersedFacetStatus::full_ext_bdry;
        }
        break;
      }
      case ImmersedFacetStatus::cut:
        return ImmersedFacetStatus::full;
      default:
        break;
      }
    }

    return facet_status;
  }

  template<int dim>
  ImmersedFacetStatus classify_facet_from_quad(const ImplicitFunc<dim> &phi,
    const alg::HyperRectangle<real, dim> &domain,
    const int local_facet_id,
    const ImmersedCellStatusTmp cell_status,
    const QuadRule<dim> &facet_quad)
  {
    assert(cell_status == ImmersedCellStatusTmp::cut || cell_status == ImmersedCellStatusTmp::full_with_unf_bdry);

    if (facet_quad.nodes.empty()) {
      return cell_status == ImmersedCellStatusTmp::full_with_unf_bdry ? ImmersedFacetStatus::full_unf_bdry
                                                                      : ImmersedFacetStatus::empty;
    }

    const Tolerance default_tol;

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);

    bool ext_bdry{ false };
    bool unf_bdry{ false };
    bool cut_facet{ false };
    for (const auto &node : facet_quad.nodes) {
      if (on_levelset(phi, node.x, default_tol)) {

        const auto normal = phi.grad(node.x);
        const auto normal_comp = normal(const_dir) / norm(normal);
        if (!default_tol.equal(std::abs(normal_comp), numbers::one)) {
          continue;
        }

        if (side == 0 ? default_tol.is_negative(normal_comp) : default_tol.is_positive(normal_comp)) {
          unf_bdry = true;
        } else {
          ext_bdry = true;
        }
      } else {
        cut_facet = true;
      }
    }

    if (cell_status == ImmersedCellStatusTmp::full_with_unf_bdry && !cut_facet) {
      return ImmersedFacetStatus::full_unf_bdry;
    }

    constexpr std::array<ImmersedFacetStatus, 8> values{ ImmersedFacetStatus::empty,
      ImmersedFacetStatus::ext_bdry,
      ImmersedFacetStatus::unf_bdry,
      ImmersedFacetStatus::unf_bdry_ext_bdry,
      ImmersedFacetStatus::cut,
      ImmersedFacetStatus::cut_ext_bdry,
      ImmersedFacetStatus::cut_unf_bdry,
      ImmersedFacetStatus::cut_unf_bdry_ext_bdry };

    const int index = ext_bdry + (unf_bdry * 2) + (cut_facet * 4);
    const auto facet_status = at(values, index);

    const real cell_facet_vol = facet_quad.sumWeights();
    const int n_pts = static_cast<int>(facet_quad.nodes.size());

    return classify_facet_full_unfitted_boundary(facet_status, domain, local_facet_id, cell_facet_vol, n_pts);
  }


  template<int dim>
  ImmersedFacetStatus classify_facet_general(const ImplicitFunc<dim> &phi,
    const SubCartGridTP<dim> &subgrid,
    const int local_facet_id,
    const ImmersedCellStatusTmp cell_status)
  {
    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    const auto domain = subgrid.get_domain().to_hyperrectangle();
    constexpr int n_pts_dir{ 1 };
    const auto facet_quad = ::algoim::quadGen(phi, domain, const_dir, side, n_pts_dir);

    return classify_facet_from_quad(phi, domain, local_facet_id, cell_status, facet_quad);
  }

  template<int dim>
  ImmersedFacetStatus classify_facet_Bezier(const BezierTP<dim> &bezier,
    const BoundBox<dim> &domain,
    const int local_facet_id,
    const ImmersedCellStatusTmp cell_status)
  {
    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);
    const auto bzr_facet = bzr_domain.extract_facet(local_facet_id);

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    constexpr int n_pts_dir{ 1 };
    constexpr auto quad_strategy = alg::QuadStrategy::AlwaysGL;

    alg::ImplicitPolyQuadrature<dim - 1> ipquad(bzr_facet->get_xarray());

    QuadRule<dim> facet_quad;

    ipquad.integrate(quad_strategy,
      n_pts_dir,
      [&facet_quad, &bzr_facet, side, const_dir](const Vector<real, dim - 1> &perm_point, const real weight) {
        const auto point = permute_vector_directions(perm_point);
        if (bzr_facet->operator()(point) < numbers::zero) {
          const auto sup_point = add_component(point, const_dir, static_cast<real>(side));
          facet_quad.nodes.emplace_back(sup_point, weight);
        }
      });

    const BoundBox<dim> domain_01;
    return classify_facet_from_quad(bzr_domain, domain_01.to_hyperrectangle(), local_facet_id, cell_status, facet_quad);
  }

  template<int dim>
  ImmersedFacetStatus classify_facet_MultiBezier(const funcs::BeziersIntersection<dim> &multi,
    const BoundBox<dim> &domain,
    const int local_facet_id,
    const ImmersedCellStatusTmp cell_status)
  {
    const auto bzrs = rescale_beziers<dim>(multi, domain);

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    constexpr int n_pts_dir{ 1 };
    constexpr auto quad_strategy = alg::QuadStrategy::AlwaysGL;

    std::vector<std::shared_ptr<BezierTP<dim - 1, 1>>> facets;
    facets.reserve(bzrs.size());
    alg::ImplicitPolyQuadrature<dim - 1> ipquad;
    for (const auto &bzr : bzrs) {
      auto facet = bzr->extract_facet(local_facet_id);
      const auto mask = alg::detail::nonzeroMask(facet->get_xarray(), alg::booluarray<dim - 1, ALGOIM_M>(true));
      if (!alg::detail::maskEmpty(mask)) {
        ipquad.phi.push_back(facet->get_xarray(), mask);
      }
      facets.push_back(std::move(facet));
    }
    ipquad.build(true, false);

    QuadRule<dim> facet_quad;
    ipquad.integrate(quad_strategy,
      n_pts_dir,
      [&facets, &facet_quad, side, const_dir](const Vector<real, dim - 1> &perm_point, const real weight) {
        const auto point = permute_vector_directions(perm_point);
        const bool inside =
          std::ranges::all_of(facets, [&point](const auto &facet) { return (*facet)(point) < numbers::zero; });
        if (inside) {
          const auto sup_point = add_component(point, const_dir, static_cast<real>(side));
          facet_quad.nodes.emplace_back(sup_point, weight);
        }
      });

    const funcs::BeziersIntersection<dim> multi_domain(
      std::vector<std::shared_ptr<const BezierTP<dim, 1>>>(bzrs.cbegin(), bzrs.cend()));
    const BoundBox<dim> domain_01;
    return classify_facet_from_quad(
      multi_domain, domain_01.to_hyperrectangle(), local_facet_id, cell_status, facet_quad);
  }

  template<int dim>
  ImmersedFacetStatus classify_facet(const ImplicitFunc<dim> &phi,
    const SubCartGridTP<dim> &subgrid,
    const int local_facet_id,
    const ImmersedCellStatusTmp cell_status)
  {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    assert(0 <= local_facet_id && local_facet_id < dim * 2);

    if (cell_status == ImmersedCellStatusTmp::empty) {
      return ImmersedFacetStatus::empty;
    } else if (cell_status == ImmersedCellStatusTmp::full) {
      return ImmersedFacetStatus::full;
    }

    // NOTE: if needed, this function could speed-up by looking at the
    // facet state of neighbor cells. However, this involves a more
    // complicated algorithm that may not be worth it.

    if (const auto *multi = dynamic_cast<const funcs::BeziersIntersection<dim> *>(&phi); multi != nullptr) {
      return classify_facet_MultiBezier(*multi, subgrid.get_domain(), local_facet_id, cell_status);
    } else if (is_bezier(phi)) {
      const auto &bzr = dynamic_cast<const BezierTP<dim> &>(phi);
      return classify_facet_Bezier(bzr, subgrid.get_domain(), local_facet_id, cell_status);
    } else {
      return classify_facet_general(phi, subgrid, local_facet_id, cell_status);
    }
  }


}// namespace


template<int dim>
UnfittedImplDomain<dim>::UnfittedImplDomain(const FuncPtr phi, const GridPtr grid)
  : UnfittedDomain<dim>(grid), phi_(phi)
{
  assert(this->phi_ != nullptr);

  const std::optional<std::vector<std::int64_t>> cells;
  const auto compute_sign = create_compute_sign_function<dim>(*phi);
  this->create_decomposition(*(this->binary_sp_part_), compute_sign, cells);
  this->init_full_cells_with_unf_bdry();
}


template<int dim>
UnfittedImplDomain<dim>::UnfittedImplDomain(const FuncPtr phi,
  const GridPtr grid,
  const std::vector<std::int64_t> &cells)
  : UnfittedDomain<dim>(grid), phi_(phi)
{
  assert(this->phi_ != nullptr);

  const auto compute_sign = create_compute_sign_function<dim>(*phi);
  this->create_decomposition(*(this->binary_sp_part_), compute_sign, cells);
  this->init_full_cells_with_unf_bdry();
}

template<int dim>
// NOLINTNEXTLINE (misc-no-recursion)
void UnfittedImplDomain<dim>::create_decomposition(KDTree &tree,
  const std::function<FuncSign(const BoundBox<dim> &)> &func_sign,
  const std::optional<std::vector<std::int64_t>> &target_cells)
{
  const auto subgrid = tree.get_subgrid();
  if (!intersect(*subgrid, target_cells)) {
    return;
  }

  const auto domain = subgrid->get_domain();
  // We slightly enlarge the domain to deal with edge cases.
  const auto ext_domain = domain.extend(numbers::eps * 1000);
  const auto sign = func_sign(ext_domain);

  if (sign == FuncSign::undetermined) {
    if (subgrid->is_unique_cell()) {
      this->classify_undetermined_sign_cell(tree);
    } else {
      tree.branch();
      this->create_decomposition(*tree.get_child(0), func_sign, target_cells);
      this->create_decomposition(*tree.get_child(1), func_sign, target_cells);
    }
  } else {
    assert(sign == FuncSign::positive || sign == FuncSign::negative);
    insert_cells(tree, target_cells, sign);
  }
}

// Note that that full cells with unfitted boundaries are transformed to full.
ImmersedCellStatus transform_cell_status(const ImmersedCellStatusTmp cell_status)
{
  switch (cell_status) {
  case ImmersedCellStatusTmp::cut:
    return ImmersedCellStatus::cut;
  case ImmersedCellStatusTmp::full:
  case ImmersedCellStatusTmp::full_with_unf_bdry:
    return ImmersedCellStatus::full;
  case ImmersedCellStatusTmp::empty:
    return ImmersedCellStatus::empty;
  }
}

template<int dim> void UnfittedImplDomain<dim>::classify_undetermined_sign_cell(KDTree &tree)
{
  const auto subgrid = tree.get_subgrid();
  assert(subgrid->is_unique_cell());

  const auto domain = subgrid->get_domain();
  auto cell_status = classify_cell(*this->phi_, domain);

  if (cell_status == ImmersedCellStatusTmp::cut || cell_status == ImmersedCellStatusTmp::full_with_unf_bdry) {
    typename UnfittedImplDomain<dim>::FacetsStatus facets{};
    bool all_full = true;
    for (int local_facet_id = 0; local_facet_id < dim * 2; ++local_facet_id) {
      const auto facet_status = classify_facet(*this->phi_, *subgrid, local_facet_id, cell_status);
      if (facet_status != ImmersedFacetStatus::full) {
        all_full = false;
      }

      at(facets, local_facet_id) = facet_status;
    }

    if (cell_status == ImmersedCellStatusTmp::full_with_unf_bdry && all_full) {
      cell_status = ImmersedCellStatusTmp::full;
    } else {
      this->facets_status_.emplace(subgrid->get_single_cell(), facets);
    }
  }

  tree.set_status(transform_cell_status(cell_status));
}


template<int dim> auto UnfittedImplDomain<dim>::get_impl_func() const -> FuncPtr
{
  return this->phi_;
}


// Instantiations
template class UnfittedImplDomain<2>;
template class UnfittedImplDomain<3>;

}// namespace qugar::impl
