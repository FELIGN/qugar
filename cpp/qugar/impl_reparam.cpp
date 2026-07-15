// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_reparam.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of reparameterization generization tools implicit functions on grids.
//! @date 2025-01-13
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_reparam.hpp>

#include <qugar/bezier_tp.hpp>
#include <qugar/impl_funcs_lib.hpp>
#include <qugar/impl_reparam_bezier.hpp>
#include <qugar/impl_reparam_general.hpp>
#include <qugar/impl_reparam_mesh.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/numbers.hpp>
#include <qugar/tolerance.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <ranges>
#include <vector>


namespace qugar::impl {

template<int dim, bool levelset>
std::shared_ptr<const ImplReparamMesh<levelset ? dim - 1 : dim, dim>>
  create_reparameterization(const UnfittedImplDomain<dim> &unf_domain, const int n_pts_dir, const bool merge_points)
{
  const auto rng = std::ranges::iota_view<std::int64_t, std::int64_t>{ 0,
    static_cast<std::int64_t>(unf_domain.get_grid()->get_num_cells()) };
  const std::vector<std::int64_t> cells(rng.begin(), rng.end());
  return create_reparameterization<dim, levelset>(unf_domain, cells, n_pts_dir, merge_points);
}

template<int dim, bool levelset>
std::shared_ptr<const ImplReparamMesh<levelset ? dim - 1 : dim, dim>> create_reparameterization(
  const UnfittedImplDomain<dim> &unf_domain,
  const std::vector<std::int64_t> &cells,
  const int n_pts_dir,
  const bool merge_points)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(1 < n_pts_dir);

  std::vector<std::int64_t> sorted_cells = cells;
  std::ranges::sort(sorted_cells);

  std::vector<std::int64_t> cut_cells;
  unf_domain.get_cut_cells(sorted_cells, cut_cells);

  const auto grid = unf_domain.get_grid();

  static const int param_dim = levelset ? dim - 1 : dim;
  const auto reparam = std::make_shared<ImplReparamMesh<param_dim, dim>>(n_pts_dir);

  const auto n_reparam_tiles_per_cell = param_dim == 2 ? 3 : 13;// Estimation
  const auto n_cells_est = (cut_cells.size() * n_reparam_tiles_per_cell) + unf_domain.get_num_full_cells();
  reparam->reserve_cells(n_cells_est);

  const auto phi = unf_domain.get_impl_func();
  const auto *bzr = dynamic_cast<const BezierTP<dim, 1> *>(phi.get());
  const bool is_bzr = bzr != nullptr;
  const auto *multi = dynamic_cast<const funcs::BeziersIntersection<dim> *>(phi.get());


  for (const auto &cell_id : cut_cells) {
    const auto domain = grid->get_cell_domain(cell_id);

    if (multi != nullptr) {
      std::vector<std::shared_ptr<const BezierTP<dim, 1>>> bzrs;
      bzrs.reserve(multi->get_beziers().size());
      for (const auto &poly : multi->get_beziers()) {
        auto bzr_domain = std::make_shared<BezierTP<dim, 1>>(*poly);
        bzr_domain->rescale_domain(domain);
        bzrs.push_back(std::move(bzr_domain));
      }
      reparam_Beziers<dim, levelset>(bzrs, domain, *reparam);
    } else if (is_bzr) {
      BezierTP<dim> bzr_domain(*bzr);
      bzr_domain.rescale_domain(domain);

      reparam_Bezier<dim, levelset>(bzr_domain, domain, *reparam);
    } else {
      reparam_general<dim, levelset>(*phi, domain, *reparam);
    }
  }

  if constexpr (!levelset) {
    std::vector<std::int64_t> full_cells;
    unf_domain.get_full_cells(sorted_cells, full_cells);

    constexpr bool wirebasket = true;
    reparam->add_full_cells(*grid, full_cells, wirebasket);
  }

  if (merge_points) {
    const Tolerance tol(1.0e4 * numbers::eps);
    reparam->merge_coincident_points(tol);
  }


  return reparam;
}

// Instantiations.

template std::shared_ptr<const ImplReparamMesh<2, 2>> create_reparameterization<2, false>(const UnfittedImplDomain<2> &,
  const std::vector<std::int64_t> &,
  const int,
  const bool);
template std::shared_ptr<const ImplReparamMesh<1, 2>> create_reparameterization<2, true>(const UnfittedImplDomain<2> &,
  const std::vector<std::int64_t> &,
  const int,
  const bool);

template std::shared_ptr<const ImplReparamMesh<3, 3>> create_reparameterization<3, false>(const UnfittedImplDomain<3> &,
  const std::vector<std::int64_t> &,
  const int,
  const bool);
template std::shared_ptr<const ImplReparamMesh<2, 3>> create_reparameterization<3, true>(const UnfittedImplDomain<3> &,
  const std::vector<std::int64_t> &,
  const int,
  const bool);

template std::shared_ptr<const ImplReparamMesh<2, 2>>
  create_reparameterization<2, false>(const UnfittedImplDomain<2> &, const int, const bool);
template std::shared_ptr<const ImplReparamMesh<1, 2>>
  create_reparameterization<2, true>(const UnfittedImplDomain<2> &, const int, const bool);

template std::shared_ptr<const ImplReparamMesh<3, 3>>
  create_reparameterization<3, false>(const UnfittedImplDomain<3> &, const int, const bool);
template std::shared_ptr<const ImplReparamMesh<2, 3>>
  create_reparameterization<3, true>(const UnfittedImplDomain<3> &, const int, const bool);

}// namespace qugar::impl