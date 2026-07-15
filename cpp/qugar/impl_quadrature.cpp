// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_quadrature.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of quadratures for general implicit functions on grids.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/impl_quadrature.hpp>

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/cut_quadrature.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_funcs_lib.hpp>
#include <qugar/impl_unfitted_domain.hpp>
#include <qugar/impl_utils.hpp>
#include <qugar/numbers.hpp>
#include <qugar/point.hpp>
#include <qugar/quadrature.hpp>
#include <qugar/tolerance.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>
#include <qugar/vector.hpp>

#include <algoim/bernstein.hpp>
#include <algoim/booluarray.hpp>
#include <algoim/polyset.hpp>
#include <algoim/quadrature_general.hpp>
#include <algoim/quadrature_multipoly.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>


namespace qugar::impl {

namespace alg = ::algoim;

namespace {


  template<int dim> struct CutCellsQuadWrapper
  {
    explicit CutCellsQuadWrapper(CutCellsQuad<dim> &_quad, const BoundBox<dim> &_domain) : quad(_quad), domain(_domain)
    {}

    // NOLINTNEXTLINE (cppcoreguidelines-avoid-const-or-ref-data-members)
    CutCellsQuad<dim> &quad;
    BoundBox<dim> domain;

    // evalIntegrand records quadrature nodes when it is invoked by ImplicitIntegral
    void evalIntegrand(const Point<dim> &point, const real weight)
    {
      quad.points.push_back(domain.scale_to_0_1(point));
      quad.weights.push_back(weight / domain.volume());
    }
  };

  template<int dim> struct CutUnfBoundsQuadWrapper
  {
    explicit CutUnfBoundsQuadWrapper(CutUnfBoundsQuad<dim> &_quad,
      const ImplicitFunc<dim> &_phi,
      const BoundBox<dim> &_domain)
      : quad(_quad), phi(_phi), domain(_domain)
    {}

    // NOLINTBEGIN (cppcoreguidelines-avoid-const-or-ref-data-members)
    CutUnfBoundsQuad<dim> &quad;
    const ImplicitFunc<dim> &phi;
    const BoundBox<dim> &domain;
    // NOLINTEND (cppcoreguidelines-avoid-const-or-ref-data-members)

    // evalIntegrand records quadrature nodes when it is invoked by ImplicitIntegral
    void evalIntegrand(const Point<dim> &point, const real weight)
    {
      quad.points.push_back(domain.scale_to_0_1(point));

      // Unit normal in element's domain.
      Point<dim> normal = phi.grad(point);
      normal /= norm(normal);

      // Mapping the normal from element's domain to [0,1]^dim.
      // It is done left-multiplying the unit normal
      // in element domain with DF^-T, where F is the map
      // from the element domain to [0,1]^dim, and
      // DF = diag(1/dx, 1/dy, ...), where dx, dy, ... are
      // the lengths of the element's bounding box along
      // the different directions.
      for (int dir = 0; dir < dim; ++dir) {
        normal(dir) *= domain.length(dir);
      }
      const auto norm_normal = norm(normal);
      normal /= norm_normal;// This is the unit normal in [0,1]^dim

      quad.weights.push_back(weight * norm_normal / domain.volume());

      quad.normals.push_back(normal);
    }
  };

  template<int dim> struct CutIsoBoundsQuadWrapper
  {
    explicit CutIsoBoundsQuadWrapper(CutIsoBoundsQuad<dim> &_quad,
      const BoundBox<dim> &_face_domain,
      const int _const_dir)
      : quad(_quad), face_domain(_face_domain), const_dir(_const_dir)
    {}

    // NOLINTBEGIN (cppcoreguidelines-avoid-const-or-ref-data-members)
    CutIsoBoundsQuad<dim> &quad;
    // const ImplicitFunc<dim> &phi;
    const BoundBox<dim> &face_domain;
    // NOLINTEND (cppcoreguidelines-avoid-const-or-ref-data-members)
    int const_dir;

    // evalIntegrand records quadrature nodes when it is invoked by ImplicitIntegral
    void evalIntegrand(const Point<dim + 1> &point, const real weight)
    {
      const auto face_point = remove_component(point, const_dir);
      quad.points.push_back(face_domain.scale_to_0_1(face_point));
      quad.weights.push_back(weight / face_domain.volume());
    }
  };

  //! @brief Builds an Algoim multi-polynomial quadrature from a list of Bernstein polynomials.
  //!
  //! Replicates what Algoim's `ImplicitPolyQuadrature` constructors do for one or two
  //! polynomials, but for an arbitrary number of them: each polynomial is pushed (its
  //! coefficients are copied into the quadrature's own buffer) with its non-zero mask, and
  //! the quadrature hierarchy is built. The resulting object is self-contained, so the input
  //! views need only be valid during this call.
  //!
  //! @tparam N Parametric dimension.
  //! @param beziers Bernstein polynomials defining the domain.
  //! @return Built multi-polynomial quadrature.
  //!
  //! @note `algoim::xarray` views are non-copyable, so the polynomials are passed as pointers
  //! and their coefficient views handed to `phi.push_back` (which copies the data into the
  //! quadrature's own buffer).
  template<int N>
  alg::ImplicitPolyQuadrature<N> create_multipoly_quadrature(
    const std::vector<std::shared_ptr<BezierTP<N, 1>>> &beziers)
  {
    alg::ImplicitPolyQuadrature<N> ipquad;
    for (const auto &bezier : beziers) {
      const auto &xarray = bezier->get_xarray();
      const auto mask = alg::detail::nonzeroMask(xarray, alg::booluarray<N, ALGOIM_M>(true));
      if (!alg::detail::maskEmpty(mask)) {
        ipquad.phi.push_back(xarray, mask);
      }
    }
    ipquad.build(true, false);
    return ipquad;
  }

  //! @brief Rescales the Bezier polynomials of an intersection domain to a given cell domain.
  //!
  //! @tparam dim Parametric dimension.
  //! @param multi Intersection-of-Beziers implicit function.
  //! @param domain Cell domain the polynomials are rescaled to.
  //! @return Heap-allocated rescaled polynomials (stable addresses, so their coefficient views
  //! remain valid).
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

  template<int dim, bool is_srf, typename T>
  void compute_quadrature_with_algoim_general(const ImplicitFunc<dim> &phi,
    const BoundBox<dim> &domain,
    const int n_pts_dir,
    T &quad_wrapper)
  {
    auto &quad = quad_wrapper.quad;
    const auto n_pts_0 = quad.points.size();

    constexpr auto n_psi = static_cast<unsigned int>(1) << static_cast<unsigned int>(dim - 1);
    std::array<alg::PsiCode<dim>, n_psi> psi;
    psi[0] = alg::PsiCode<dim>(0, -1);

    const Vector<bool, dim> free = true;

    alg::ImplicitIntegral<dim, dim, ImplicitFunc<dim>, T, is_srf>(
      phi, quad_wrapper, free, psi, 1, domain.to_hyperrectangle(), n_pts_dir);

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_cell.push_back(n_pts);
  }

  template<int dim, bool is_srf, typename T>
  void compute_quadrature_with_algoim_Bezier(const BezierTP<dim, 1> &bezier,
    const BoundBox<dim> &domain,
    const int n_pts_dir,
    T &quad)
  {
    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);

    const auto n_pts_0 = quad.points.size();

    alg::ImplicitPolyQuadrature<dim> ipquad(bzr_domain.get_xarray());

    constexpr auto strategy = alg::QuadStrategy::AutoMixed;

    if constexpr (is_srf) {
      ipquad.integrate_surf(strategy,
        n_pts_dir,
        [&quad, &bzr_domain](
          const Point<dim> &perm_point_01, const real weight_01, const Point<dim> & /*aggreg_normal_01*/) {
          quad.points.push_back(permute_vector_directions(perm_point_01));
          quad.weights.push_back(weight_01);

          auto grad = ::algoim::bernstein::evalBernsteinPolyGradient(bzr_domain.get_xarray(), perm_point_01);
          grad /= norm(grad);
          quad.normals.push_back(permute_vector_directions(grad));
        });

    } else {
      ipquad.integrate(
        strategy, n_pts_dir, [&quad, &bzr_domain](const Point<dim> &perm_point_01, const real weight_01) {
          const auto point_01 = permute_vector_directions(perm_point_01);
          if (bzr_domain(point_01) < numbers::zero) {
            quad.points.push_back(point_01);
            quad.weights.push_back(weight_01);
          }
        });
    }

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_cell.push_back(n_pts);
  }

  template<int dim, bool is_srf, typename T>
  void compute_quadrature_with_algoim_MultiBezier(const funcs::BeziersIntersection<dim> &multi,
    const BoundBox<dim> &domain,
    const int n_pts_dir,
    T &quad)
  {
    const auto bzrs = rescale_beziers<dim>(multi, domain);

    const auto n_pts_0 = quad.points.size();

    auto ipquad = create_multipoly_quadrature<dim>(bzrs);

    constexpr auto strategy = alg::QuadStrategy::AutoMixed;

    if constexpr (is_srf) {
      const Tolerance tol;
      ipquad.integrate_surf(strategy,
        n_pts_dir,
        [&quad, &bzrs, tol](const Point<dim> &perm_point_01, const real weight_01, const Point<dim> &perm_normal_01) {
          const auto point_01 = permute_vector_directions(perm_point_01);

          // The surface generated by Algoim contains the zero-level-set of every
          // polynomial. Only the pieces bounding the intersection domain are kept, i.e.,
          // those where no other polynomial is strictly positive.
          for (const auto &bzr : bzrs) {
            if (tol.is_positive((*bzr)(point_01))) {
              return;
            }
          }

          quad.points.push_back(point_01);
          quad.weights.push_back(weight_01);

          // Algoim provides the (scaled) gradient of the active polynomial as normal.
          auto normal = permute_vector_directions(perm_normal_01);
          normal /= norm(normal);
          quad.normals.push_back(normal);
        });

    } else {
      ipquad.integrate(
        strategy, n_pts_dir, [&quad, &bzrs](const Point<dim> &perm_point_01, const real weight_01) {
          const auto point_01 = permute_vector_directions(perm_point_01);
          const bool inside =
            std::ranges::all_of(bzrs, [&point_01](const auto &bzr) { return (*bzr)(point_01) < numbers::zero; });
          if (inside) {
            quad.points.push_back(point_01);
            quad.weights.push_back(weight_01);
          }
        });
    }

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_cell.push_back(n_pts);
  }

  template<int dim, bool is_srf, typename T>
  void compute_quadrature_with_algoim(const ImplicitFunc<dim> &phi,
    const BoundBox<dim> &domain,
    const int n_pts_dir,
    T &quad_wrapper)
  {
    if (const auto *multi = dynamic_cast<const funcs::BeziersIntersection<dim> *>(&phi); multi != nullptr) {
      compute_quadrature_with_algoim_MultiBezier<dim, is_srf>(*multi, domain, n_pts_dir, quad_wrapper.quad);
    } else if (is_bezier(phi)) {
      const auto &bezier = dynamic_cast<const BezierTP<dim, 1> &>(phi);
      compute_quadrature_with_algoim_Bezier<dim, is_srf>(bezier, domain, n_pts_dir, quad_wrapper.quad);
    } else {
      compute_quadrature_with_algoim_general<dim, is_srf>(phi, domain, n_pts_dir, quad_wrapper);
    }
  }

  template<typename T, const bool is_facet_quad> void erase_points_in_quad(std::vector<int> &points_to_erase, T &quad)
  {
    // Sort in descending order
    std::ranges::sort(points_to_erase, std::ranges::greater());

    for (const auto &pt_id : points_to_erase) {
      quad.points.erase(quad.points.begin() + pt_id);
      quad.weights.erase(quad.weights.begin() + pt_id);
      if constexpr (!is_facet_quad) {
        quad.normals.erase(quad.normals.begin() + pt_id);
      }
    }

    const auto n_pts_to_erase = static_cast<int>(points_to_erase.size());
    if constexpr (is_facet_quad) {
      quad.n_pts_per_facet.back() -= n_pts_to_erase;
    } else {
      quad.n_pts_per_cell.back() -= n_pts_to_erase;
    }
  }

  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void purge_facet_points(const ImplicitFunc<dim> &func,
    const UnfittedImplDomain<dim> &unf_domain,
    const std::int64_t cell_id,
    const int local_facet_id,
    CutUnfBoundsQuad<dim> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    const bool unf_bound = unf_domain.has_unfitted_boundary(cell_id, local_facet_id);
    const bool ext_bound = unf_domain.has_external_boundary(cell_id, local_facet_id);
    if (!unf_bound && !ext_bound) {
      return;
    }

    const Tolerance tol(1000.0 * numbers::eps);

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    const auto domain = unf_domain.get_grid()->get_cell_domain(cell_id);
    const auto facet_coord_0_1 = side == 0 ? numbers::zero : numbers::one;

    const int n_pts = quad.n_pts_per_cell.back();
    const auto n_tot_pts = static_cast<int>(quad.points.size());

    std::vector<int> points_to_erase;

    // NOLINTNEXTLINE (misc-const-correctness)
    for (int pt_id = n_tot_pts - n_pts; pt_id < n_tot_pts; ++pt_id) {

      const auto &pt_01 = at(quad.points, pt_id);

      if (!tol.equal(pt_01(const_dir), facet_coord_0_1)) {
        // Point not on facet.
        continue;
      }

      const auto normal = at(quad.normals, pt_id);
      const auto normal_comp = normal(const_dir);
      if (tol.smaller_than(std::abs(normal_comp), numbers::one)) {
        // Not normal to facet.
        continue;
      }

      if (on_levelset(func, domain.scale_from_0_1(pt_01), tol)) {
        points_to_erase.push_back(pt_id);
      }
    }

    erase_points_in_quad<CutUnfBoundsQuad<dim>, false>(points_to_erase, quad);
  }


  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void purge_facet_points(const ImplicitFunc<dim> &func,
    const UnfittedImplDomain<dim> &unf_domain,
    const std::int64_t cell_id,
    const int local_facet_id,
    bool purge_unf_bdry,
    bool purge_unf_ext_bdry,
    bool purge_cut,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    const bool has_unf_bdry = unf_domain.has_unfitted_boundary(cell_id, local_facet_id);
    purge_unf_bdry = purge_unf_bdry && has_unf_bdry;
    purge_unf_ext_bdry = purge_unf_ext_bdry && has_unf_bdry;
    purge_cut = purge_cut && unf_domain.is_cut_facet(cell_id, local_facet_id);

    if (!purge_unf_bdry && !purge_unf_ext_bdry && !purge_cut) {
      return;
    }

    const Tolerance tol(1000.0 * numbers::eps);

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    const auto domain = unf_domain.get_grid()->get_cell_domain(cell_id);
    const auto facet_coord_0_1 = side == 0 ? numbers::zero : numbers::one;

    const auto n_pts = quad.n_pts_per_facet.back();
    const auto n_tot_pts = static_cast<int>(quad.points.size());

    std::vector<int> points_to_purge;
    points_to_purge.reserve(static_cast<std::size_t>(n_pts));

    Point<dim> point;
    // NOLINTNEXTLINE (misc-const-correctness)
    for (int pt_id = n_tot_pts - n_pts; pt_id < n_tot_pts; ++pt_id) {

      const auto &pt_01 = at(quad.points, pt_id);
      const Point<dim> sup_pt_01 = add_component<real, dim - 1>(pt_01, const_dir, facet_coord_0_1);
      point = domain.scale_from_0_1(sup_pt_01);

      if (on_levelset(func, point, tol)) {// On unfitted boundary, either internal or external.
        const auto normal = func.grad(point);
        const auto normal_comp = normal(const_dir);

        const auto pointing_outside = side == 0 ? tol.is_negative(normal_comp) : tol.is_positive(normal_comp);
        if ((pointing_outside && purge_unf_bdry) || (!pointing_outside && purge_unf_ext_bdry)) {
          points_to_purge.push_back(pt_id);
        }
      } else if (purge_cut) {// Not on unfitted boundary, so on cut facet.
        points_to_purge.push_back(pt_id);
      }
    }

    erase_points_in_quad<CutIsoBoundsQuad<dim - 1>, true>(points_to_purge, quad);
  }

  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void add_unf_bdry_quad(const UnfittedImplDomain<dim> &unf_domain,
    const std::int64_t cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutUnfBoundsQuad<dim> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    if (!unf_domain.has_unfitted_boundary(cell_id, local_facet_id)) {
      return;
    }

    CutIsoBoundsQuad<dim - 1> facet_quad;
    constexpr bool remove_unf_bdry = false;
    constexpr bool remove_cut = true;
    create_facet_quadrature(unf_domain, cell_id, local_facet_id, n_pts_dir, remove_unf_bdry, remove_cut, facet_quad);

    const auto n_pts = facet_quad.n_pts_per_facet.back();

    quad.points.reserve(quad.points.size() + static_cast<std::size_t>(n_pts));
    quad.weights.reserve(quad.weights.size() + static_cast<std::size_t>(n_pts));
    quad.normals.reserve(quad.normals.size() + static_cast<std::size_t>(n_pts));

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);
    const auto facet_coord_0_1 = side == 0 ? numbers::zero : numbers::one;

    Point<dim> normal;
    normal(const_dir) = side == 0 ? -numbers::one : numbers::one;

    for (int i = 0; i < n_pts; ++i) {
      ++quad.n_pts_per_cell.back();
      const auto &facet_pt = at(facet_quad.points, i);
      const Point<dim> pt = add_component<real, dim - 1>(facet_pt, const_dir, facet_coord_0_1);
      quad.points.push_back(pt);

      quad.weights.push_back(at(facet_quad.weights, i));
      quad.normals.push_back(normal);
    }
  }


  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void compute_facet_quadrature_with_algoim_general(const ImplicitFunc<dim> &phi,
    const UnfittedImplDomain<dim> &unf_domain,
    const std::int64_t cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    static_assert(dim == 2 || dim == 3, "Invalid dimension.");
    assert(0 < n_pts_dir);
    assert(unf_domain.is_cut_facet(cell_id, local_facet_id));

    const int const_dir = get_facet_constant_dir<dim>(local_facet_id);
    const int side = get_facet_side<dim>(local_facet_id);

    const auto grid = unf_domain.get_grid();
    const auto domain = grid->get_cell_domain(cell_id);
    const auto facet_domain = domain.slice(const_dir);

    CutIsoBoundsQuadWrapper<dim - 1> quad_wrapper(quad, facet_domain, const_dir);

    constexpr auto n_psi = static_cast<unsigned int>(1) << static_cast<unsigned int>(dim - 1);
    std::array<alg::PsiCode<dim>, n_psi> psi;
    psi[0] = alg::PsiCode<dim>(set_component<int, dim>(0, const_dir, side), -1);

    const auto free = set_component<int, dim>(true, const_dir, false);

    const auto n_pts_0 = quad.points.size();

    alg::ImplicitIntegral<dim - 1, dim, ImplicitFunc<dim>, CutIsoBoundsQuadWrapper<dim - 1>, false>(
      phi, quad_wrapper, free, psi, 1, domain.to_hyperrectangle(), n_pts_dir);

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_facet.push_back(n_pts);
  }

  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void compute_facet_quadrature_with_algoim_Bezier(const BezierTP<dim> &bezier,
    const UnfittedImplDomain<dim> &unf_domain,
    const std::int64_t cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    static_assert(dim == 2 || dim == 3, "Invalid dimension.");

    assert(0 < n_pts_dir);
    assert(unf_domain.is_cut_facet(cell_id, local_facet_id));


    const auto grid = unf_domain.get_grid();
    const auto domain = grid->get_cell_domain(cell_id);

    BezierTP<dim> bzr_domain(bezier);
    bzr_domain.rescale_domain(domain);
    const auto bezier_facet = bzr_domain.extract_facet(local_facet_id);

    const auto n_pts_0 = quad.points.size();

    alg::ImplicitPolyQuadrature<dim - 1> ipquad(bezier_facet->get_xarray());

    constexpr auto strategy = alg::QuadStrategy::AutoMixed;
    ipquad.integrate(
      strategy, n_pts_dir, [&quad, bezier_facet](const Point<dim - 1> &perm_point_01, const real weight_01) {
        const auto point_01 = permute_vector_directions(perm_point_01);
        if (bezier_facet->operator()(point_01) < numbers::zero) {
          quad.points.push_back(point_01);
          quad.weights.push_back(weight_01);
        }
      });

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_facet.push_back(n_pts);
  }

  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void compute_facet_quadrature_with_algoim_MultiBezier(const funcs::BeziersIntersection<dim> &multi,
    const UnfittedImplDomain<dim> &unf_domain,
    const std::int64_t cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    static_assert(dim == 2 || dim == 3, "Invalid dimension.");

    assert(0 < n_pts_dir);
    assert(unf_domain.is_cut_facet(cell_id, local_facet_id));

    const auto grid = unf_domain.get_grid();
    const auto domain = grid->get_cell_domain(cell_id);

    const auto bzrs = rescale_beziers<dim>(multi, domain);

    std::vector<std::shared_ptr<BezierTP<dim - 1, 1>>> facets;
    facets.reserve(bzrs.size());
    for (const auto &bzr : bzrs) {
      facets.push_back(bzr->extract_facet(local_facet_id));
    }

    const auto n_pts_0 = quad.points.size();

    auto ipquad = create_multipoly_quadrature<dim - 1>(facets);

    constexpr auto strategy = alg::QuadStrategy::AutoMixed;
    ipquad.integrate(
      strategy, n_pts_dir, [&quad, &facets](const Point<dim - 1> &perm_point_01, const real weight_01) {
        const auto point_01 = permute_vector_directions(perm_point_01);
        const bool inside =
          std::ranges::all_of(facets, [&point_01](const auto &facet) { return (*facet)(point_01) < numbers::zero; });
        if (inside) {
          quad.points.push_back(point_01);
          quad.weights.push_back(weight_01);
        }
      });

    const auto n_pts = static_cast<int>(quad.points.size() - n_pts_0);
    quad.n_pts_per_facet.push_back(n_pts);
  }

  template<int dim>
  // NOLINTBEGIN (bugprone-easily-swappable-parameters)
  void compute_facet_quadrature_with_algoim(const ImplicitFunc<dim> &phi,
    const UnfittedImplDomain<dim> &unf_domain,
    const std::int64_t cell_id,
    const int local_facet_id,
    const int n_pts_dir,
    CutIsoBoundsQuad<dim - 1> &quad)
  // NOLINTEND (bugprone-easily-swappable-parameters)
  {
    if (const auto *multi = dynamic_cast<const funcs::BeziersIntersection<dim> *>(&phi); multi != nullptr) {
      compute_facet_quadrature_with_algoim_MultiBezier<dim>(
        *multi, unf_domain, cell_id, local_facet_id, n_pts_dir, quad);
    } else if (is_bezier(phi)) {
      const auto &bezier = dynamic_cast<const BezierTP<dim, 1> &>(phi);
      compute_facet_quadrature_with_algoim_Bezier(bezier, unf_domain, cell_id, local_facet_id, n_pts_dir, quad);
    } else {
      compute_facet_quadrature_with_algoim_general(phi, unf_domain, cell_id, local_facet_id, n_pts_dir, quad);
    }
  }


}// namespace

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void create_cell_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  const std::int64_t cell_id,
  const int n_pts_dir,
  CutCellsQuad<dim> &quad)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);
  assert(0 <= cell_id);

  auto &points = quad.points;
  auto &weights = quad.weights;
  auto &n_pts_per_cell = quad.n_pts_per_cell;


  if (unf_domain.is_cut_cell(cell_id)) {
    const auto grid = unf_domain.get_grid();
    const auto phi = unf_domain.get_impl_func();

    const auto domain = grid->get_cell_domain(cell_id);

    CutCellsQuadWrapper<dim> quad_wrapper(quad, domain);
    compute_quadrature_with_algoim<dim, false>(*phi, domain, n_pts_dir, quad_wrapper);

  } else if (unf_domain.is_full_cell(cell_id)) {
    const int n_pts_per_quad_set = n_pts_dir * n_pts_dir * (dim == 3 ? n_pts_dir : 1);

    static auto gauss_01 = Quadrature<dim>::create_Gauss_01(n_pts_dir);
    if (gauss_01->get_num_points() != static_cast<std::size_t>(n_pts_per_quad_set)) {
      gauss_01 = Quadrature<dim>::create_Gauss_01(n_pts_dir);
    }

    const auto &gauss_pt = gauss_01->points();
    points.insert(points.end(), gauss_pt.cbegin(), gauss_pt.cend());

    const auto &gauss_wg = gauss_01->weights();
    weights.insert(weights.end(), gauss_wg.cbegin(), gauss_wg.cend());

    n_pts_per_cell.push_back(n_pts_per_quad_set);
  } else {// empty
    n_pts_per_cell.push_back(0);
  }
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void create_cell_unfitted_bound_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  const std::int64_t cell_id,
  const int n_pts_dir,
  const bool include_facet_unf_bdry,
  const bool exclude_ext_bdry,
  CutUnfBoundsQuad<dim> &quad)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 < n_pts_dir);
  assert(0 <= cell_id);

  auto &n_pts_per_cell = quad.n_pts_per_cell;

  if (!unf_domain.is_cell_with_unf_bdry(cell_id)) {
    n_pts_per_cell.push_back(0);
    return;
  }

  constexpr int n_local_facets = dim * 2;

  if (unf_domain.is_cut_cell(cell_id)) {
    const auto grid = unf_domain.get_grid();
    const auto domain = grid->get_cell_domain(cell_id);

    const auto phi = unf_domain.get_impl_func();
    CutUnfBoundsQuadWrapper<dim> quad_wrapper(quad, *phi, domain);
    compute_quadrature_with_algoim<dim, true>(*phi, domain, n_pts_dir, quad_wrapper);

    // We remove generated points on facets.
    for (int local_facet_id = 0; local_facet_id < n_local_facets; ++local_facet_id) {
      if (unf_domain.has_unfitted_boundary(cell_id, local_facet_id)) {
        purge_facet_points(*phi, unf_domain, cell_id, local_facet_id, quad);
      }
    }
  } else {
    // it's full with unfitted boundaries.
    n_pts_per_cell.push_back(0);
  }

  // We add points on unfitted boundaries that belong to facets.
  if (include_facet_unf_bdry) {
    for (int local_facet_id = 0; local_facet_id < n_local_facets; ++local_facet_id) {
      if (unf_domain.has_unfitted_boundary(cell_id, local_facet_id)) {
        if (!exclude_ext_bdry || !unf_domain.is_exterior_facet(cell_id, local_facet_id)) {
          add_unf_bdry_quad(unf_domain, cell_id, local_facet_id, n_pts_dir, quad);
        }
      }
    }
  }
}

// NOLINTBEGIN (bugprone-easily-swappable-parameters)
template<int dim>
void create_facet_quadrature(const UnfittedImplDomain<dim> &unf_domain,
  const std::int64_t cell_id,
  const int local_facet_id,
  const int n_pts_dir,
  const bool remove_unf_bdry,
  const bool remove_cut,
  CutIsoBoundsQuad<dim - 1> &quad)
// NOLINTEND (bugprone-easily-swappable-parameters)
{
  static_assert(dim == 2 || dim == 3, "Invalid dimension.");

  assert(0 <= cell_id);
  assert(0 <= local_facet_id && local_facet_id < dim * 2);
  assert(0 < n_pts_dir);

  auto &points = quad.points;
  auto &weights = quad.weights;
  auto &n_pts_per_facet = quad.n_pts_per_facet;

  const bool is_full_unf = unf_domain.is_full_unfitted_facet(cell_id, local_facet_id);
  if (is_full_unf || unf_domain.is_full_facet(cell_id, local_facet_id)) {

    if (is_full_unf && remove_unf_bdry) {
      n_pts_per_facet.push_back(0);
      return;
    }

    const int n_pts_per_quad_set = n_pts_dir * (dim == 3 ? n_pts_dir : 1);

    static auto gauss_01 = Quadrature<dim - 1>::create_Gauss_01(n_pts_dir);
    if (gauss_01->get_num_points() != static_cast<std::size_t>(n_pts_per_quad_set)) {
      gauss_01 = Quadrature<dim - 1>::create_Gauss_01(n_pts_dir);
    }

    const auto &gauss_pt = gauss_01->points();
    points.insert(points.end(), gauss_pt.cbegin(), gauss_pt.cend());

    const auto &gauss_wg = gauss_01->weights();
    weights.insert(weights.end(), gauss_wg.cbegin(), gauss_wg.cend());

    n_pts_per_facet.push_back(n_pts_per_quad_set);
  } else if (unf_domain.is_cut_facet(cell_id, local_facet_id)
             || unf_domain.has_unfitted_boundary(cell_id, local_facet_id)) {

    const auto phi = unf_domain.get_impl_func();
    compute_facet_quadrature_with_algoim<dim>(*phi, unf_domain, cell_id, local_facet_id, n_pts_dir, quad);

    constexpr bool remove_unf_ext_bdry{ true };
    purge_facet_points(
      *phi, unf_domain, cell_id, local_facet_id, remove_unf_bdry, remove_unf_ext_bdry, remove_cut, quad);
  } else {// if (unf_domain.is_empty_facet(cell_id, local_facet_id)) {
    n_pts_per_facet.push_back(0);
  }
}


// Instantiations

template void
  create_cell_quadrature<2>(const UnfittedImplDomain<2> &, const std::int64_t, const int, CutCellsQuad<2> &);
template void
  create_cell_quadrature<3>(const UnfittedImplDomain<3> &, const std::int64_t, const int, CutCellsQuad<3> &);

template void create_cell_unfitted_bound_quadrature<2>(const UnfittedImplDomain<2> &unf_domain,
  const std::int64_t,
  const int,
  const bool,
  const bool,
  CutUnfBoundsQuad<2> &);
template void create_cell_unfitted_bound_quadrature<3>(const UnfittedImplDomain<3> &unf_domain,
  const std::int64_t,
  const int,
  const bool,
  const bool,
  CutUnfBoundsQuad<3> &);

template void create_facet_quadrature<2>(const UnfittedImplDomain<2> &,
  const std::int64_t,
  const int,
  const int,
  const bool,
  const bool,
  CutIsoBoundsQuad<1> &);
template void create_facet_quadrature<3>(const UnfittedImplDomain<3> &,
  const std::int64_t,
  const int,
  const int,
  const bool,
  const bool,
  CutIsoBoundsQuad<2> &);

}// namespace qugar::impl