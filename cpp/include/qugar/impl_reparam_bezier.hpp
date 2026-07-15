// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

#ifndef QUGAR_IMPL_REPARAM_BEZIER_HPP
#define QUGAR_IMPL_REPARAM_BEZIER_HPP

//! @file impl_reparam_bezier.hpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Declaration of tools for creating Bezier reparameterizations.
//!
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include <qugar/bbox.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/impl_reparam_mesh.hpp>

#include <memory>
#include <vector>

namespace qugar::impl {

//! @brief Reparameterizes a Bezier implicit function in the unit hypercube domain.
//!
//! @note The generated reparameterization has a wirebasket associated, but coincident were not merged.
//!
//! @tparam dim Parametric dimension of the function.
//! @tparam S Flag indicating if the reparameterization must
//!         be performed only for the levelset surface (true), i.e.,
//!         the manifold where the Bezier function is equal to 0,
//!         or the volume (false), i.e., the subregion where
//!         the Bezier function is negative.
//! @param bzr Bezier implicit function to reparameterize.
//! @param domain Domain to which the implicit function refers to
//! (even if Beziers are defined in the unit domain).
//! @param order Order of the reparameterization (number of points
//!        per direction in each reparameterization cell).
//! @return Generated reparameterization.
template<int dim, bool S = false>
std::shared_ptr<ImplReparamMesh<S ? dim - 1 : dim, dim>>
  reparam_Bezier(const BezierTP<dim, 1> &bzr, const BoundBox<dim> &domain, int order);

//! @brief Reparameterizes a Bezier implicit function in the unit hypercube domain.
//!
//! @note The generated reparameterization has a wirebasket associated, but coincident were not merged.
//!
//! @tparam dim Parametric dimension of the function.
//! @tparam S Flag indicating if the reparameterization must
//!         be performed only for the levelset surface (true), i.e.,
//!         the manifold where the Bezier function is equal to 0,
//!         or the volume (false), i.e., the subregion where
//!         the Bezier function is negative.
//! @param bzr Bezier implicit function to reparameterize.
//! @param domain Domain to which the implicit function refers to
//! (even if Beziers are defined in the unit domain).
//! @param reparam Reparameterization container to which new generated cells
//! are appended to.
template<int dim, bool S = false>
void reparam_Bezier(const BezierTP<dim, 1> &bzr,
  const BoundBox<dim> &domain,
  ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam);

//! @brief Reparameterizes a domain implicitly defined by the intersection of the negative
//! regions of several Bezier polynomials.
//!
//! The reparameterized domain is the subregion where all the given polynomials are
//! simultaneously negative (for the volume case), or its bounding levelset (for the
//! surface case).
//!
//! @note The generated reparameterization has a wirebasket associated, but coincident were not merged.
//!
//! @tparam dim Parametric dimension of the functions.
//! @tparam S Flag indicating if the reparameterization must be performed only for the
//!         levelset surface (true) or the volume (false).
//! @param bzrs Bezier polynomials (already rescaled to @p domain) whose negative regions
//! are intersected. It must contain at least one polynomial.
//! @param domain Domain to which the implicit functions refer to
//! (even if Beziers are defined in the unit domain).
//! @param reparam Reparameterization container to which new generated cells are appended to.
template<int dim, bool S = false>
void reparam_Beziers(const std::vector<std::shared_ptr<const BezierTP<dim, 1>>> &bzrs,
  const BoundBox<dim> &domain,
  ImplReparamMesh<S ? dim - 1 : dim, dim> &reparam);


}// namespace qugar::impl

#endif// QUGAR_IMPL_REPARAM_BEZIER_HPP
