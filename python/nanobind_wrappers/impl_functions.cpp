// --------------------------------------------------------------------------
//
// Copyright (C) 2025-present by Pablo Antolin
//
// This file is part of the QUGaR library.
//
// SPDX-License-Identifier:    MIT
//
// --------------------------------------------------------------------------

//! @file impl_functions.cpp
//! @author Pablo Antolin (pablo.antolin@epfl.ch)
//! @brief Implementation of nanobind wrappers for implicit functions
//! library.
//! @date 2025-01-21
//!
//! @copyright Copyright (c) 2025-present

#include "wrapper_utils.hpp"

#include <qugar/affine_transf.hpp>
#include <qugar/bezier_tp.hpp>
#include <qugar/domain_function.hpp>
#include <qugar/impl_funcs_lib.hpp>
#include <qugar/point.hpp>
#include <qugar/primitive_funcs_lib.hpp>
#include <qugar/ref_system.hpp>
#include <qugar/tpms_lib.hpp>
#include <qugar/types.hpp>
#include <qugar/utils.hpp>


#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// NOLINTBEGIN (misc-include-cleaner)
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
// NOLINTEND (misc-include-cleaner)

#include <memory>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace qugar::wrappers {

namespace {


  template<int dim> void declare_base_implicit(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;

    const std::string pyclass_name{ std::string("ImplicitFunc_") + std::to_string(dim) + "D" };
    nb::class_<ImplicitFunc>(module, pyclass_name.c_str(), "Implicit function")
      .def_prop_ro(
        "dim", [](ImplicitFunc & /*func*/) { return dim; }, nb::rv_policy::reference_internal)
      .def(
        "eval",
        [](ImplicitFunc &func, const npPointConstArray<dim> &points) {
          const auto n_points = points.shape(0);
          std::vector<real> values(n_points);

          Point<dim> point;
          for (std::size_t i = 0; i < n_points; ++i) {

            for (int dir = 0; dir < dim; ++dir) {
              point(dir) = points(i, dir);
            }
            at(values, i) = func(point);
          }

          return as_nbarray(values, { n_points });
        },
        nb::arg("points"))
      .def(
        "grad",
        [](ImplicitFunc &func, const npPointConstArray<dim> &points) {
          const auto n_points = points.shape(0);

          std::vector<real> grads(n_points * dim);

          Point<dim> point;
          auto g = grads.begin();
          for (std::size_t i = 0; i < n_points; ++i) {

            for (int dir = 0; dir < dim; ++dir) {
              point(dir) = points(i, dir);
            }
            const auto grad = func.grad(point);

            for (int dir = 0; dir < dim; ++dir) {
              *g++ = grad(dir);
            }
          }

          return as_nbarray(grads, { n_points, dim });
        },
        nb::arg("points"));
  }

  void declare_sphere(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<3>;
    using SphereBzr = impl::funcs::SphereBzr<3>;
    using Sphere = impl::funcs::Sphere<3>;

    nb::class_<SphereBzr, ImplicitFunc>(module, "SphereBzr", "Sphere function (Bezier version)")
      .def_prop_ro(
        "radius", [](SphereBzr &sphere) { return sphere.radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](SphereBzr &sphere) { return npPointConst<3>(sphere.center().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    nb::class_<Sphere, ImplicitFunc>(module, "Sphere", "Sphere function (non Bezier version)")
      .def_prop_ro(
        "radius", [](Sphere &sphere) { return sphere.radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](Sphere &sphere) { return npPointConst<3>(sphere.center().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    module.def(
      "create_sphere",
      [](const real radius, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<SphereBzr>(radius);
        } else {
          return std::make_shared<Sphere>(radius);
        }
      },
      nb::arg("radius"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_sphere",
      [](const real radius, const npPointConst<3> &center, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<SphereBzr>(radius, transform_point<3>(center));
        } else {
          return std::make_shared<Sphere>(radius, transform_point<3>(center));
        }
      },
      nb::arg("radius"),
      nb::arg("center"),
      nb::arg("use_bzr") = true);
  }


  void declare_disk(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<2>;
    using Disk = impl::funcs::Sphere<2>;
    using DiskBzr = impl::funcs::SphereBzr<2>;

    nb::class_<DiskBzr, ImplicitFunc>(module, "DiskBzr", "Disk function (Bezier version)")
      .def_prop_ro(
        "radius", [](DiskBzr &disk) { return disk.radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](DiskBzr &disk) { return npPointConst<2>(disk.center().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    nb::class_<Disk, ImplicitFunc>(module, "Disk", "Disk function (non-Bezier version)")
      .def_prop_ro(
        "radius", [](Disk &disk) { return disk.radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](Disk &disk) { return npPointConst<2>(disk.center().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    module.def(
      "create_disk",
      [](const real radius, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<DiskBzr>(radius);
        } else {
          return std::make_shared<Disk>(radius);
        }
      },
      nb::arg("radius"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_disk",
      [](const real radius, const npPointConst<2> &center, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<DiskBzr>(radius, transform_point<2>(center));
        } else {
          return std::make_shared<Disk>(radius, transform_point<2>(center));
        }
      },
      nb::arg("radius"),
      nb::arg("center"),
      nb::arg("use_bzr") = true);
  }

  void declare_ellipsoid(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<3>;
    using EllipsoidBzr = impl::funcs::EllipsoidBzr<3>;
    using Ellipsoid = impl::funcs::Ellipsoid<3>;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<EllipsoidBzr, ImplicitFunc>(module, "EllipsoidBzr", "Ellipsoid function (Bezier version)")
      .def_prop_ro(
        "semi_axes",
        [](EllipsoidBzr &ellipsoid) { return npPointConst<3>(ellipsoid.semi_axes().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "ref_system",
        [](EllipsoidBzr &ellipsoid) { return ellipsoid.ref_system(); },
        nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Ellipsoid, ImplicitFunc>(module, "Ellipsoid", "Ellipsoid function (non-Bezier version)")
      .def_prop_ro(
        "semi_axes",
        [](Ellipsoid &ellipsoid) { return npPointConst<3>(ellipsoid.semi_axes().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "ref_system", [](Ellipsoid &ellipsoid) { return ellipsoid.ref_system(); }, nb::rv_policy::reference_internal);

    module.def(
      "create_ellipsoid",
      [](const npPointConst<3> &semi_axes, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<EllipsoidBzr>(transform_point<3>(semi_axes));
        } else {
          return std::make_shared<Ellipsoid>(transform_point<3>(semi_axes));
        }
      },
      nb::arg("semi_axes"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_ellipsoid",
      [](const npPointConst<3> &semi_axes,
        const impl::RefSystem<3> &system,
        const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<EllipsoidBzr>(transform_point<3>(semi_axes), system);
        } else {
          return std::make_shared<Ellipsoid>(transform_point<3>(semi_axes), system);
        }
      },
      nb::arg("semi_axes"),
      nb::arg("ref_system"),
      nb::arg("use_bzr") = true);
  }

  void declare_ellipse(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<2>;
    using EllipseBzr = impl::funcs::EllipsoidBzr<2>;
    using Ellipse = impl::funcs::Ellipsoid<2>;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<EllipseBzr, ImplicitFunc>(module, "EllipseBzr", "Ellipse function (Bezier version)")
      .def_prop_ro(
        "semi_axes",
        [](EllipseBzr &ellipse) { return npPointConst<2>(ellipse.semi_axes().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "ref_system", [](EllipseBzr &ellipse) { return ellipse.ref_system(); }, nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Ellipse, ImplicitFunc>(module, "Ellipse", "Ellipse function (non-Bezier version)")
      .def_prop_ro(
        "semi_axes",
        [](Ellipse &ellipse) { return npPointConst<2>(ellipse.semi_axes().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "ref_system", [](Ellipse &ellipse) { return ellipse.ref_system(); }, nb::rv_policy::reference_internal);

    module.def(
      "create_ellipse",
      [](const npPointConst<2> &semi_axes, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<EllipseBzr>(transform_point<2>(semi_axes));
        } else {
          return std::make_shared<Ellipse>(transform_point<2>(semi_axes));
        }
      },
      nb::arg("semi_axes"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_ellipse",
      [](const npPointConst<2> &semi_axes,
        const impl::RefSystem<2> &system,
        const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<EllipseBzr>(transform_point<2>(semi_axes), system);
        } else {
          return std::make_shared<Ellipse>(transform_point<2>(semi_axes), system);
        }
      },
      nb::arg("semi_axes"),
      nb::arg("ref_system"),
      nb::arg("use_bzr") = true);
  }

  void declare_cylinder(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<3>;
    using CylinderBzr = impl::funcs::CylinderBzr;
    using Cylinder = impl::funcs::Cylinder;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<CylinderBzr, ImplicitFunc>(module, "CylinderBzr", "Cylinder function (Bezier version)")
      .def_prop_ro(
        "radius", [](CylinderBzr &cylinder) { return cylinder.radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "origin",
        [](CylinderBzr &cylinder) { return npPointConst<3>(cylinder.origin().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "axis",
        [](CylinderBzr &cylinder) { return npPointConst<3>(cylinder.axis().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Cylinder, ImplicitFunc>(module, "Cylinder", "Cylinder function (non-Bezier version)")
      .def_prop_ro(
        "radius", [](Cylinder &cylinder) { return cylinder.radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "origin",
        [](Cylinder &cylinder) { return npPointConst<3>(cylinder.origin().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "axis",
        [](Cylinder &cylinder) { return npPointConst<3>(cylinder.axis().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    module.def(
      "create_cylinder",
      [](const real radius, const npPointConst<3> &origin, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<CylinderBzr>(radius, transform_point<3>(origin));
        } else {
          return std::make_shared<Cylinder>(radius, transform_point<3>(origin));
        }
      },
      nb::arg("radius"),
      nb::arg("origin"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_cylinder",
      [](const real radius, const npPointConst<3> &origin, const npPointConst<3> &axis, const bool use_bzr)
        -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<CylinderBzr>(radius, transform_point<3>(origin), transform_point<3>(axis));
        } else {
          return std::make_shared<Cylinder>(radius, transform_point<3>(origin), transform_point<3>(axis));
        }
      },
      nb::arg("radius"),
      nb::arg("origin"),
      nb::arg("axis"),
      nb::arg("use_bzr") = true);
  }

  void declare_annulus(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<2>;
    using AnnulusBzr = impl::funcs::AnnulusBzr;
    using Annulus = impl::funcs::Annulus;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<AnnulusBzr, ImplicitFunc>(module, "AnnulusBzr", "Annulus class (Bezier version)")
      .def_prop_ro(
        "inner_radius", [](AnnulusBzr &annulus) { return annulus.inner_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "outer_radius", [](AnnulusBzr &annulus) { return annulus.outer_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](AnnulusBzr &annulus) { return npPointConst<2>(annulus.center().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Annulus, ImplicitFunc>(module, "Annulus", "Annulus class (non-Bezier version)")
      .def_prop_ro(
        "inner_radius", [](Annulus &annulus) { return annulus.inner_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "outer_radius", [](Annulus &annulus) { return annulus.outer_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](Annulus &annulus) { return npPointConst<2>(annulus.center().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    module.def(
      "create_annulus",
      [](const real inner_radius, const real outer_radius, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<AnnulusBzr>(inner_radius, outer_radius);
        } else {
          return std::make_shared<Annulus>(inner_radius, outer_radius);
        }
      },
      nb::arg("inner_radius"),
      nb::arg("outer_radius"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_annulus",
      [](const real inner_radius, const real outer_radius, const npPointConst<2> &center, const bool use_bzr)
        -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<AnnulusBzr>(inner_radius, outer_radius, transform_point<2>(center));
        } else {
          return std::make_shared<Annulus>(inner_radius, outer_radius, transform_point<2>(center));
        }
      },
      nb::arg("inner_radius"),
      nb::arg("outer_radius"),
      nb::arg("center"),
      nb::arg("use_bzr") = true);
  }

  void declare_torus(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<3>;
    using TorusBzr = impl::funcs::TorusBzr;
    using Torus = impl::funcs::Torus;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<TorusBzr, ImplicitFunc>(module, "TorusBzr", "Torus class (Bezier version)")
      .def_prop_ro(
        "major_radius", [](TorusBzr &torus) { return torus.major_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "minor_radius", [](TorusBzr &torus) { return torus.minor_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](TorusBzr &torus) { return npPointConst<3>(torus.center().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "axis",
        [](TorusBzr &torus) { return npPointConst<3>(torus.axis().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Torus, ImplicitFunc>(module, "Torus", "Torus class (non-Bezier version)")
      .def_prop_ro(
        "major_radius", [](Torus &torus) { return torus.major_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "minor_radius", [](Torus &torus) { return torus.minor_radius(); }, nb::rv_policy::reference_internal)
      .def_prop_ro(
        "center",
        [](Torus &torus) { return npPointConst<3>(torus.center().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "axis",
        [](Torus &torus) { return npPointConst<3>(torus.axis().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    module.def(
      "create_torus",
      [](const real major_radius, const real minor_radius, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<TorusBzr>(major_radius, minor_radius);
        } else {
          return std::make_shared<Torus>(major_radius, minor_radius);
        }
      },
      nb::arg("major_radius"),
      nb::arg("minor_radius"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_torus",
      [](const real major_radius, const real minor_radius, const npPointConst<3> &center, const bool use_bzr)
        -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<TorusBzr>(major_radius, minor_radius, transform_point<3>(center));
        } else {
          return std::make_shared<Torus>(major_radius, minor_radius, transform_point<3>(center));
        }
      },
      nb::arg("major_radius"),
      nb::arg("minor_radius"),
      nb::arg("center"),
      nb::arg("use_bzr") = true);

    module.def(
      "create_torus",
      [](const real major_radius,
        const real minor_radius,
        const npPointConst<3> &center,
        const npPointConst<3> &axis,
        const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<TorusBzr>(
            major_radius, minor_radius, transform_point<3>(center), transform_point<3>(axis));
        } else {
          return std::make_shared<Torus>(
            major_radius, minor_radius, transform_point<3>(center), transform_point<3>(axis));
        }
      },
      nb::arg("major_radius"),
      nb::arg("minor_radius"),
      nb::arg("center"),
      nb::arg("axis"),
      nb::arg("use_bzr") = true);
  }

  template<int dim> void declare_square(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using Square = impl::funcs::Square<dim>;

    const std::string pyclass_name{ std::string("Square_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Square, ImplicitFunc>(module, pyclass_name.c_str(), "Square function");

    const std::string func_name = { std::string("create_square_") + std::to_string(dim) + "D" };

    module.def(func_name.c_str(), []() { return std::make_shared<Square>(); });

    module.def(
      "create_square",
      [](const impl::AffineTransf<dim> &transf) { return std::make_shared<Square>(transf); },
      nb::arg("affine_transf"));
  }

  void declare_plane(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<3>;
    using PlaneBzr = impl::funcs::PlaneBzr<3>;
    using Plane = impl::funcs::Plane<3>;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<PlaneBzr, ImplicitFunc>(module, "PlaneBzr", "Plane function (Bezier version)")
      .def_prop_ro(
        "origin",
        [](PlaneBzr &plane) { return npPointConst<3>(plane.origin().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "normal",
        [](PlaneBzr &plane) { return npPointConst<3>(plane.normal().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Plane, ImplicitFunc>(module, "Plane", "Plane function (non-Bezier version)")
      .def_prop_ro(
        "origin",
        [](Plane &plane) { return npPointConst<3>(plane.origin().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "normal",
        [](Plane &plane) { return npPointConst<3>(plane.normal().data(), { 3 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    module.def(
      "create_plane",
      [](const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<PlaneBzr>();
        } else {
          return std::make_shared<Plane>();
        }
      },
      nb::arg("use_bzr") = true);

    module.def(
      "create_plane",
      [](const npPointConst<3> &origin,
        const npPointConst<3> &normal,
        const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<PlaneBzr>(transform_point<3>(origin), transform_point<3>(normal));
        } else {
          return std::make_shared<Plane>(transform_point<3>(origin), transform_point<3>(normal));
        }
      },
      nb::arg("origin"),
      nb::arg("normal"),
      nb::arg("use_bzr") = true);
  }


  void declare_line(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<2>;
    using LineBzr = impl::funcs::PlaneBzr<2>;
    using Line = impl::funcs::Plane<2>;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<LineBzr, ImplicitFunc>(module, "LineBzr", "Line function (Bezier version)")
      .def_prop_ro(
        "origin",
        [](LineBzr &line) { return npPointConst<2>(line.origin().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "normal",
        [](LineBzr &line) { return npPointConst<2>(line.normal().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Line, ImplicitFunc>(module, "Line", "Line function (non-Bezier version)")
      .def_prop_ro(
        "origin",
        [](Line &line) { return npPointConst<2>(line.origin().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal)
      .def_prop_ro(
        "normal",
        [](Line &line) { return npPointConst<2>(line.normal().data(), { 2 }, nb::handle()); },
        nb::rv_policy::reference_internal);

    module.def(
      "create_line",
      [](const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<LineBzr>();
        } else {
          return std::make_shared<Line>();
        }
      },
      nb::arg("use_bzr") = true);

    module.def(
      "create_line",
      [](const npPointConst<2> &origin,
        const npPointConst<2> &normal,
        const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<LineBzr>(transform_point<2>(origin), transform_point<2>(normal));
        } else {
          return std::make_shared<Line>(transform_point<2>(origin), transform_point<2>(normal));
        }
      },
      nb::arg("origin"),
      nb::arg("normal"),
      nb::arg("use_bzr") = true);
  }

  template<int dim> void declare_constant(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using Constant = impl::funcs::Constant<dim>;
    using ConstantBzr = impl::funcs::ConstantBzr<dim>;

    const std::string pyclass_name_bzr{ std::string("ConstantBzr_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<ConstantBzr, ImplicitFunc>(module, pyclass_name_bzr.c_str(), "Constant function (Bezier version)")
      .def_prop_ro("value", [](ConstantBzr &constant) { return constant.value(); }, nb::rv_policy::reference_internal);

    const std::string pyclass_name{ std::string("Constant_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Constant, ImplicitFunc>(module, pyclass_name.c_str(), "Constant function (non-Bezier version)")
      .def_prop_ro("value", [](Constant &constant) { return constant.value(); }, nb::rv_policy::reference_internal);

    const std::string func_name = { std::string("create_constant_") + std::to_string(dim) + "D" };
    module.def(
      func_name.c_str(),
      [](const real value, const bool use_bzr) -> std::shared_ptr<ImplicitFunc> {
        if (use_bzr) {
          return std::make_shared<ConstantBzr>(value);
        } else {
          return std::make_shared<Constant>(value);
        }
      },
      nb::arg("value"),
      nb::arg("use_bzr") = true);
  }

  template<int dim> void declare_dim_linear(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using DimLinear = impl::funcs::DimLinear<dim>;
    static const int num_coeffs = DimLinear::num_coeffs;
    using CoefsArray = nb::ndarray<real, nb::shape<num_coeffs>>;

    const std::string pyclass_name{ std::string("DimLinear_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<DimLinear, ImplicitFunc>(module, pyclass_name.c_str(), "Dim-linear function");

    module.def(
      "create_dim_linear",
      [](const std::array<real, num_coeffs> &coeffs) { return std::make_shared<DimLinear>(coeffs); },
      nb::arg("coefficients"));

    module.def(
      "create_dim_linear",
      [](const CoefsArray &coeffs_py) {
        std::array<real, num_coeffs> coeffs{};
        for (int i = 0; i < num_coeffs; ++i) {
          at(coeffs, i) = coeffs_py(i);
        }

        return std::make_shared<DimLinear>(coeffs);
      },
      nb::arg("coefficients"));

    module.def(
      "create_dim_linear",
      [](const std::array<real, num_coeffs> &coeffs, const impl::AffineTransf<dim> &transf) {
        return std::make_shared<DimLinear>(coeffs, transf);
      },
      nb::arg("coefficients"),
      nb::arg("affine_transf"));

    module.def(
      "create_dim_linear",
      [](const CoefsArray &coeffs_py, const impl::AffineTransf<dim> &transf) {
        std::array<real, num_coeffs> coeffs{};
        for (int i = 0; i < num_coeffs; ++i) {
          at(coeffs, i) = coeffs_py(i);
        }

        return std::make_shared<DimLinear>(coeffs, transf);
      },
      nb::arg("coefficients"),
      nb::arg("affine_transf"));
  }

  template<int dim> void declare_transformed(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using ImplicitFuncPtr = std::shared_ptr<const ImplicitFunc>;
    using Transformed = impl::funcs::TransformedFunction<dim>;

    const std::string pyclass_name{ std::string("TransformedFunction_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Transformed, ImplicitFunc>(module, pyclass_name.c_str(), "Affinely transformed function");

    module.def(
      "create_affinely_transformed",
      [](const ImplicitFuncPtr &func, const impl::AffineTransf<dim> &transf) {
        return std::make_shared<Transformed>(func, transf);
      },
      nb::arg("base_func"),
      nb::arg("affine_transf"));
  }

  template<int dim> void declare_negative(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using ImplicitFuncPtr = std::shared_ptr<const ImplicitFunc>;
    using Negative = impl::funcs::Negative<dim>;

    const std::string pyclass_name{ std::string("Negative_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<Negative, ImplicitFunc>(module, pyclass_name.c_str(), "Negative function");

    module.def(
      "create_negative",
      [](const ImplicitFuncPtr &func) -> ImplicitFuncPtr {
        if (impl::is_bezier<dim, 1>(*func)) {
          const auto &bezier = dynamic_cast<const impl::BezierTP<dim, 1> &>(*func);
          return bezier.negate();
        } else {
          return std::make_shared<Negative>(func);
        }
      },

      nb::arg("impl_func"));
  }

  template<int dim> void declare_add_functions(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using ImplicitFuncPtr = std::shared_ptr<const ImplicitFunc>;
    using AddFunctions = impl::funcs::AddFunctions<dim>;

    const std::string pyclass_name{ std::string("AddFunctions_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<AddFunctions, ImplicitFunc>(module, pyclass_name.c_str(), "Add functions");

    module.def(
      "create_functions_addition",
      [](const ImplicitFuncPtr &lhs_func, const ImplicitFuncPtr &rhs_func) -> ImplicitFuncPtr {
        if (impl::is_bezier<dim, 1>(*lhs_func) && impl::is_bezier<dim, 1>(*rhs_func)) {
          const auto &lhs_bezier = dynamic_cast<const impl::BezierTP<dim, 1> &>(*lhs_func);
          const auto &rhs_bezier = dynamic_cast<const impl::BezierTP<dim, 1> &>(*rhs_func);
          return lhs_bezier + rhs_bezier;
        } else {
          return std::make_shared<AddFunctions>(lhs_func, rhs_func);
        }
      },
      nb::arg("lhs_func"),
      nb::arg("rhs_func"));
  }

  template<int dim> void declare_subtract_functions(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using ImplicitFuncPtr = std::shared_ptr<const ImplicitFunc>;
    using SubtractFunctions = impl::funcs::SubtractFunctions<dim>;

    const std::string pyclass_name{ std::string("SubtractFunctions_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<SubtractFunctions, ImplicitFunc>(module, pyclass_name.c_str(), "Subtract functions");

    module.def(
      "create_functions_subtraction",
      [](const ImplicitFuncPtr &lhs_func, const ImplicitFuncPtr &rhs_func) -> ImplicitFuncPtr {
        if (impl::is_bezier<dim, 1>(*lhs_func) && impl::is_bezier<dim, 1>(*rhs_func)) {
          const auto &lhs_bezier = dynamic_cast<const impl::BezierTP<dim, 1> &>(*lhs_func);
          const auto &rhs_bezier = dynamic_cast<const impl::BezierTP<dim, 1> &>(*rhs_func);
          return lhs_bezier - rhs_bezier;
        } else {
          return std::make_shared<SubtractFunctions>(lhs_func, rhs_func);
        }
      },
      nb::arg("lhs_func"),
      nb::arg("rhs_func"));
  }

  template<int dim> void declare_beziers_intersection(nanobind::module_ &module)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;
    using ImplicitFuncPtr = std::shared_ptr<const ImplicitFunc>;
    using BeziersIntersection = impl::funcs::BeziersIntersection<dim>;

    const std::string pyclass_name{ std::string("BeziersIntersection_") + std::to_string(dim) + "D" };
    // NOLINTNEXTLINE (bugprone-unused-raii)
    nb::class_<BeziersIntersection, ImplicitFunc>(
      module, pyclass_name.c_str(), "Intersection of the negative regions of several Bezier polynomials");

    module.def(
      "create_functions_intersection",
      [](const std::vector<ImplicitFuncPtr> &funcs) -> ImplicitFuncPtr {
        std::vector<std::shared_ptr<const impl::BezierTP<dim, 1>>> beziers;
        beziers.reserve(funcs.size());
        for (const auto &func : funcs) {
          auto bezier = std::dynamic_pointer_cast<const impl::BezierTP<dim, 1>>(func);
          if (bezier == nullptr) {
            throw std::invalid_argument(
              "create_functions_intersection requires all the functions to be Bezier polynomials.");
          }
          beziers.push_back(std::move(bezier));
        }
        return std::make_shared<BeziersIntersection>(beziers);
      },
      nb::arg("funcs"));
  }

  template<int dim> void declare_general_functions(nanobind::module_ &module)
  {
    declare_base_implicit<dim>(module);
    if constexpr (dim == 2) {
      declare_disk(module);
      declare_annulus(module);
      declare_line(module);
      declare_ellipse(module);
    }
    if constexpr (dim == 3) {
      declare_sphere(module);
      declare_cylinder(module);
      declare_torus(module);
      declare_plane(module);
      declare_ellipsoid(module);
    }
    declare_square<dim>(module);
    declare_constant<dim>(module);
    declare_dim_linear<dim>(module);
    declare_transformed<dim>(module);
    declare_negative<dim>(module);
    declare_add_functions<dim>(module);
    declare_subtract_functions<dim>(module);
    declare_beziers_intersection<dim>(module);
  }

  //  NOLINTBEGIN(bugprone-easily-swappable-parameters)
  template<typename TPMS, int dim>
  void declare_tpms_function_case(nanobind::module_ &module,
    const std::string &name_prefix,
    const std::string &description)
  //  NOLINTEND(bugprone-easily-swappable-parameters)
  {
    using ImplicitFunc = impl::ImplicitFunc<dim>;

    // NOLINTNEXTLINE (bugprone-unused-raii)
    const std::string class_name{ name_prefix + "_" + std::to_string(dim) + "D" };
    nb::class_<TPMS, ImplicitFunc>(module, class_name.c_str(), description.c_str());

    const std::string func_name{ "create_" + name_prefix };

    if constexpr (dim == 2) {
      module.def(
        func_name.c_str(),
        [](const std::array<real, dim> &periods_arr, const real z) {
          Point<dim> periods;
          for (int dir = 0; dir < dim; ++dir) {
            periods(dir) = at(periods_arr, dir);
          }
          return std::make_shared<TPMS>(periods, z);
        },
        nb::arg("periods"),
        nb::arg("z") = numbers::zero);

      module.def(
        func_name.c_str(),
        [](const npPointConst<dim> &periods, const real z) {
          return std::make_shared<TPMS>(transform_point<dim>(periods), z);
        },
        nb::arg("periods"),
        nb::arg("z") = numbers::zero);

    } else {
      module.def(
        func_name.c_str(),
        [](const std::array<real, dim> &periods_arr) {
          Point<dim> periods;
          for (int dir = 0; dir < dim; ++dir) {
            periods(dir) = at(periods_arr, dir);
          }
          return std::make_shared<TPMS>(periods);
        },
        nb::arg("periods"));

      module.def(
        func_name.c_str(),
        [](const npPointConst<dim> &periods) { return std::make_shared<TPMS>(transform_point<dim>(periods)); },
        nb::arg("periods"));
    }
  }

  template<int dim> void declare_tpms_functions(nanobind::module_ &module)
  {
    namespace tpms = impl::tpms;
    declare_tpms_function_case<tpms::Schoen<dim>, dim>(module, "Schoen", "Schoen gyroid TPMS");
    declare_tpms_function_case<tpms::SchoenIWP<dim>, dim>(module, "SchoenIWP", "Schoen I-WP TPMS");
    declare_tpms_function_case<tpms::SchoenFRD<dim>, dim>(module, "SchoenFRD", "Schoen F-RD TPMS");
    declare_tpms_function_case<tpms::FischerKochS<dim>, dim>(module, "FischerKochS", "Fischer-Koch S TPMS");
    declare_tpms_function_case<tpms::SchwarzPrimitive<dim>, dim>(module, "SchwarzPrimitive", "Schwarz primitive TPMS");
    declare_tpms_function_case<tpms::SchwarzDiamond<dim>, dim>(module, "SchwarzDiamond", "Schwarz diamond TPMS");
  }


}// namespace

// NOLINTNEXTLINE (misc-use-internal-linkage)
void impl_functions(nanobind::module_ &module)
{
  declare_general_functions<2>(module);
  declare_general_functions<3>(module);
  declare_tpms_functions<2>(module);
  declare_tpms_functions<3>(module);
}


}// namespace qugar::wrappers
