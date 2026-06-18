# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Data structures and functionalities for dealing with integrals."""

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import NamedTuple

import ffcx.analysis
import ffcx.ir.representation
import numpy as np
import numpy.typing as npt
import ufl.domain
import ufl.geometry

from qugar.dolfinx.quadrature_data import QuadratureData, extract_quadrature_data


def _get_integral_subdomain_ids(ir: ffcx.ir.representation.DataIR, name: str) -> list[int]:
    """Extracts the subdomain ids of the integrals in `ir` that have the
    given `name`.

    Args:
        ir (ffcx.ir.representation.DataIR): FFCx Intermediate
            Representation that contains the integrals to be inspected.
        name (str): Integral sought. This string must start with
            ``integral_``.

    Note:
        This function's implementation assumes that ordering the
        subdomain ids stored in the dictionary `form.integral_names`
        (for each form and integral type) matches the ordering of the
        names in `form.integral_names` (for each form and integral
        type).


    Returns:
        list[int]: Subdomain ids of the integrals associated to the
        given `name`.
    """

    for form in ir.forms:
        for itg_type, names in form.integral_names.items():
            all_ids = form.subdomain_ids[itg_type]
            ids = [id for i, id in enumerate(all_ids) if names[i] == name]
            if len(ids) > 0:
                return ids
    return list()


def _get_integral_dimension(
    ir: ffcx.ir.representation.DataIR, itg_ir: ffcx.ir.representation.IntegralIR
) -> int:
    """Gets the topology dimension associated to the given integral
    representation.

    E.g., dimension 2 if the integral is performed inside (or on the
    facet of) triangles or quadrilaterals. Dimension 3 in the case of
    tetrahedra, hexahedra, etc.

    Note:
        This dimension is extracted from the topology dimension of the
        coordinate element associated to the mesh in which the integral
        is computed.

    Arguments:
        ir (ffcx.ir.representation.DataIR): FFCx Intermediate
            Representation that contains the integral `ir_itg`.
        itg_ir (ffcx.ir.representation.IntegralIR): FFCx Intermediate
            Representation of the integral to be modified.

    Returns:
        int: Topology dimension of the integral.
    """

    # Code adapted from function generate_geometry_fables in file
    # ffcx.codegeneration.integral_generator
    cells = set()
    for integrand in itg_ir.expression.integrand.values():
        for attr in integrand["factorization"].nodes.values():
            mt = attr.get("mt")
            if mt is not None:
                if issubclass(type(mt.terminal), ufl.geometry.GeometricQuantity):
                    domain = ufl.domain.extract_unique_domain(mt.terminal)
                    assert domain is not None
                    cells.add(domain.ufl_cell())
    assert len(cells) == 1

    cell = cells.pop()
    return cell.topological_dimension


class IntegralData(NamedTuple):
    """Class for storing inmutable infomation associated to an integral.

    Parameters:
        name (str): Integral name containing the prefix ``integral_``.
        subdomain_ids (list[int]): Ids of the integral's subdomain.
        integral_type (str): Integral type.
        tdim (int): Topology dimension associated to the given
            integral's domain. E.g., dimension 2 if the integral is
            performed inside (or on the facet of) triangles or
            quadrilaterals. Dimension 3 in the case of tetrahedra,
            hexahedra, etc. domain.
        is_mixed_dim (bool): Flag indicating whether the integral
            presents mixed dimensions among elements and integration
            domain.
        quad_data_and_FE_tables (dict[QuadratureData, list[FETable]]):
            Dictionary mapping quadrature data to all the FE tables
            that share the same quadrature. The ordering of the tables
            for each quadrature stays constant between successive Python
            calls, what allows to reuse previously cached compiled
            forms.
    """

    name: str
    subdomain_ids: list[int]
    integral_type: str
    tdim: int
    is_mixed_dim: bool
    quad_data_FE_tables: dict[QuadratureData, list]

    @property
    def dtype(self) -> type[np.float32 | np.float64]:
        """`numpy` type used in the integral."""
        FE_tables = next(iter(self.quad_data_FE_tables.values()))
        assert FE_tables
        return FE_tables[0].dtype

def extract_integral_data_from_ir(
    ufl_analysis: ffcx.analysis.UFLData,
    ir: ffcx.ir.representation.DataIR,
    itg_ir: ffcx.ir.representation.IntegralIR,
    ffcx_options: dict[str, int | float | npt.DTypeLike],
) -> IntegralData:
    """Like :func:`extract_integral_data` but builds the FE tables from the IR
    (:func:`qugar.dolfinx.ffcx_backend.extract_ir_tables`) instead of parsing
    the rendered C, so it needs no integral implementation string. Used by the
    FFCx language-backend path. The quadrature data still comes from the UFL
    analysis (the per-cell packer needs the real quadrature ``degree`` for
    unfitted-boundary rules, which the IR does not retain)."""
    from ffcx.codegeneration.utils import dtype_to_scalar_dtype

    from qugar.dolfinx.ffcx_backend import extract_ir_tables

    itg_name = itg_ir.expression.name
    tdim = _get_integral_dimension(ir, itg_ir)
    itg_ids = _get_integral_subdomain_ids(ir, itg_name)
    assert len(itg_ids) > 0

    short_itg_name = itg_name[len("integral_") :]
    all_quads_data = extract_quadrature_data(ufl_analysis, ffcx_options)
    real_dtype = np.dtype(dtype_to_scalar_dtype(ffcx_options["scalar_type"]))  # type: ignore[arg-type]
    FE_tables = extract_ir_tables(itg_ir, all_quads_data, real_dtype)

    is_mixed_dim = False
    quad_FE_tables: dict[QuadratureData, list] = {}
    for FE_table in FE_tables:
        quad_data = all_quads_data[FE_table.quad_name]
        quad_FE_tables.setdefault(quad_data, []).append(FE_table)
        if FE_table.element_dim != tdim:
            is_mixed_dim = True

    return IntegralData(
        short_itg_name,
        itg_ids,
        itg_ir.expression.integral_type,
        tdim,
        is_mixed_dim,
        quad_FE_tables,
    )
