from __future__ import annotations

import math

from .blocks import (
    check_const_dt_required,
    check_whistler_cfl_hybrid,
    validate_amr,
    validate_diag,
    validate_domain,
    validate_hybrid_ion,
    validate_ohm_solver,
    validate_solver,
)
from .hybrid_plasma import HybridPlasmaSpec
from .spec import Severity, ValidationReport


def validate_hybrid_plasma_spec(spec: HybridPlasmaSpec) -> ValidationReport:
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_solver(spec.solver))
    r.merge(validate_ohm_solver(spec.ohm))
    r.merge(validate_hybrid_ion(spec.ions))
    r.merge(validate_diag(spec.diag))

    check_const_dt_required(spec.const_dt, r, code_prefix="hybrid")
    _check_b0(r, spec)
    _check_density_consistency(r, spec)

    b0_mag = math.sqrt(sum(b**2 for b in spec.B0)) if len(spec.B0) == 3 else 0.0
    dx_min = min(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )
    check_whistler_cfl_hybrid(
        dx_min, b0_mag, spec.ohm.n0_ref, spec.ions.mass_amu,
        spec.const_dt, spec.ohm.substeps, r, code_prefix="hybrid",
    )

    return r


def _check_b0(r: ValidationReport, spec: HybridPlasmaSpec) -> None:
    if len(spec.B0) != 3:
        r.add(Severity.ERROR, "hybrid.B0.len",
              "B0 must be a 3-element list [Bx, By, Bz]", got=len(spec.B0))
        return
    if all(b == 0.0 for b in spec.B0):
        r.add(Severity.WARNING, "hybrid.B0.zero",
              "B0 = [0,0,0]: a zero background field may cause numerical issues "
              "in the Ohm's law solver")


def _check_density_consistency(r: ValidationReport, spec: HybridPlasmaSpec) -> None:
    """Warn if ion density and Ohm solver reference density differ by more than 10×."""
    ratio = spec.ions.density / spec.ohm.n0_ref
    if ratio < 0.1 or ratio > 10.0:
        r.add(
            Severity.WARNING,
            "hybrid.density_mismatch",
            "Ion density and Ohm solver n0_ref differ by more than 10×; "
            "check that n0_ref is representative of the plasma density",
            ion_density=spec.ions.density,
            n0_ref=spec.ohm.n0_ref,
            ratio=ratio,
        )
