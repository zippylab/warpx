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
from .ion_beam_instability import IonBeamInstabilitySpec
from .spec import Severity, ValidationReport


def validate_ion_beam_instability_spec(spec: IonBeamInstabilitySpec) -> ValidationReport:
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_solver(spec.solver))
    r.merge(validate_ohm_solver(spec.ohm))
    r.merge(validate_hybrid_ion(spec.core))
    r.merge(validate_hybrid_ion(spec.beam))
    r.merge(validate_diag(spec.diag))

    check_const_dt_required(spec.const_dt, r, code_prefix="beam")
    _check_b0(r, spec)
    _check_beam_drift(r, spec)
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
        dx_min, b0_mag, spec.ohm.n0_ref, spec.core.mass_amu,
        spec.const_dt, spec.ohm.substeps, r, code_prefix="beam",
    )

    return r


def _check_b0(r: ValidationReport, spec: IonBeamInstabilitySpec) -> None:
    if len(spec.B0) != 3:
        r.add(Severity.ERROR, "beam.B0.len",
              "B0 must be a 3-element list [Bx, By, Bz]", got=len(spec.B0))
        return
    if all(b == 0.0 for b in spec.B0):
        r.add(Severity.WARNING, "beam.B0.zero",
              "B0 = [0,0,0]: a zero background field may cause numerical issues "
              "in the Ohm's law solver")


def _check_beam_drift(r: ValidationReport, spec: IonBeamInstabilitySpec) -> None:
    """Warn if beam density fraction is very high or drift is very large."""
    n_frac = spec.beam.density / (spec.core.density + spec.beam.density)
    if n_frac > 0.5:
        r.add(Severity.WARNING, "beam.density_fraction",
              "beam density > 50% of total: results may not match thin-beam theory",
              n_beam=spec.beam.density, n_core=spec.core.density, fraction=round(n_frac, 3))

    _EPS0 = 8.854187817e-12
    _M_P  = 1.67262192369e-27
    _MU0  = 4.0 * math.pi * 1e-7
    _Q_E  = 1.602176634e-19

    B = math.sqrt(sum(b ** 2 for b in spec.B0))
    if B > 0:
        m_i = spec.core.mass_amu * _M_P
        n_tot = spec.core.density + spec.beam.density
        vA = B / math.sqrt(_MU0 * n_tot * m_i)
        drift_va = abs(spec.beam_drift_velocity) / vA
        if drift_va < 1.0:
            r.add(Severity.WARNING, "beam.drift_subsonic",
                  f"beam drift ({drift_va:.2f} vA) < 1 vA: R-instability may be weak",
                  drift_va=round(drift_va, 3))


def _check_density_consistency(r: ValidationReport, spec: IonBeamInstabilitySpec) -> None:
    """Warn if total ion density and Ohm solver reference density differ by more than 10×."""
    n_total = spec.core.density + spec.beam.density
    ratio = n_total / spec.ohm.n0_ref
    if ratio < 0.1 or ratio > 10.0:
        r.add(
            Severity.WARNING,
            "beam.density_mismatch",
            "Total ion density (core + beam) and Ohm solver n0_ref differ by more than 10×; "
            "check that n0_ref is representative of the total plasma density",
            n_total=n_total,
            n0_ref=spec.ohm.n0_ref,
            ratio=round(ratio, 3),
        )
