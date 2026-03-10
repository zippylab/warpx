from __future__ import annotations

import math

from .blocks import (
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

    _check_const_dt(r, spec)
    _check_b0(r, spec)
    _check_beam_drift(r, spec)
    _check_whistler_cfl(r, spec)

    return r


def _check_const_dt(r: ValidationReport, spec: IonBeamInstabilitySpec) -> None:
    if spec.const_dt <= 0:
        r.add(Severity.ERROR, "beam.const_dt",
              "const_dt must be > 0", const_dt=spec.const_dt)


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


def _check_whistler_cfl(r: ValidationReport, spec: IonBeamInstabilitySpec) -> None:
    """Reuse the whistler CFL check from hybrid_plasma for the core species."""
    _EPS0 = 8.854187817e-12
    _C    = 299792458.0
    _Q_E  = 1.602176634e-19
    _M_P  = 1.67262192369e-27
    _RK4_LIMIT = 2.0 * math.sqrt(2.0)

    B = math.sqrt(sum(b ** 2 for b in spec.B0))
    if B == 0.0:
        return

    m_i = spec.core.mass_amu * _M_P
    n = spec.ohm.n0_ref
    omega_ci = _Q_E * B / m_i
    omega_pi = math.sqrt(n * _Q_E ** 2 / (m_i * _EPS0))
    l_i = _C / omega_pi

    dx = min(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )

    dt_sub = spec.const_dt / spec.ohm.substeps
    k_nyq_li = math.pi * l_i / dx
    z_max = k_nyq_li ** 2 * omega_ci * dt_sub

    if z_max > _RK4_LIMIT:
        substeps_min = int(
            math.ceil(spec.const_dt * omega_ci * k_nyq_li ** 2 / _RK4_LIMIT)
        )
        r.add(
            Severity.ERROR,
            "beam.cfl.whistler",
            (
                f"Whistler CFL z={z_max:.3g} > {_RK4_LIMIT:.3g}: "
                f"sub-inertial Nyquist modes are numerically unstable. "
                f"l_i={l_i:.3e} m; dx={dx:.3e} m; dx/l_i={dx/l_i:.4f}. "
                f"Set dx <= l_i/10 = {l_i/10:.3e} m or increase substeps>={substeps_min}."
            ),
            z_whistler=round(z_max, 3),
            substeps_min=substeps_min,
            l_i=round(l_i, 6),
            dx=round(dx, 8),
        )
