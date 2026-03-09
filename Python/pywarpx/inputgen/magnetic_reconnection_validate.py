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
from .magnetic_reconnection import MagneticReconnectionSpec
from .spec import Severity, ValidationReport


def validate_magnetic_reconnection_spec(spec: MagneticReconnectionSpec) -> ValidationReport:
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_solver(spec.solver))
    r.merge(validate_ohm_solver(spec.ohm))
    r.merge(validate_hybrid_ion(spec.ions))
    r.merge(validate_diag(spec.diag))

    _check_const_dt(r, spec)
    _check_bfield(r, spec)
    _check_current_sheet(r, spec)
    _check_whistler_cfl(r, spec)

    return r


def _check_const_dt(r: ValidationReport, spec: MagneticReconnectionSpec) -> None:
    if spec.const_dt <= 0:
        r.add(Severity.ERROR, "reconnect.const_dt",
              "const_dt must be > 0", const_dt=spec.const_dt)


def _check_bfield(r: ValidationReport, spec: MagneticReconnectionSpec) -> None:
    if spec.B0 <= 0:
        r.add(Severity.ERROR, "reconnect.B0",
              "B0 must be > 0", B0=spec.B0)
    if spec.Bg < 0:
        r.add(Severity.ERROR, "reconnect.Bg",
              "Guide field Bg must be >= 0", Bg=spec.Bg)
    if spec.delta <= 0:
        r.add(Severity.ERROR, "reconnect.delta",
              "Current sheet half-width delta must be > 0", delta=spec.delta)
    if not (0.0 <= spec.dB_fraction <= 1.0):
        r.add(Severity.WARNING, "reconnect.dB_fraction",
              "dB_fraction should be in [0, 1] (fraction of B0)",
              dB_fraction=spec.dB_fraction)


def _check_current_sheet(r: ValidationReport, spec: MagneticReconnectionSpec) -> None:
    """Warn if the current sheet is unresolved or doesn't fit in the domain."""
    Lz = spec.domain.upper_bound[-1] - spec.domain.lower_bound[-1]
    nz = spec.domain.number_of_cells[-1]
    dz = Lz / nz

    if spec.delta < dz:
        r.add(
            Severity.WARNING,
            "reconnect.delta_unresolved",
            f"Current sheet half-width delta={spec.delta:.3e} m < dz={dz:.3e} m; "
            f"increase number_of_cells[-1] or increase delta to at least {dz:.3e} m.",
            delta=spec.delta,
            dz=round(dz, 9),
        )

    # Current sheet should be much smaller than the domain
    if spec.delta > Lz / 4.0:
        r.add(
            Severity.WARNING,
            "reconnect.delta_too_large",
            f"Current sheet half-width delta={spec.delta:.3e} m is > Lz/4={Lz/4:.3e} m; "
            f"the sheet may not fit properly in the domain.",
            delta=spec.delta,
            Lz=round(Lz, 6),
        )


def _check_whistler_cfl(r: ValidationReport, spec: MagneticReconnectionSpec) -> None:
    """Whistler CFL check using B0 as the reference field."""
    _EPS0 = 8.854187817e-12
    _C    = 299792458.0
    _Q_E  = 1.602176634e-19
    _M_P  = 1.67262192369e-27
    _RK4_LIMIT = 2.0 * math.sqrt(2.0)

    B = spec.B0
    if B == 0.0:
        return

    m_i = spec.ions.mass_amu * _M_P
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
            Severity.WARNING,
            "reconnect.cfl.whistler",
            (
                f"Whistler CFL z={z_max:.3g} > {_RK4_LIMIT:.3g}: unstable. "
                f"l_i={l_i:.3e} m; dx={dx:.3e} m (dx/l_i={dx/l_i:.4f}). "
                f"Recommend dx <= l_i/10 = {l_i/10:.3e} m or substeps>={substeps_min}."
            ),
            z_whistler=round(z_max, 3),
            substeps_min=substeps_min,
            l_i=round(l_i, 6),
            dx=round(dx, 8),
        )
