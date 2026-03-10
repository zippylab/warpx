from __future__ import annotations

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

    check_const_dt_required(spec.const_dt, r, code_prefix="reconnect")
    _check_bfield(r, spec)
    _check_current_sheet(r, spec)

    dx_min = min(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )
    check_whistler_cfl_hybrid(
        dx_min, spec.B0, spec.ohm.n0_ref, spec.ions.mass_amu,
        spec.const_dt, spec.ohm.substeps, r, code_prefix="reconnect",
    )

    return r


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
