from __future__ import annotations

import math

from .blocks import (
    validate_amr,
    validate_diag,
    validate_domain,
    validate_particle_beam,
    validate_solver,
    validate_species,
)
from .pwfa import PWFASpec
from .spec import Severity, ValidationReport


def validate_pwfa_spec(spec: PWFASpec) -> ValidationReport:
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_solver(spec.solver))
    r.merge(validate_species(spec.plasma, spec.domain))
    r.merge(validate_particle_beam(spec.driver, label="driver"))
    if spec.witness is not None:
        r.merge(validate_particle_beam(spec.witness, label="witness"))
    r.merge(validate_diag(spec.diag))

    _check_relativistic(r, spec)
    _check_debye_resolution(r, spec)

    return r


def _check_relativistic(r: ValidationReport, spec: PWFASpec) -> None:
    """Warn if the driver beam is non-relativistic (uz_m < 10)."""
    if spec.driver.uz_m < 10.0:
        r.add(
            Severity.WARNING,
            "pwfa.driver.uz_m",
            f"driver uz_m = {spec.driver.uz_m:.2f} < 10: driver may not be sufficiently "
            f"relativistic for strong wake excitation.  Typical PWFA drivers have uz_m >> 100.",
            uz_m=spec.driver.uz_m,
        )


def _check_debye_resolution(r: ValidationReport, spec: PWFASpec) -> None:
    """Warn if the grid cannot resolve the plasma skin depth c/ω_pe.

    PWFA simulations require dx ~ c/ω_pe for the wake to be accurately
    captured.  A coarser grid under-resolves the blowout/linear regime.
    """
    _EPS0 = 8.854187817e-12
    _M_E  = 9.1093837015e-31
    _Q_E  = 1.602176634e-19
    _C    = 299792458.0

    n = spec.plasma.plasma_density
    omega_pe = math.sqrt(n * _Q_E ** 2 / (_M_E * _EPS0))
    kp_inv   = _C / omega_pe   # plasma skin depth = c/ω_pe

    dx_max = max(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )

    ratio = dx_max / kp_inv
    if ratio > 0.5:
        r.add(
            Severity.WARNING,
            "pwfa.skin_depth",
            (
                f"Max grid cell dx={dx_max:.3e} m > c/ω_pe/2 = {kp_inv/2:.3e} m "
                f"(dx/(c/ω_pe)={ratio:.2f}). "
                f"PWFA simulations require dx < c/ω_pe ≈ {kp_inv:.3e} m to resolve the wake. "
                f"Increase number_of_cells or reduce domain size."
            ),
            dx_max=round(dx_max, 9),
            skin_depth=round(kp_inv, 9),
            dx_over_kpinv=round(ratio, 4),
        )
