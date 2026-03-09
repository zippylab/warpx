from __future__ import annotations

import math

from .blocks import validate_amr, validate_diag, validate_domain, validate_eb, validate_solver
from .electrostatic_plasma import ElectrostaticPlasmaSpec
from .spec import Severity, ValidationReport

# CODATA 2018 values
_EPS0 = 8.854187817e-12
_M_E  = 9.1093837015e-31
_Q_E  = 1.602176634e-19
_C    = 299792458.0


def validate_electrostatic_plasma_spec(spec: ElectrostaticPlasmaSpec) -> ValidationReport:
    """Validate an ElectrostaticPlasmaSpec.

    Checks (in order):
    1. Domain structure (dim in {1,2,3}, list lengths, bounds)
    2. Solver parameters (max_steps > 0)
    3. Diagnostics
    4. ES-specific scalar parameters (n0, Te, Ti, const_dt, …)
    5. Debye length resolution: dx < λ_De
    6. Electron plasma frequency stability: dt * ω_pe < 2
    """
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_solver(spec.solver))
    r.merge(validate_diag(spec.diag))
    r.merge(validate_eb(spec.eb))

    _check_scalars(r, spec)
    if not r.ok:
        return r

    _check_debye_resolution(r, spec)
    _check_plasma_frequency(r, spec)

    return r


def _check_scalars(r: ValidationReport, spec: ElectrostaticPlasmaSpec) -> None:
    if spec.const_dt <= 0:
        r.add(Severity.ERROR, "es.const_dt",
              "const_dt must be > 0 (required by the electrostatic solver)",
              const_dt=spec.const_dt)
    if spec.n0 <= 0:
        r.add(Severity.ERROR, "es.n0",
              "n0 must be > 0", n0=spec.n0)
    if spec.Te <= 0:
        r.add(Severity.ERROR, "es.Te",
              "Electron temperature Te must be > 0 eV", Te=spec.Te)
    if spec.Ti < 0:
        r.add(Severity.ERROR, "es.Ti",
              "Ion temperature Ti must be >= 0 eV", Ti=spec.Ti)
    if spec.ion_mass_amu <= 0:
        r.add(Severity.ERROR, "es.ion_mass_amu",
              "ion_mass_amu must be > 0", ion_mass_amu=spec.ion_mass_amu)
    if spec.ppc <= 0:
        r.add(Severity.ERROR, "es.ppc",
              "ppc must be > 0", ppc=spec.ppc)
    if spec.electrostatic_solver not in ("labframe", "relativistic"):
        r.add(Severity.ERROR, "es.solver_type",
              "electrostatic_solver must be 'labframe' or 'relativistic'",
              electrostatic_solver=spec.electrostatic_solver)
    if spec.poisson_precision <= 0:
        r.add(Severity.ERROR, "es.poisson_precision",
              "poisson_precision must be > 0", poisson_precision=spec.poisson_precision)


def _check_debye_resolution(r: ValidationReport, spec: ElectrostaticPlasmaSpec) -> None:
    """Warn if the largest grid cell exceeds the electron Debye length.

    Electrostatic simulations must resolve λ_De to correctly capture
    space-charge shielding and plasma oscillations. Aliasing occurs for dx > λ_De.

    Debye length: λ_De = sqrt(ε₀ · Te_eV / (n₀ · q_e))
    """
    # λ_De = sqrt(ε₀ * Te [J] / (n0 * q_e²)) = sqrt(ε₀ * Te_eV / (n0 * q_e))
    lam_De = math.sqrt(_EPS0 * spec.Te / (spec.n0 * _Q_E))

    dx_max = max(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )

    ratio = dx_max / lam_De
    if ratio > 1.0:
        r.add(
            Severity.WARNING,
            "es.debye_resolution",
            (
                f"Max grid cell dx={dx_max:.3e} m > Debye length λ_De={lam_De:.3e} m "
                f"(dx/λ_De={ratio:.2f}). Electrostatic simulations require dx < λ_De "
                f"to resolve space-charge shielding. "
                f"Increase number_of_cells or reduce domain size so that dx < {lam_De:.3e} m, "
                f"or increase n0 above {_EPS0 * spec.Te / (dx_max**2 * _Q_E):.2e} m^-3."
            ),
            dx_max=round(dx_max, 9),
            lambda_De=round(lam_De, 9),
            dx_over_lambda_De=round(ratio, 4),
        )


def _check_plasma_frequency(r: ValidationReport, spec: ElectrostaticPlasmaSpec) -> None:
    """Check explicit Boris pusher stability and accuracy w.r.t. ω_pe.

    The explicit leapfrog (Boris) pusher is unstable when dt * ω_pe >= 2.
    Accuracy degrades noticeably when dt * ω_pe > 0.2.

    ω_pe = sqrt(n₀ · q_e² / (m_e · ε₀))
    """
    omega_pe = math.sqrt(spec.n0 * _Q_E ** 2 / (_M_E * _EPS0))
    dt_ope   = spec.const_dt * omega_pe

    if dt_ope >= 2.0:
        r.add(
            Severity.ERROR,
            "es.plasma_frequency",
            (
                f"dt × ω_pe = {dt_ope:.3g} >= 2: the explicit Boris pusher is unstable. "
                f"Plasma frequency ω_pe = {omega_pe:.3e} rad/s; "
                f"maximum stable dt = {2.0 / omega_pe:.3e} s. "
                f"Reduce const_dt or decrease n0."
            ),
            omega_pe=round(omega_pe, 3),
            dt_ope=round(dt_ope, 4),
            max_stable_dt=round(2.0 / omega_pe, 12),
        )
    elif dt_ope > 0.2:
        r.add(
            Severity.WARNING,
            "es.plasma_frequency.accuracy",
            (
                f"dt × ω_pe = {dt_ope:.3g} > 0.2: electron plasma oscillations may be "
                f"under-resolved. Recommend dt × ω_pe < 0.1 for good accuracy. "
                f"ω_pe = {omega_pe:.3e} rad/s; suggested dt < {0.1 / omega_pe:.3e} s."
            ),
            omega_pe=round(omega_pe, 3),
            dt_ope=round(dt_ope, 4),
            suggested_dt=round(0.1 / omega_pe, 12),
        )
