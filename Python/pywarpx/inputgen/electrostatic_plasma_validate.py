from __future__ import annotations

from .blocks import (
    check_boris_stability,
    check_debye_resolution,
    validate_amr,
    validate_diag,
    validate_domain,
    validate_eb,
    validate_solver,
)
from .electrostatic_plasma import ElectrostaticPlasmaSpec
from .spec import Severity, ValidationReport


def validate_electrostatic_plasma_spec(spec: ElectrostaticPlasmaSpec) -> ValidationReport:
    """Validate an ElectrostaticPlasmaSpec.

    Checks (in order):
    1. Domain structure (dim in {1,2,3}, list lengths, bounds)
    2. Solver parameters (max_steps > 0)
    3. Diagnostics
    4. ES-specific scalar parameters (n0, Te, Ti, const_dt, …)
    5. FFT/periodic BC compatibility
    6. Debye length resolution: dx < λ_De
    7. Electron plasma frequency stability: dt * ω_pe < 2
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

    dx_max = max(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )
    check_debye_resolution(dx_max, spec.n0, spec.Te, r, code_prefix="es")
    check_boris_stability(spec.const_dt, spec.n0, r, code_prefix="es")

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
