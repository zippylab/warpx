from __future__ import annotations

"""Validator for ElectrostaticPICSpec."""

import math

from .blocks import (
    Severity,
    ValidationReport,
    validate_amr,
    validate_collision,
    validate_diag,
    validate_domain,
    validate_eb,
    validate_es_solver,
    validate_species_def,
)
from .electrostatic_pic import ElectrostaticPICSpec

_M_E = 9.1093837015e-31
_Q_E = 1.602176634e-19
_EPS0 = 8.8541878128e-12


def _check_bc_lengths(spec: ElectrostaticPICSpec, r: ValidationReport) -> None:
    dim = spec.domain.dim
    for attr, val in [("field_bc_lo", spec.field_bc_lo), ("field_bc_hi", spec.field_bc_hi)]:
        if val is not None and len(val) != dim:
            r.add(
                Severity.ERROR, f"es.{attr}.len",
                f"{attr} must have length {dim} (one entry per axis)",
                attr=attr, got=len(val), expected=dim,
            )


def _check_plasma_frequency(spec: ElectrostaticPICSpec, r: ValidationReport) -> None:
    """Warn / error when dt * omega_pe is dangerously large.

    Finds the highest-density electron-like species (charge == -1) and checks
    the Boris stability criterion: dt * omega_pe < 2 (hard limit).
    """
    electron_specs = [
        sp for sp in spec.species
        if sp.charge == -1.0 and sp.injection_style != "none"
    ]
    if not electron_specs:
        return

    highest_n = max(sp.density for sp in electron_specs)
    omega_pe = math.sqrt(highest_n * _Q_E**2 / (_M_E * _EPS0))
    dt = spec.solver.const_dt
    dt_omega = dt * omega_pe

    if dt_omega >= 2.0:
        r.add(
            Severity.ERROR, "es.debye.dt_omega_pe",
            f"dt * omega_pe = {dt_omega:.3f} >= 2 — explicit Boris is unconditionally "
            f"unstable (max density: {highest_n:.2e} m^-3, const_dt: {dt:.2e} s)",
            dt_omega_pe=dt_omega, n_max=highest_n,
        )
    elif dt_omega > 0.2:
        r.add(
            Severity.WARNING, "es.debye.dt_omega_pe",
            f"dt * omega_pe = {dt_omega:.3f} > 0.2 — accuracy may be reduced "
            f"(max density: {highest_n:.2e} m^-3, const_dt: {dt:.2e} s)",
            dt_omega_pe=dt_omega, n_max=highest_n,
        )


def validate_electrostatic_pic_spec(spec: ElectrostaticPICSpec) -> ValidationReport:
    """Validate an ElectrostaticPICSpec.

    Runs per-block validators then cross-block checks (species product
    references, collision species references, BC length consistency,
    plasma frequency stability).
    """
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_es_solver(spec.solver))
    r.merge(validate_diag(spec.diag))

    if spec.eb is not None:
        r.merge(validate_eb(spec.eb))

    # Build name set for cross-reference checks
    all_names: set = {sp.name for sp in spec.species}

    for sp in spec.species:
        r.merge(validate_species_def(sp, all_names))

    for col in spec.collisions:
        r.merge(validate_collision(col, all_names))

    _check_bc_lengths(spec, r)
    _check_plasma_frequency(spec, r)

    return r
