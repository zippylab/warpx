from __future__ import annotations

"""Validator for ElectrostaticPICSpec."""

import math

from .blocks import (
    Severity,
    ValidationReport,
    check_boris_stability,
    check_debye_resolution,
    check_fft_requires_periodic,
    validate_amr,
    validate_collision,
    validate_diag,
    validate_domain,
    validate_eb,
    validate_es_solver,
    validate_species_def,
)
from .electrostatic_pic import ElectrostaticPICSpec


def _check_bc_lengths(spec: ElectrostaticPICSpec, r: ValidationReport) -> None:
    dim = spec.domain.dim
    for attr, val in [("field_bc_lo", spec.field_bc_lo), ("field_bc_hi", spec.field_bc_hi)]:
        if val is not None and len(val) != dim:
            r.add(
                Severity.ERROR, f"es.{attr}.len",
                f"{attr} must have length {dim} (one entry per axis)",
                attr=attr, got=len(val), expected=dim,
            )


def validate_electrostatic_pic_spec(spec: ElectrostaticPICSpec) -> ValidationReport:
    """Validate an ElectrostaticPICSpec.

    Runs per-block validators then cross-block checks (species product
    references, collision species references, BC length consistency,
    FFT/periodic BC compatibility, Debye resolution, plasma frequency stability).
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
    check_fft_requires_periodic(
        spec.solver.poisson_solver, spec.domain.field_bc, r, code_prefix="es"
    )

    # Debye resolution and plasma-frequency stability checks (electron species only)
    electron_specs = [
        sp for sp in spec.species
        if sp.charge == -1.0 and sp.injection_style != "none"
    ]
    if electron_specs:
        dx_max = max(
            (hi - lo) / nc
            for lo, hi, nc in zip(
                spec.domain.lower_bound,
                spec.domain.upper_bound,
                spec.domain.number_of_cells,
            )
        )
        # Debye check: use highest-density electron with known temperature
        e_with_temp = [sp for sp in electron_specs if sp.density > 0 and sp.temperature_eV > 0]
        if e_with_temp:
            ref = max(e_with_temp, key=lambda sp: sp.density)
            check_debye_resolution(
                dx_max, ref.density, ref.temperature_eV, r,
                code_prefix="es", species_name=ref.name,
            )
        # Boris stability: use highest-density electron species
        highest_n = max(sp.density for sp in electron_specs)
        check_boris_stability(spec.solver.const_dt, highest_n, r, code_prefix="es")

    return r
