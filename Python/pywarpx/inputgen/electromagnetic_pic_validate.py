from __future__ import annotations

"""Validator for ElectromagneticPICSpec."""

from .blocks import (
    Severity,
    ValidationReport,
    validate_diag,
    validate_domain,
    validate_eb,
    validate_em_solver,
    validate_implicit_solver,
    validate_laser,
    validate_species_def,
    validate_collision,
)
from .electromagnetic_pic import ElectromagneticPICSpec


def _check_bc_lengths(spec: ElectromagneticPICSpec, r: ValidationReport) -> None:
    dim = spec.domain.dim
    for attr, val in [("field_bc_lo", spec.field_bc_lo), ("field_bc_hi", spec.field_bc_hi)]:
        if val is not None and len(val) != dim:
            r.add(
                Severity.ERROR, f"em.{attr}.len",
                f"{attr} must have length {dim} (one entry per axis)",
                attr=attr, got=len(val), expected=dim,
            )


def _check_psatd_with_implicit(spec: ElectromagneticPICSpec, r: ValidationReport) -> None:
    if spec.solver.maxwell_solver == "psatd" and spec.implicit.enabled:
        r.add(
            Severity.ERROR, "em.psatd_implicit_incompatible",
            "PSATD and the theta-implicit EM solver are incompatible; "
            "use Yee or CKC for implicit runs",
        )


def validate_electromagnetic_pic_spec(spec: ElectromagneticPICSpec) -> ValidationReport:
    """Validate an ElectromagneticPICSpec.

    Runs per-block validators then cross-block checks (species product
    references, collision species references, BC length consistency).
    """
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_em_solver(spec.solver))
    r.merge(validate_diag(spec.diag))

    if spec.implicit.enabled:
        r.merge(validate_implicit_solver(spec.implicit))

    if spec.eb is not None:
        r.merge(validate_eb(spec.eb))

    if spec.laser is not None:
        r.merge(validate_laser(spec.laser))

    # Build name set for cross-reference checks
    all_names: set = {sp.name for sp in spec.species}

    for sp in spec.species:
        r.merge(validate_species_def(sp, all_names))

    for col in spec.collisions:
        r.merge(validate_collision(col, all_names))

    _check_bc_lengths(spec, r)
    _check_psatd_with_implicit(spec, r)

    return r
