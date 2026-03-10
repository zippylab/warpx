from __future__ import annotations

"""Validator for ElectromagneticPICSpec."""

import math

from .blocks import (
    Severity,
    ValidationReport,
    check_boris_stability,
    check_debye_resolution,
    validate_amr,
    validate_collision,
    validate_diag,
    validate_domain,
    validate_eb,
    validate_em_solver,
    validate_implicit_solver,
    validate_laser,
    validate_species_def,
)
from .electromagnetic_pic import ElectromagneticPICSpec

_C = 299792458.0


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


def _check_plasma_physics(spec: ElectromagneticPICSpec, r: ValidationReport) -> None:
    """Advisory checks for explicit FDTD runs: Debye resolution and Boris stability.

    Both checks are WARNING-only for EM-PIC (not ERROR), since EM fields propagate
    at the speed of light and the finite-grid instability is less catastrophic than
    in ES-PIC.  They are skipped entirely for PSATD (no CFL limit) and implicit runs.
    """
    is_explicit_fdtd = (
        spec.solver.maxwell_solver in ("yee", "ckc")
        and not spec.implicit.enabled
    )

    electron_specs = [
        sp for sp in spec.species
        if sp.charge == -1.0 and sp.injection_style != "none" and sp.density > 0
    ]
    if not electron_specs:
        return

    dx_min = min(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )
    dx_max = max(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )

    # Debye resolution: WARNING only for EM-PIC (error_above=inf)
    e_with_temp = [sp for sp in electron_specs if sp.temperature_eV > 0]
    if e_with_temp:
        ref = max(e_with_temp, key=lambda sp: sp.density)
        check_debye_resolution(
            dx_max, ref.density, ref.temperature_eV, r,
            code_prefix="em", species_name=ref.name,
            error_above=float("inf"),  # WARNING only, never ERROR, for EM-PIC
        )

    # Boris stability: WARNING only for explicit FDTD (estimate dt from CFL and dx)
    if is_explicit_fdtd:
        dim = spec.domain.dim
        dt_est = spec.solver.cfl * dx_min / (_C * math.sqrt(dim))
        highest_n = max(sp.density for sp in electron_specs)
        check_boris_stability(
            dt_est, highest_n, r,
            code_prefix="em",
            error_above=float("inf"),  # WARNING only, never ERROR, for EM-PIC
        )


def _check_laser_resolution(spec: ElectromagneticPICSpec, r: ValidationReport) -> None:
    """Warn if the grid is too coarse to resolve the laser wavelength."""
    if spec.laser is None:
        return
    dx_max = max(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )
    cells_per_wavelength = spec.laser.wavelength / dx_max
    if cells_per_wavelength < 5:
        r.add(
            Severity.ERROR, "em.laser_resolution",
            f"Only {cells_per_wavelength:.1f} cells per laser wavelength "
            f"(λ={spec.laser.wavelength:.3e} m, dx_max={dx_max:.3e} m). "
            f"At least 10 cells per wavelength are required for accurate propagation.",
            cells_per_wavelength=round(cells_per_wavelength, 2),
            wavelength=spec.laser.wavelength,
            dx_max=round(dx_max, 9),
        )
    elif cells_per_wavelength < 10:
        r.add(
            Severity.WARNING, "em.laser_resolution",
            f"Only {cells_per_wavelength:.1f} cells per laser wavelength "
            f"(λ={spec.laser.wavelength:.3e} m, dx_max={dx_max:.3e} m). "
            f"At least 10 cells per wavelength are recommended for accurate propagation.",
            cells_per_wavelength=round(cells_per_wavelength, 2),
            wavelength=spec.laser.wavelength,
            dx_max=round(dx_max, 9),
        )


def validate_electromagnetic_pic_spec(spec: ElectromagneticPICSpec) -> ValidationReport:
    """Validate an ElectromagneticPICSpec.

    Runs per-block validators then cross-block checks (species product
    references, collision species references, BC length consistency,
    PSATD/implicit compatibility, laser resolution, Debye/Boris advisory
    checks for explicit FDTD runs).
    """
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
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
    _check_laser_resolution(spec, r)
    _check_plasma_physics(spec, r)

    return r
