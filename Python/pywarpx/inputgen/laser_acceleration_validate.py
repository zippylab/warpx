from __future__ import annotations

from .blocks import (
    validate_diag,
    validate_domain,
    validate_implicit_solver,
    validate_laser,
    validate_solver,
    validate_species,
)
from .laser_acceleration import LaserAccelerationSpec
from .spec import Severity, ValidationReport


def validate_laser_acceleration_spec(spec: LaserAccelerationSpec) -> ValidationReport:
    r = ValidationReport()

    # Domain validation runs first; if it fails (wrong dim, bad lengths) we
    # cannot safely index into the lists below.
    r.merge(validate_domain(spec.domain, allowed_dims=(2, 3)))
    if not r.ok:
        return r

    r.merge(validate_solver(spec.solver))
    r.merge(validate_laser(spec.laser))
    r.merge(validate_species(spec.species, spec.domain))
    r.merge(validate_diag(spec.diag))
    r.merge(validate_implicit_solver(spec.implicit))

    # Cross-cutting: resolution heuristic
    _check_laser_resolution(r, spec)

    return r


def _check_laser_resolution(r: ValidationReport, spec: LaserAccelerationSpec) -> None:
    """Warn if any axis has fewer than 10 cells per laser wavelength."""
    try:
        cell_sizes = [
            (hi - lo) / n
            for n, lo, hi in zip(
                spec.domain.number_of_cells,
                spec.domain.lower_bound,
                spec.domain.upper_bound,
            )
        ]
        cells_per_lambda = [spec.laser.wavelength / dx for dx in cell_sizes]
        min_cpl = min(cells_per_lambda)
        if min_cpl < 10:
            r.add(
                Severity.WARNING,
                "laser.resolution",
                "Resolution may be too coarse for the laser wavelength (<10 cells per wavelength)",
                cells_per_lambda=cells_per_lambda,
                cell_sizes=cell_sizes,
                wavelength=spec.laser.wavelength,
            )
    except Exception as e:
        r.add(Severity.WARNING, "laser.resolution.compute_failed",
              "Failed to compute resolution heuristic", error=str(e))
