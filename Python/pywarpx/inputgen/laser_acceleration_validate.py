from __future__ import annotations

import math

from .spec import Severity, ValidationReport
from .laser_acceleration import LaserAccelerationSpec


def validate_laser_acceleration_spec(spec: LaserAccelerationSpec) -> ValidationReport:
    r = ValidationReport()

    if spec.dim != 2:
        r.add(Severity.ERROR, "laser.dim", "laser_acceleration currently supports dim=2 only", dim=spec.dim)
        return r

    if len(spec.number_of_cells) != 2:
        r.add(Severity.ERROR, "laser.ncell.len", "number_of_cells must have length 2", n=len(spec.number_of_cells))

    if len(spec.lower_bound) != 2 or len(spec.upper_bound) != 2:
        r.add(Severity.ERROR, "laser.bounds.len", "bounds must have length 2")

    if len(spec.field_bc) != 2:
        r.add(Severity.ERROR, "laser.bc.len", "field_bc must have length 2")

    if not r.ok:
        return r

    if any(n <= 0 for n in spec.number_of_cells):
        r.add(Severity.ERROR, "laser.ncell.nonpositive", "cell counts must be > 0")

    if any(hi <= lo for lo, hi in zip(spec.lower_bound, spec.upper_bound)):
        r.add(Severity.ERROR, "laser.bounds.invalid", "upper_bound must be > lower_bound")

    if spec.max_steps <= 0:
        r.add(Severity.ERROR, "laser.max_steps", "max_steps must be > 0")

    if not (0.0 < spec.cfl <= 1.0):
        r.add(Severity.WARNING, "laser.cfl", "cfl should be in (0,1]", cfl=spec.cfl)

    if spec.plasma_density <= 0:
        r.add(Severity.ERROR, "laser.plasma_density", "plasma_density must be > 0", plasma_density=spec.plasma_density)

    if not (spec.plasma_zmin < spec.plasma_zmax):
        r.add(Severity.ERROR, "laser.plasma_slab", "plasma_zmin must be < plasma_zmax")

    # Sanity: ensure plasma slab overlaps domain z-range
    zmin = spec.lower_bound[1]
    zmax = spec.upper_bound[1]
    if spec.plasma_zmax <= zmin or spec.plasma_zmin >= zmax:
        r.add(
            Severity.WARNING,
            "laser.plasma_outside_domain",
            "plasma slab does not overlap simulation z-range",
            domain_zmin=zmin,
            domain_zmax=zmax,
            plasma_zmin=spec.plasma_zmin,
            plasma_zmax=spec.plasma_zmax,
        )

    # Laser sanity
    if spec.wavelength <= 0:
        r.add(Severity.ERROR, "laser.wavelength", "wavelength must be > 0")

    if spec.waist <= 0:
        r.add(Severity.ERROR, "laser.waist", "waist must be > 0")

    if spec.duration <= 0:
        r.add(Severity.ERROR, "laser.duration", "duration must be > 0")

    if spec.a0 <= 0:
        r.add(Severity.WARNING, "laser.a0", "a0 should be > 0", a0=spec.a0)

    # Resolution heuristics
    try:
        dx = (spec.upper_bound[0] - spec.lower_bound[0]) / spec.number_of_cells[0]
        dz = (spec.upper_bound[1] - spec.lower_bound[1]) / spec.number_of_cells[1]
        # for typical LWFA, ~20 cells per wavelength is a baseline
        min_cells_per_lambda = min(spec.wavelength / dx, spec.wavelength / dz)
        if min_cells_per_lambda < 10:
            r.add(
                Severity.WARNING,
                "laser.resolution",
                "Resolution may be too coarse for the laser wavelength (<10 cells per wavelength)",
                cells_per_lambda=min_cells_per_lambda,
                dx=dx,
                dz=dz,
                wavelength=spec.wavelength,
            )
    except Exception as e:
        r.add(Severity.WARNING, "laser.resolution.compute_failed", "Failed to compute resolution heuristic", error=str(e))

    return r
