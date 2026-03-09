from __future__ import annotations

"""Composable building-block dataclasses for WarpX input generation.

Each block captures one logical concern (domain, solver, species, laser,
diagnostics) and comes with its own validator.  Simulation-family specs
(e.g. LaserAccelerationSpec) compose these blocks rather than carrying
every field at the top level.

The external JSON format remains flat (backward compatible); a `from_dict`
classmethod on the composite spec handles distribution of flat keys to the
appropriate block.
"""

from dataclasses import dataclass, field
from typing import List

from .spec import Severity, ValidationReport


# ---------------------------------------------------------------------------
# Block dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DomainSpec:
    """Simulation domain: geometry, cell counts, bounds, field BCs."""
    dim: int = 2
    number_of_cells: List[int] = field(default_factory=lambda: [128, 512])
    lower_bound: List[float] = field(default_factory=lambda: [-20e-6, 0.0])
    upper_bound: List[float] = field(default_factory=lambda: [20e-6, 200e-6])
    field_bc: List[str] = field(default_factory=lambda: ["periodic", "open"])


@dataclass
class SolverSpec:
    """Time-stepping and field-solver parameters."""
    max_steps: int = 200
    cfl: float = 0.99
    particle_shape: str = "linear"


@dataclass
class SpeciesSpec:
    """Plasma species parameters for the two-species (e + H+) laser setup."""
    plasma_density: float = 1.0e24   # m^-3
    plasma_zmin: float = 20e-6
    plasma_zmax: float = 180e-6


@dataclass
class LaserSpec:
    """Gaussian laser pulse parameters."""
    wavelength: float = 0.8e-6
    a0: float = 2.0
    waist: float = 15e-6
    duration: float = 30e-15
    focal_position_z: float = 60e-6
    centroid_position_z: float = 0.0


@dataclass
class DiagSpec:
    """Diagnostics output parameters."""
    diag_period: int = 50
    diag_fields: List[str] = field(
        default_factory=lambda: ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    )


# ---------------------------------------------------------------------------
# Per-block validators
# ---------------------------------------------------------------------------

_VALID_FIELD_BCS = {"periodic", "open", "pec", "pmc", "damped"}


def validate_domain(domain: DomainSpec, *, allowed_dims=(1, 2, 3)) -> ValidationReport:
    """Validate domain structure.

    *allowed_dims* restricts the permitted values of ``domain.dim``; each
    simulation family passes its own constraint (e.g. laser acceleration uses
    ``allowed_dims=(2, 3)``, hybrid-PIC uses ``allowed_dims=(1, 2, 3)``).
    """
    r = ValidationReport()

    if domain.dim not in allowed_dims:
        r.add(Severity.ERROR, "domain.dim",
              f"dim must be one of {allowed_dims}", dim=domain.dim)
        return r

    for name, lst in [
        ("number_of_cells", domain.number_of_cells),
        ("lower_bound", domain.lower_bound),
        ("upper_bound", domain.upper_bound),
        ("field_bc", domain.field_bc),
    ]:
        if len(lst) != domain.dim:
            r.add(Severity.ERROR, f"domain.{name}.len",
                  f"{name} must have length {domain.dim}",
                  got=len(lst), expected=domain.dim)

    if not r.ok:
        return r

    if any(n <= 0 for n in domain.number_of_cells):
        r.add(Severity.ERROR, "domain.ncell.nonpositive",
              "all cell counts must be > 0")

    if any(hi <= lo for lo, hi in zip(domain.lower_bound, domain.upper_bound)):
        r.add(Severity.ERROR, "domain.bounds.invalid",
              "upper_bound must be > lower_bound in every dimension")

    for bc in domain.field_bc:
        if bc not in _VALID_FIELD_BCS:
            r.add(Severity.WARNING, "domain.bc.unknown",
                  f"unrecognised field BC '{bc}'", bc=bc)

    return r


def validate_solver(solver: SolverSpec) -> ValidationReport:
    r = ValidationReport()

    if solver.max_steps <= 0:
        r.add(Severity.ERROR, "solver.max_steps",
              "max_steps must be > 0", max_steps=solver.max_steps)

    if not (0.0 < solver.cfl <= 1.0):
        r.add(Severity.WARNING, "solver.cfl",
              "cfl should be in (0, 1]", cfl=solver.cfl)

    return r


def validate_species(species: SpeciesSpec, domain: DomainSpec) -> ValidationReport:
    r = ValidationReport()

    if species.plasma_density <= 0:
        r.add(Severity.ERROR, "species.density",
              "plasma_density must be > 0", plasma_density=species.plasma_density)

    if not (species.plasma_zmin < species.plasma_zmax):
        r.add(Severity.ERROR, "species.slab",
              "plasma_zmin must be < plasma_zmax")

    # Sanity: plasma slab should overlap the domain z-range (z is the last axis)
    zmin = domain.lower_bound[-1]
    zmax = domain.upper_bound[-1]
    if species.plasma_zmax <= zmin or species.plasma_zmin >= zmax:
        r.add(
            Severity.WARNING,
            "species.plasma_outside_domain",
            "plasma slab does not overlap simulation z-range",
            domain_zmin=zmin, domain_zmax=zmax,
            plasma_zmin=species.plasma_zmin, plasma_zmax=species.plasma_zmax,
        )

    return r


def validate_laser(laser: LaserSpec) -> ValidationReport:
    r = ValidationReport()

    if laser.wavelength <= 0:
        r.add(Severity.ERROR, "laser.wavelength", "wavelength must be > 0")
    if laser.waist <= 0:
        r.add(Severity.ERROR, "laser.waist", "waist must be > 0")
    if laser.duration <= 0:
        r.add(Severity.ERROR, "laser.duration", "duration must be > 0")
    if laser.a0 <= 0:
        r.add(Severity.WARNING, "laser.a0", "a0 should be > 0", a0=laser.a0)

    return r


def validate_diag(diag: DiagSpec) -> ValidationReport:
    r = ValidationReport()

    if diag.diag_period <= 0:
        r.add(Severity.ERROR, "diag.period",
              "diag_period must be > 0", diag_period=diag.diag_period)

    if not diag.diag_fields:
        r.add(Severity.WARNING, "diag.fields_empty",
              "diag_fields is empty — no field data will be written")

    return r


# ---------------------------------------------------------------------------
# Hybrid-PIC specific blocks
# ---------------------------------------------------------------------------

@dataclass
class OhmSolverSpec:
    """Parameters for the kinetic-fluid hybrid Ohm's law solver.

    Default density n0_ref=3.3e22 m^-3 is self-consistent with B0=0.25 T and
    vA/c=1e-4 (proton mass), giving ion skin depth l_i ≈ 1.25 mm.  With
    dx=0.1*l_i ≈ 1.25e-4 m and substeps=40, the whistler CFL is ~0.77 (stable).
    """
    Te: float = 0.05             # Electron temperature [eV]
    n0_ref: float = 3.3e22      # Reference density [m^-3]; required when gamma != 1
    gamma: float = 1.0           # Adiabatic exponent (1 = isothermal)
    resistivity: float = 1e-7   # Plasma resistivity [Ohm·m]
    hyper_resistivity: float = 0.0  # Hyper-resistivity [Ohm·m³]
    substeps: int = 40           # B-field RK4 sub-steps per E-field half-step
    n_floor: float = 3.3e19     # Density floor [m^-3] (prevents division by zero)


@dataclass
class HybridIonSpec:
    """Single kinetic ion species for a hybrid-PIC run."""
    density: float = 3.3e22     # Number density [m^-3]
    mass_amu: float = 1.0       # Ion mass in proton masses (1.0 = proton)
    temperature_eV: float = 0.05  # Ion temperature [eV]
    ppc: int = 64               # Particles per cell


# ---------------------------------------------------------------------------
# Hybrid-PIC block validators
# ---------------------------------------------------------------------------

def validate_ohm_solver(ohm: OhmSolverSpec) -> ValidationReport:
    r = ValidationReport()

    if ohm.Te <= 0:
        r.add(Severity.ERROR, "ohm.Te", "Electron temperature Te must be > 0 eV", Te=ohm.Te)
    if ohm.n0_ref <= 0:
        r.add(Severity.ERROR, "ohm.n0_ref", "n0_ref must be > 0", n0_ref=ohm.n0_ref)
    if ohm.gamma < 1.0:
        r.add(Severity.WARNING, "ohm.gamma",
              "gamma < 1 is unusual (isothermal=1, adiabatic=5/3)", gamma=ohm.gamma)
    if ohm.resistivity < 0:
        r.add(Severity.ERROR, "ohm.resistivity", "resistivity must be >= 0")
    if ohm.hyper_resistivity < 0:
        r.add(Severity.ERROR, "ohm.hyper_resistivity", "hyper_resistivity must be >= 0")
    if ohm.substeps < 1:
        r.add(Severity.ERROR, "ohm.substeps", "substeps must be >= 1", substeps=ohm.substeps)
    if ohm.substeps % 2 != 0:
        r.add(Severity.ERROR, "ohm.substeps",
              "substeps must be divisible by 2", substeps=ohm.substeps)
    if ohm.n_floor < 0:
        r.add(Severity.ERROR, "ohm.n_floor", "n_floor must be >= 0")

    return r


def validate_hybrid_ion(ions: HybridIonSpec) -> ValidationReport:
    r = ValidationReport()

    if ions.density <= 0:
        r.add(Severity.ERROR, "ions.density", "ion density must be > 0", density=ions.density)
    if ions.mass_amu <= 0:
        r.add(Severity.ERROR, "ions.mass_amu",
              "ion mass_amu must be > 0", mass_amu=ions.mass_amu)
    if ions.temperature_eV < 0:
        r.add(Severity.ERROR, "ions.temperature_eV",
              "ion temperature must be >= 0", temperature_eV=ions.temperature_eV)
    if ions.ppc <= 0:
        r.add(Severity.ERROR, "ions.ppc", "ppc must be > 0", ppc=ions.ppc)

    return r
