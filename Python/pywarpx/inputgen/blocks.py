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
from typing import List, Optional

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


# ---------------------------------------------------------------------------
# Embedded boundary block
# ---------------------------------------------------------------------------

@dataclass
class EBSpec:
    """Embedded boundary (EB) geometry.

    The EB surface is the zero level-set of ``eb_implicit_function``.
    Domain cells are where the function value is positive; EB interior where
    negative.

    Common expressions:
      Sphere (3D):   ``-(x**2 + y**2 + z**2 - R**2)``   R = radius in m
      Sphere (2D):   ``-(x**2 + y**2 - R**2)``
      Cylinder (z):  ``-(x**2 + y**2 - R**2)``
      Box (2D):      ``max(max(x - xhi, xlo - x), max(y - yhi, ylo - y))``
      Plane (z=z0):  ``z0 - z``  (domain above z0)

    Set ``eb_potential`` to apply a Dirichlet voltage on the EB surface
    (electrostatic mode only; expression of x,y,z,t).
    Set ``stl_file`` to load geometry from an STL file instead.
    Leave all fields empty to disable EB.
    """
    eb_implicit_function: str = ""   # leave empty to disable EB
    eb_potential: str = ""           # potential on EB surface; empty = PEC (grounded)
    stl_file: str = ""               # path to STL file (alternative to implicit function)


def validate_eb(eb: EBSpec) -> ValidationReport:
    """Validate embedded boundary spec.

    Returns OK immediately when no EB is configured (both eb_implicit_function
    and stl_file are empty).
    """
    r = ValidationReport()
    has_fn  = bool(eb.eb_implicit_function.strip())
    has_stl = bool(eb.stl_file.strip())

    if not has_fn and not has_stl:
        return r  # no EB configured — valid

    if has_fn and has_stl:
        r.add(Severity.ERROR, "eb.overspecified",
              "Set eb_implicit_function OR stl_file, not both")

    if has_stl and not eb.stl_file.strip().endswith(".stl"):
        r.add(Severity.WARNING, "eb.stl_extension",
              "stl_file does not end with '.stl'", stl_file=eb.stl_file)

    return r


# ---------------------------------------------------------------------------
# Analytic external B-field block
# ---------------------------------------------------------------------------

@dataclass
class ExtBFieldSpec:
    """External/initial magnetic field defined by analytic (x,y,z) expressions.

    When any expression is set, WarpX uses
    ``warpx.B_ext_grid_init_style = parse_B_ext_grid_function``
    and evaluates Bx/By/Bz at each grid point at t=0.
    All three axes must be specified if any one is set.

    Leave all fields empty to use a constant uniform B0 from the parent spec
    (``warpx.B_ext_grid_init_style = constant``).

    Example (Harris sheet, current in y-direction):
        Bx_expression = "B0*tanh(z/delta)"
        By_expression = "sqrt(Bg**2 + B0**2 - (B0*tanh(z/delta))**2)"
        Bz_expression = "dB*sin(2*pi*x/Lx)*cos(pi*z/Lz)"

    Constants like B0, delta, etc. must be prefixed ``my_constants.NAME``
    in the parent generator, or embedded as numeric literals.
    """
    Bx_expression: str = ""
    By_expression: str = ""
    Bz_expression: str = ""


def validate_ext_bfield(bf: ExtBFieldSpec) -> ValidationReport:
    r = ValidationReport()
    exprs = [bf.Bx_expression, bf.By_expression, bf.Bz_expression]
    if not any(e.strip() for e in exprs):
        return r  # constant mode — ok

    missing = [ax for ax, e in zip(("Bx", "By", "Bz"), exprs) if not e.strip()]
    if missing:
        r.add(Severity.ERROR, "extbfield.incomplete",
              f"All three B expressions must be set when any one is set; missing: {missing}",
              missing=missing)
    return r


# ---------------------------------------------------------------------------
# Particle beam block (PWFA / beam-driven wakefields)
# ---------------------------------------------------------------------------

@dataclass
class ParticleBeamSpec:
    """Gaussian particle beam for PWFA or other beam-driven scenarios.

    Lengths in metres; momenta normalised to m_e * c (relativistic units).
    q_tot is the total charge in Coulombs (negative for electron beams).
    """
    x_rms: float = 2e-6     # Transverse RMS width [m]
    y_rms: float = 2e-6     # Transverse RMS width [m]
    z_rms: float = 4e-6     # Longitudinal RMS length [m]
    z_cut: float = 3.0      # Cutoff in sigma (gaussian_beam injection)
    uz_m: float = 2000.0    # Mean normalised momentum (gamma * beta_z)
    uz_th: float = 20.0     # Thermal spread in uz
    q_tot: float = -1e-9    # Total charge [C]; negative for electron beams
    z_mean: float = -50e-6  # Beam centroid z-position [m]
    n_macro: int = 1000     # Number of macro-particles


def validate_particle_beam(beam: ParticleBeamSpec, label: str = "beam") -> ValidationReport:
    r = ValidationReport()
    for attr, val in [("x_rms", beam.x_rms), ("y_rms", beam.y_rms), ("z_rms", beam.z_rms)]:
        if val <= 0:
            r.add(Severity.ERROR, f"{label}.{attr}", f"{attr} must be > 0", **{attr: val})
    if beam.uz_m <= 0:
        r.add(Severity.WARNING, f"{label}.uz_m",
              f"uz_m <= 0: beam is not forward-propagating", uz_m=beam.uz_m)
    if beam.n_macro <= 0:
        r.add(Severity.ERROR, f"{label}.n_macro", "n_macro must be > 0", n_macro=beam.n_macro)
    return r


# ---------------------------------------------------------------------------
# Implicit EM solver block
# ---------------------------------------------------------------------------

@dataclass
class ImplicitSolverSpec:
    """Theta-implicit EM field solver (WarpX predictor-corrector implicit EM).

    When enabled, WarpX uses ``algo.evolve_scheme = theta_implicit_em`` which
    solves the Maxwell equations implicitly, removing the CFL stability
    restriction on dt.  The simulation must also use a fixed ``warpx.const_dt``
    instead of a CFL-based timestep.

    This is useful when:
      - The grid resolves the plasma skin depth but not the wave period at CFL
      - The physics of interest evolves slowly relative to c/dx
      - Long-duration magnetised plasma runs (drift waves, MHD-like regimes)

    Typical settings: theta=0.5 (Crank-Nicolson), solver_type="picard",
    max_iters=30, tolerance=1e-3.
    """
    enabled: bool = False
    theta: float = 0.5           # Temporal centering: 0.5 = Crank-Nicolson
    solver_type: str = "picard"  # Nonlinear solver: "picard" or "newton"
    max_iters: int = 30          # Maximum solver iterations per step
    tolerance: float = 1e-3     # Relative convergence tolerance
    const_dt: float = 0.0        # Fixed timestep [s]; required when enabled (0 = use CFL)


def validate_implicit_solver(imp: ImplicitSolverSpec) -> ValidationReport:
    """Validate implicit EM solver settings.

    Returns OK immediately when ``enabled=False``.
    """
    r = ValidationReport()
    if not imp.enabled:
        return r

    if not (0.0 < imp.theta <= 1.0):
        r.add(Severity.ERROR, "implicit.theta",
              "theta must be in (0, 1]", theta=imp.theta)
    if imp.solver_type not in ("picard", "newton"):
        r.add(Severity.ERROR, "implicit.solver_type",
              "solver_type must be 'picard' or 'newton'", solver_type=imp.solver_type)
    if imp.max_iters <= 0:
        r.add(Severity.ERROR, "implicit.max_iters", "max_iters must be > 0")
    if imp.tolerance <= 0:
        r.add(Severity.ERROR, "implicit.tolerance", "tolerance must be > 0")
    if imp.const_dt <= 0:
        r.add(Severity.ERROR, "implicit.const_dt",
              "const_dt must be > 0 when implicit solver is enabled "
              "(implicit EM does not use a CFL-based timestep)",
              const_dt=imp.const_dt)
    return r
