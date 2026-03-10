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

import math
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


_KNOWN_REDUCED_SIMPLE: frozenset = frozenset({
    "ParticleEnergy", "ParticleMomentum", "FieldEnergy", "FieldMomentum",
    "FieldMaximum", "FieldPoyntingFlux", "RhoMaximum", "ParticleNumber",
    "LoadBalanceCosts", "LoadBalanceEfficiency", "Timestep",
})
_KNOWN_REDUCED_SPECIES: frozenset = frozenset({"BeamRelevant", "ParticleExtrema"})
_KNOWN_REDUCED_ALL: frozenset = (
    _KNOWN_REDUCED_SIMPLE | _KNOWN_REDUCED_SPECIES
    | frozenset({"FieldProbe", "FieldReduction", "ParticleHistogram", "ChargeOnEB"})
)


@dataclass
class ReducedDiagSpec:
    """A single WarpX reduced diagnostic entry.

    ``type`` selects the diagnostic kind (e.g. ``"FieldEnergy"``,
    ``"FieldProbe"``, ``"ParticleHistogram"``).  ``name`` is the ParmParse
    prefix; if empty it is auto-derived from ``type`` by lowercasing.
    ``period`` (0 = use parent DiagSpec.diag_period) sets the output interval.

    FieldProbe fields:
        probe_geometry: "Point" | "Line" | "Plane"
        x/y/z_probe: probe position; z1_probe: Line end point
        resolution: number of points (Line/Plane); interp_order; integrate

    FieldReduction fields:
        reduction_type: "Maximum" | "Minimum" | "Integral"
        reduced_function: analytic expression of (x,y,z,Ex,Ey,Ez,Bx,By,Bz,jx,jy,jz)

    Particle-species fields (BeamRelevant, ParticleExtrema, ParticleHistogram):
        species: species name string

    ParticleHistogram fields:
        bin_number, bin_min, bin_max, histogram_function, normalization, filter_function

    ChargeOnEB fields:
        weighting_function: optional analytic expression of (x,y,z)
    """
    type: str = ""                       # required; must be in _KNOWN_REDUCED_ALL
    name: str = ""                       # auto = type.lower() if empty
    period: int = 0                      # 0 → use parent DiagSpec.diag_period
    path: str = "diags/"
    # FieldProbe
    probe_geometry: str = "Point"        # "Point" | "Line" | "Plane"
    x_probe: float = 0.0
    y_probe: float = 0.0
    z_probe: float = 0.0
    z1_probe: float = 0.0               # Line end point
    resolution: int = 64                 # Line / Plane point count
    interp_order: int = 1
    integrate: bool = False
    # FieldReduction
    reduction_type: str = ""             # "Maximum" | "Minimum" | "Integral"
    reduced_function: str = ""           # analytic expression
    # Particle-species diagnostics
    species: str = ""                    # for BeamRelevant, ParticleExtrema, ParticleHistogram
    bin_number: int = 0
    bin_min: float = 0.0
    bin_max: float = 0.0
    histogram_function: str = ""
    normalization: str = ""              # "unity_particle_weight"|"max_to_unity"|"area_to_unity"
    filter_function: str = ""
    # ChargeOnEB
    weighting_function: str = ""


@dataclass
class DiagSpec:
    """Diagnostics output parameters (full diagnostics + reduced diagnostics).

    The main "Full" field diagnostic is controlled by ``diag_period``,
    ``diag_fields``, ``diag_format``, and ``write_species``.

    ``reduced_diags`` is a list of :class:`ReducedDiagSpec` entries that map
    to WarpX reduced diagnostics (scalar TSV output, one row per timestep).
    """
    diag_period: int = 50
    diag_fields: List[str] = field(
        default_factory=lambda: ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    )
    diag_format: str = "openpmd"         # "openpmd" | "plotfile"
    write_species: bool = False          # write particle data in the field diagnostic
    reduced_diags: List[ReducedDiagSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "DiagSpec":
        """Extract DiagSpec fields from a flat JSON dict."""
        kw = {k: d[k] for k in ("diag_period", "diag_fields", "diag_format", "write_species")
              if k in d}
        rds = [ReducedDiagSpec(**e) for e in d.get("reduced_diags", [])]
        return cls(**kw, reduced_diags=rds)


# ---------------------------------------------------------------------------
# Per-block validators
# ---------------------------------------------------------------------------

_VALID_FIELD_BCS = {"periodic", "open", "pml", "pec", "pmc", "damped", "none", "absorbing"}


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

    for i, rd in enumerate(diag.reduced_diags):
        prefix = f"diag.reduced[{i}]"
        if rd.type not in _KNOWN_REDUCED_ALL:
            r.add(Severity.ERROR, f"{prefix}.unknown_type",
                  f"Unknown reduced diagnostic type '{rd.type}'. "
                  f"Valid types: {sorted(_KNOWN_REDUCED_ALL)}",
                  type=rd.type)
            continue
        if rd.type == "FieldProbe":
            if rd.probe_geometry not in ("Point", "Line", "Plane"):
                r.add(Severity.ERROR, f"{prefix}.probe_geometry",
                      "probe_geometry must be 'Point', 'Line', or 'Plane'",
                      probe_geometry=rd.probe_geometry)
        if rd.type == "FieldReduction":
            if rd.reduction_type not in ("Maximum", "Minimum", "Integral"):
                r.add(Severity.ERROR, f"{prefix}.reduction_type",
                      "reduction_type must be 'Maximum', 'Minimum', or 'Integral'",
                      reduction_type=rd.reduction_type)
            if not rd.reduced_function.strip():
                r.add(Severity.ERROR, f"{prefix}.reduced_function",
                      "reduced_function must be set for FieldReduction")
        if rd.type in (_KNOWN_REDUCED_SPECIES | {"ParticleHistogram"}):
            if not rd.species.strip():
                r.add(Severity.ERROR, f"{prefix}.species",
                      f"species must be set for {rd.type}")
        if rd.type == "ParticleHistogram" and rd.bin_number <= 0:
            r.add(Severity.ERROR, f"{prefix}.bin_number",
                  "bin_number must be > 0 for ParticleHistogram",
                  bin_number=rd.bin_number)

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


# ---------------------------------------------------------------------------
# Multi-species block: SpeciesDefSpec (for broad EM-PIC and ES-PIC sim types)
# ---------------------------------------------------------------------------

@dataclass
class SpeciesDefSpec:
    """Definition of a single particle species for broad-scope PIC sim types.

    Used as elements of the ``species`` list in ``ElectromagneticPICSpec``
    and ``ElectrostaticPICSpec``.  All physics flags default to ``False``; the
    generator only emits lines for flags that are ``True``.

    Momentum priority rule: if ``temperature_eV > 0``, the generator computes
    an isotropic thermal speed ``u_th = sqrt(T_eV * q_e / m_kg) / c`` and
    applies it to all three axes, overriding any explicit ``u*_th`` values.
    If ``temperature_eV == 0`` but any ``u*_th`` is non-zero, the explicit
    per-component values are used.  If all thermal spreads are zero, the
    species is injected as a cold (constant) distribution.
    """
    name: str = ""
    charge: float = -1.0                # units of q_e (+1=proton, -1=electron)
    mass_amu: float = 5.486e-4          # atomic mass units; electron default
    injection_style: str = "NRandomPerCell"
    # --- NRandomPerCell / NUniformPerCell injection ---
    density: float = 1e20              # m^-3
    ppc: int = 4                        # particles per cell
    ppc_each_dim: Optional[List[int]] = None  # for NUniformPerCell; overrides ppc
    profile: str = "constant"          # "constant" | "parse_density_function"
    density_function: str = ""         # WarpX parser expression for profile
    temperature_eV: float = 0.0        # thermal temperature (see priority rule above)
    ux_m: float = 0.0                  # mean normalised momentum x
    uy_m: float = 0.0                  # mean normalised momentum y
    uz_m: float = 0.0                  # mean normalised momentum z (drift)
    ux_th: float = 0.0                 # thermal spread in ux
    uy_th: float = 0.0                 # thermal spread in uy
    uz_th: float = 0.0                 # thermal spread in uz
    # Spatial bounds (slab injection; None = no limit)
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    zmin: Optional[float] = None
    zmax: Optional[float] = None
    # --- gaussian_beam injection ---
    x_rms: float = 1e-6
    y_rms: float = 1e-6
    z_rms: float = 2e-6
    z_cut: float = 3.0
    q_tot: float = -1e-12              # total charge [C]
    z_mean: float = 0.0                # beam centroid z [m]
    n_macro: int = 1000
    # --- Field ionization ---
    do_field_ionization: bool = False
    physical_element: str = ""         # atomic element symbol, e.g. "N", "Ar", "He"
    ionization_initial_level: int = 0
    ionization_product_species: str = ""  # name of electron product species
    # --- QED ---
    do_qed_breit_wheeler: bool = False
    qed_bw_ele_product: str = ""       # electron product species name
    qed_bw_pos_product: str = ""       # positron product species name
    do_qed_quantum_sync: bool = False
    qed_qs_phot_product: str = ""      # photon product species name
    # --- Classical radiation reaction ---
    do_classical_radiation_reaction: bool = False
    # --- Continuous injection (moving window) ---
    do_continuous_injection: bool = False


def validate_species_def(sp: SpeciesDefSpec, all_names: set) -> ValidationReport:
    """Validate a single species definition.

    *all_names* is the set of all species names in the simulation, used to
    check that product/ionization target species actually exist.
    """
    r = ValidationReport()

    if not sp.name or " " in sp.name:
        r.add(Severity.ERROR, "species.name",
              "species name must be non-empty and contain no spaces", name=sp.name)

    if sp.mass_amu <= 0:
        r.add(Severity.ERROR, "species.mass_amu",
              "mass_amu must be > 0", name=sp.name, mass_amu=sp.mass_amu)

    if sp.injection_style == "gaussian_beam":
        if sp.n_macro <= 0:
            r.add(Severity.ERROR, "species.n_macro",
                  "n_macro must be > 0 for gaussian_beam injection",
                  name=sp.name, n_macro=sp.n_macro)
    elif sp.injection_style not in ("NRandomPerCell", "NUniformPerCell", "none"):
        r.add(Severity.WARNING, "species.injection_style",
              f"unrecognised injection_style '{sp.injection_style}'",
              name=sp.name, injection_style=sp.injection_style)

    if sp.do_field_ionization:
        if not sp.physical_element.strip():
            r.add(Severity.ERROR, "species.ionization.element",
                  "physical_element must be set when do_field_ionization=True",
                  name=sp.name)
        if sp.ionization_product_species not in all_names:
            r.add(Severity.ERROR, "species.ionization.product",
                  f"ionization_product_species '{sp.ionization_product_species}' "
                  f"not found in species list",
                  name=sp.name, product=sp.ionization_product_species)

    if sp.do_qed_breit_wheeler:
        for attr, val in [("qed_bw_ele_product", sp.qed_bw_ele_product),
                          ("qed_bw_pos_product", sp.qed_bw_pos_product)]:
            if not val or val not in all_names:
                r.add(Severity.ERROR, f"species.qed_bw.{attr}",
                      f"{attr} must name an existing species",
                      name=sp.name, product=val)

    if sp.do_qed_quantum_sync:
        if not sp.qed_qs_phot_product or sp.qed_qs_phot_product not in all_names:
            r.add(Severity.ERROR, "species.qed_qs.phot_product",
                  "qed_qs_phot_product must name an existing species",
                  name=sp.name, product=sp.qed_qs_phot_product)

    if sp.do_classical_radiation_reaction and sp.do_qed_quantum_sync:
        r.add(Severity.WARNING, "species.rr.redundant",
              "Both classical radiation reaction and QED quantum sync are enabled; "
              "this combination is unusual (QED already includes radiation reaction)",
              name=sp.name)

    return r


# ---------------------------------------------------------------------------
# Collision / nuclear reaction block
# ---------------------------------------------------------------------------

@dataclass
class CollisionSpec:
    """Definition of a single particle collision or nuclear reaction.

    Supported types:
      ``"coulomb"``       — binary Coulomb collisions (TA77/Nanbu)
      ``"nuclearfusion"`` — nuclear fusion (D-T, D-He3, D-D, p-B11, etc.)

    For nuclear fusion the ``event_multiplier`` should be set to a large value
    (typically 1e10–1e18) to boost the statistical probability of rare events
    to a level where they are observed within a simulation.  Setting it to 1.0
    is almost always physically incorrect.
    """
    name: str = ""
    type: str = "coulomb"                    # "coulomb" | "nuclearfusion"
    species: List[str] = field(default_factory=list)        # 2 reactant species names
    product_species: List[str] = field(default_factory=list)
    CoulombLog: float = 0.0                  # 0 = auto-compute from plasma parameters
    event_multiplier: float = 1.0            # nuclear only
    probability_target_value: float = 0.02  # nuclear only; target per-step probability


def validate_collision(col: CollisionSpec, all_names: set) -> ValidationReport:
    """Validate a single collision/reaction definition."""
    r = ValidationReport()

    if not col.name:
        r.add(Severity.ERROR, "collision.name", "collision name must be non-empty")

    _VALID_COLLISION_TYPES = {"coulomb", "nuclearfusion"}
    if col.type not in _VALID_COLLISION_TYPES:
        r.add(Severity.ERROR, "collision.type",
              f"type must be one of {sorted(_VALID_COLLISION_TYPES)}",
              name=col.name, type=col.type)
        return r

    if len(col.species) != 2:
        r.add(Severity.ERROR, "collision.species",
              "exactly 2 reactant species names required",
              name=col.name, got=len(col.species))
    else:
        for sp in col.species:
            if sp not in all_names:
                r.add(Severity.ERROR, "collision.species.unknown",
                      f"reactant species '{sp}' not found in species list",
                      name=col.name, species=sp)

    if col.type == "nuclearfusion":
        if not col.product_species:
            r.add(Severity.ERROR, "collision.nuclearfusion.products",
                  "product_species must be non-empty for nuclearfusion type",
                  name=col.name)
        else:
            for sp in col.product_species:
                if sp not in all_names:
                    r.add(Severity.ERROR, "collision.products.unknown",
                          f"product species '{sp}' not found in species list",
                          name=col.name, species=sp)
        if col.event_multiplier <= 1.0:
            r.add(Severity.WARNING, "collision.nuclearfusion.multiplier",
                  "event_multiplier <= 1.0 is almost certainly wrong for nuclear fusion "
                  "(typical values: 1e10–1e18)",
                  name=col.name, event_multiplier=col.event_multiplier)

    if col.type == "coulomb" and col.CoulombLog < 0:
        r.add(Severity.WARNING, "collision.coulomb.log",
              "CoulombLog < 0 is invalid; use 0 for auto-compute",
              name=col.name, CoulombLog=col.CoulombLog)

    return r


# ---------------------------------------------------------------------------
# EM-PIC solver block
# ---------------------------------------------------------------------------

@dataclass
class EMSolverSpec:
    """Field solver and particle pusher parameters for EM-PIC (FDTD/spectral).

    Maxwell solver options:
      ``"yee"``   — classic FDTD staggered-grid Yee scheme (default)
      ``"ckc"``   — Cole-Karkkainen-Cowan (reduced dispersion FDTD)
      ``"psatd"`` — pseudo-spectral analytical time-domain (no CFL limit)
      ``"none"``  — no field evolution (particle tracing)

    Particle pusher options:
      ``"boris"``    — standard Boris leap-frog (default)
      ``"vay"``      — Vay pusher (better energy conservation at high gamma)
      ``"higuera"``  — Higuera-Cary (implicit; designed for theta-implicit EM)

    Current deposition options:
      ``"esirkepov"`` — charge-conserving Esirkepov (default)
      ``"direct"``    — direct deposition
      ``"vay"``       — Vay current deposition
    """
    max_steps: int = 200
    cfl: float = 0.99
    particle_shape: str = "linear"       # "linear" | "quadratic" | "cubic"
    maxwell_solver: str = "yee"          # "yee" | "ckc" | "psatd" | "none"
    particle_pusher: str = "boris"       # "boris" | "vay" | "higuera"
    current_deposition: str = "esirkepov"  # "esirkepov" | "direct" | "vay"


def validate_em_solver(sol: EMSolverSpec) -> ValidationReport:
    r = ValidationReport()

    if sol.max_steps <= 0:
        r.add(Severity.ERROR, "em.max_steps",
              "max_steps must be > 0", max_steps=sol.max_steps)

    _VALID_MAXWELL = {"yee", "ckc", "psatd", "none"}
    if sol.maxwell_solver not in _VALID_MAXWELL:
        r.add(Severity.ERROR, "em.maxwell_solver",
              f"maxwell_solver must be one of {sorted(_VALID_MAXWELL)}",
              maxwell_solver=sol.maxwell_solver)

    _VALID_PUSHERS = {"boris", "vay", "higuera"}
    if sol.particle_pusher not in _VALID_PUSHERS:
        r.add(Severity.ERROR, "em.particle_pusher",
              f"particle_pusher must be one of {sorted(_VALID_PUSHERS)}",
              particle_pusher=sol.particle_pusher)
    elif sol.particle_pusher == "higuera":
        r.add(Severity.WARNING, "em.pusher.higuera_explicit",
              "The Higuera-Cary pusher is designed for implicit EM solvers; "
              "using it with an explicit FDTD solver is unusual")

    if sol.maxwell_solver != "psatd" and sol.cfl > 1.0:
        r.add(Severity.ERROR, "em.cfl",
              f"cfl={sol.cfl} > 1.0 violates the von Neumann stability limit for "
              f"FDTD solvers (Yee/CKC); set cfl <= 0.9 (or <= 1/sqrt(dim) for "
              f"the tightest safe value)",
              cfl=sol.cfl)

    _VALID_CURRENT_DEP = {"esirkepov", "direct", "vay"}
    if sol.current_deposition not in _VALID_CURRENT_DEP:
        r.add(Severity.ERROR, "em.current_deposition",
              f"current_deposition must be one of {sorted(_VALID_CURRENT_DEP)}",
              current_deposition=sol.current_deposition)

    return r


# ---------------------------------------------------------------------------
# ES-PIC solver block
# ---------------------------------------------------------------------------

@dataclass
class ESSolverSpec:
    """Field solver and particle pusher parameters for ES-PIC (Poisson-based).

    Poisson solver options:
      ``"multigrid"`` — algebraic multigrid (MLMG), works for any BCs (default)
      ``"fft"``       — spectral FFT Poisson, requires periodic BCs in all dims

    Particle pusher options:
      ``"boris"`` — standard Boris leap-frog (default)
      ``"vay"``   — Vay pusher

    ``const_dt`` sets the fixed timestep [s].  Unlike EM-PIC, the ES solver
    is not CFL-limited by the speed of light but must still resolve the plasma
    period: ``dt * omega_pe < 2`` (hard stability limit).
    """
    max_steps: int = 200
    particle_shape: str = "linear"
    poisson_solver: str = "multigrid"    # "multigrid" | "fft"
    particle_pusher: str = "boris"       # "boris" | "vay"
    poisson_precision: float = 1e-11    # MLMG relative tolerance
    const_dt: float = 1e-11             # fixed timestep [s]
    electrostatic_mode: str = "labframe"  # "labframe" | "relativistic"


def validate_es_solver(sol: ESSolverSpec) -> ValidationReport:
    r = ValidationReport()

    if sol.max_steps <= 0:
        r.add(Severity.ERROR, "es.max_steps",
              "max_steps must be > 0", max_steps=sol.max_steps)

    _VALID_POISSON = {"multigrid", "fft"}
    if sol.poisson_solver not in _VALID_POISSON:
        r.add(Severity.ERROR, "es.poisson_solver",
              f"poisson_solver must be one of {sorted(_VALID_POISSON)}",
              poisson_solver=sol.poisson_solver)

    _VALID_PUSHERS = {"boris", "vay"}
    if sol.particle_pusher not in _VALID_PUSHERS:
        r.add(Severity.ERROR, "es.particle_pusher",
              f"particle_pusher must be one of {sorted(_VALID_PUSHERS)} "
              f"(higuera is for implicit EM, not ES-PIC)",
              particle_pusher=sol.particle_pusher)

    if sol.const_dt <= 0:
        r.add(Severity.ERROR, "es.const_dt",
              "const_dt must be > 0", const_dt=sol.const_dt)

    if sol.poisson_precision <= 0:
        r.add(Severity.ERROR, "es.poisson_precision",
              "poisson_precision must be > 0", poisson_precision=sol.poisson_precision)

    _VALID_MODE = {"labframe", "relativistic"}
    if sol.electrostatic_mode not in _VALID_MODE:
        r.add(Severity.ERROR, "es.electrostatic_mode",
              f"electrostatic_mode must be one of {sorted(_VALID_MODE)}",
              electrostatic_mode=sol.electrostatic_mode)

    return r


# ---------------------------------------------------------------------------
# AMR block
# ---------------------------------------------------------------------------

#: Flat JSON key → AMRSpec attribute mapping (shared by all sim-type from_dict).
_AMR_KEY_MAP: dict = {
    "amr_max_level":      "max_level",
    "amr_ref_ratio":      "ref_ratio",
    "amr_max_grid_size":  "max_grid_size",
    "amr_blocking_factor": "blocking_factor",
    "amr_tag_by":         "tag_by",
    "fine_tag_lo":        "fine_tag_lo",
    "fine_tag_hi":        "fine_tag_hi",
    "ref_patch_function": "ref_patch_function",
}


@dataclass
class AMRSpec:
    """Adaptive mesh refinement parameters.

    When ``max_level == 0`` (default) the simulation runs on a single uniform
    grid and WarpX behaves exactly as before.  When ``max_level >= 1`` the
    generator emits the full AMR ParmParse block and requires either:

    * ``tag_by = "box"`` — static refinement of a physical sub-box;
      ``fine_tag_lo`` / ``fine_tag_hi`` must be provided (list of floats, one
      per spatial dimension, in metres).
    * ``tag_by = "plasma"`` — refine any cell containing macro-particles
      (``warpx.refine_plasma = 1``).
    * ``tag_by = "function"`` — analytic refinement region via WarpX parser
      expression; ``ref_patch_function`` must be provided.

    ``blocking_factor`` is applied when ``max_level > 0``; it must be a power
    of two and must divide ``number_of_cells`` in every dimension.  When
    ``max_level == 0`` the generator always uses ``amr.blocking_factor = 1``
    to preserve backward compatibility with arbitrary cell counts.

    ``ref_ratio`` (2 or 4) is isotropic and replicated once per level in the
    output (e.g. ``amr.ref_ratio = 2 2`` for ``max_level=2``).
    """
    max_level: int = 0
    ref_ratio: int = 2               # isotropic; one value per level in output
    max_grid_size: int = 128         # max AMReX patch size (load balancing)
    blocking_factor: int = 8         # used when max_level>0; 1 when max_level==0
    tag_by: str = "box"              # "box" | "plasma" | "function" | "none"
    fine_tag_lo: Optional[List[float]] = None   # physical coords; len == dim
    fine_tag_hi: Optional[List[float]] = None
    ref_patch_function: str = ""     # WarpX parser expr (tag_by="function")


def validate_amr(amr: AMRSpec, domain: DomainSpec) -> ValidationReport:
    """Validate AMR configuration against the domain.

    Checks:
    - max_level >= 0
    - ref_ratio in {2, 4}
    - blocking_factor is a power of 2
    - (when max_level > 0) n_cell divisible by blocking_factor per axis
    - (when max_level > 0, tag_by="box") fine_tag_lo/hi both provided
    - (when max_level > 0, tag_by="function") ref_patch_function non-empty
    - fine_tag_lo/hi lengths match domain.dim when provided
    - max_grid_size >= blocking_factor
    """
    r = ValidationReport()

    if amr.max_level < 0:
        r.add(Severity.ERROR, "amr.max_level",
              "max_level must be >= 0", max_level=amr.max_level)
        return r

    if amr.ref_ratio not in (2, 4):
        r.add(Severity.ERROR, "amr.ref_ratio",
              "ref_ratio must be 2 or 4", ref_ratio=amr.ref_ratio)

    bf = amr.blocking_factor
    if bf < 1 or (bf & (bf - 1)) != 0:
        r.add(Severity.ERROR, "amr.blocking_factor",
              "blocking_factor must be a power of 2", blocking_factor=bf)
    else:
        if amr.max_level > 0:
            bad = [n for n in domain.number_of_cells if n % bf != 0]
            if bad:
                r.add(Severity.ERROR, "amr.blocking_factor.divisibility",
                      f"number_of_cells must be divisible by blocking_factor={bf} "
                      f"in every dimension; offending counts: {bad}",
                      blocking_factor=bf, bad_cells=bad)

    if amr.max_grid_size < bf:
        r.add(Severity.ERROR, "amr.max_grid_size",
              "max_grid_size must be >= blocking_factor",
              max_grid_size=amr.max_grid_size, blocking_factor=bf)

    if amr.max_level > 0:
        if amr.tag_by == "box":
            if amr.fine_tag_lo is None or amr.fine_tag_hi is None:
                r.add(Severity.ERROR, "amr.fine_tag.missing",
                      "fine_tag_lo and fine_tag_hi are required when "
                      "max_level > 0 and tag_by='box'")
        elif amr.tag_by == "function":
            if not amr.ref_patch_function.strip():
                r.add(Severity.ERROR, "amr.ref_patch_function.missing",
                      "ref_patch_function must be set when "
                      "max_level > 0 and tag_by='function'")
        elif amr.tag_by == "none":
            r.add(Severity.WARNING, "amr.tag_by.none",
                  "max_level > 0 but tag_by='none': no cells will be refined")

    for name, lst in [("fine_tag_lo", amr.fine_tag_lo), ("fine_tag_hi", amr.fine_tag_hi)]:
        if lst is not None and len(lst) != domain.dim:
            r.add(Severity.ERROR, f"amr.{name}.len",
                  f"{name} must have length {domain.dim} (one entry per dim)",
                  got=len(lst), expected=domain.dim)

    return r


# ---------------------------------------------------------------------------
# Shared physics validation helpers (used across multiple sim-type validators)
# ---------------------------------------------------------------------------

_EPS0_PHYS = 8.854187817e-12
_Q_E_PHYS  = 1.602176634e-19
_M_E_PHYS  = 9.1093837015e-31
_M_P_PHYS  = 1.67262192369e-27
_C_PHYS    = 299792458.0
_RK4_WHISTLER_LIMIT = 2.0 * math.sqrt(2.0)   # ≈ 2.828


def check_const_dt_required(const_dt: float, r: ValidationReport, code_prefix: str) -> None:
    """ERROR if const_dt <= 0."""
    if const_dt <= 0:
        r.add(Severity.ERROR, f"{code_prefix}.const_dt",
              "const_dt must be > 0", const_dt=const_dt)


def check_debye_resolution(
    dx_max: float,
    density: float,
    temperature_eV: float,
    r: ValidationReport,
    code_prefix: str = "es",
    species_name: Optional[str] = None,
    error_above: float = 2.0,
) -> None:
    """Warn/error when the grid cannot resolve the electron Debye length.

    WARNING when 1 < dx/λ_De ≤ error_above; ERROR when dx/λ_De > error_above.
    λ_De = sqrt(ε₀·Te_eV / (n·q_e)).
    Set error_above=inf to emit WARNING only for all ratios > 1 (e.g. EM-PIC).
    """
    if density <= 0 or temperature_eV <= 0:
        return
    lam_De = math.sqrt(_EPS0_PHYS * temperature_eV / (density * _Q_E_PHYS))
    ratio = dx_max / lam_De
    if ratio <= 1.0:
        return
    sp_info = f"species '{species_name}', " if species_name else ""
    n_crit = _EPS0_PHYS * temperature_eV / (dx_max**2 * _Q_E_PHYS)
    base_msg = (
        f"dx={dx_max:.3e} m vs electron Debye length λ_De={lam_De:.3e} m "
        f"(dx/λ_De={ratio:.2f}; {sp_info}"
        f"n={density:.2e} m⁻³, Te={temperature_eV} eV). "
        f"ES-PIC requires dx ≲ λ_De to avoid finite-grid instability "
        f"(exponential numerical heating regardless of ppc). "
        f"Reduce density below {n_crit:.2e} m⁻³ so that λ_De ≥ dx, "
        f"or increase number_of_cells so that dx ≤ {lam_De:.3e} m."
    )
    kwargs: dict = dict(
        dx_max=round(dx_max, 9),
        lambda_De=round(lam_De, 9),
        dx_over_lambda_De=round(ratio, 4),
    )
    if species_name:
        kwargs["species"] = species_name
    if ratio > error_above:
        r.add(Severity.ERROR, f"{code_prefix}.debye_resolution",
              f"dx/λ_De={ratio:.1f} >> 1 — finite-grid instability guaranteed: " + base_msg,
              **kwargs)
    else:
        r.add(Severity.WARNING, f"{code_prefix}.debye_resolution",
              f"dx > λ_De — finite-grid instability risk: " + base_msg,
              **kwargs)


def check_boris_stability(
    const_dt: float,
    density: float,
    r: ValidationReport,
    code_prefix: str = "es",
    error_above: float = 2.0,
) -> None:
    """ERROR if dt·ω_pe ≥ error_above (explicit Boris unstable); WARNING if dt·ω_pe > 0.2.

    ω_pe = sqrt(n·q_e² / (m_e·ε₀)).
    Set error_above=inf to emit WARNING only for all marginal values (e.g. EM-PIC).
    """
    if density <= 0:
        return
    omega_pe = math.sqrt(density * _Q_E_PHYS**2 / (_M_E_PHYS * _EPS0_PHYS))
    dt_omega = const_dt * omega_pe
    if dt_omega >= error_above:
        r.add(
            Severity.ERROR, f"{code_prefix}.debye.dt_omega_pe",
            f"dt * omega_pe = {dt_omega:.3f} >= {error_above:.3g} — explicit Boris is "
            f"unconditionally unstable (max density: {density:.2e} m^-3, const_dt: {const_dt:.2e} s)",
            dt_omega_pe=dt_omega, n_max=density,
        )
    elif dt_omega > 0.2:
        r.add(
            Severity.WARNING, f"{code_prefix}.debye.dt_omega_pe",
            f"dt * omega_pe = {dt_omega:.3f} > 0.2 — accuracy may be reduced "
            f"(max density: {density:.2e} m^-3, const_dt: {const_dt:.2e} s)",
            dt_omega_pe=dt_omega, n_max=density,
        )


def check_whistler_cfl_hybrid(
    dx_min: float,
    b0_mag: float,
    n: float,
    mass_amu: float,
    const_dt: float,
    substeps: int,
    r: ValidationReport,
    code_prefix: str,
) -> None:
    """ERROR if the whistler CFL criterion is violated for hybrid-PIC.

    For explicit RK4 subcycling, the Nyquist whistler mode must satisfy:
        z = (π·l_i/dx)²·ω_ci·dt_sub < 2√2 ≈ 2.83
    where l_i = c/ω_pi is the ion skin depth, dt_sub = const_dt/substeps.
    """
    if b0_mag == 0.0 or n <= 0 or mass_amu <= 0:
        return
    m_i = mass_amu * _M_P_PHYS
    omega_ci = _Q_E_PHYS * b0_mag / m_i
    omega_pi = math.sqrt(n * _Q_E_PHYS**2 / (m_i * _EPS0_PHYS))
    l_i = _C_PHYS / omega_pi
    dt_sub = const_dt / substeps
    k_nyq_li = math.pi * l_i / dx_min
    z_max = k_nyq_li**2 * omega_ci * dt_sub
    if z_max > _RK4_WHISTLER_LIMIT:
        substeps_min = int(math.ceil(const_dt * omega_ci * k_nyq_li**2 / _RK4_WHISTLER_LIMIT))
        r.add(
            Severity.ERROR,
            f"{code_prefix}.cfl.whistler",
            (
                f"Whistler CFL z={z_max:.3g} > {_RK4_WHISTLER_LIMIT:.3g}: "
                f"sub-inertial Nyquist modes are numerically unstable. "
                f"l_i={l_i:.3e} m; dx={dx_min:.3e} m (dx/l_i={dx_min/l_i:.4f}). "
                f"Recommend dx <= l_i/10 = {l_i/10:.3e} m or substeps>={substeps_min}."
            ),
            z_whistler=round(z_max, 3),
            substeps_min=substeps_min,
            l_i=round(l_i, 6),
            dx=round(dx_min, 8),
        )


def check_fft_requires_periodic(
    poisson_solver: str,
    field_bc: List[str],
    r: ValidationReport,
    code_prefix: str = "es",
) -> None:
    """ERROR if poisson_solver='fft' but any field_bc is not 'periodic'."""
    if poisson_solver != "fft":
        return
    non_periodic = [bc for bc in field_bc if bc != "periodic"]
    if non_periodic:
        r.add(
            Severity.ERROR, f"{code_prefix}.fft_requires_periodic",
            f"poisson_solver='fft' requires all field_bc to be 'periodic'; "
            f"found non-periodic BCs: {non_periodic}. "
            f"Use poisson_solver='multigrid' for non-periodic boundaries.",
            non_periodic_bcs=non_periodic,
        )


def suggest_cells(
    lower_bound: List[float],
    upper_bound: List[float],
    dx_target: float,
    blocking_factor: int = 8,
) -> List[int]:
    """Suggest ``number_of_cells`` for a given physical resolution target.

    For each spatial axis the number of cells is computed as:
        n = ceil((hi - lo) / dx_target)
    then rounded up to the nearest multiple of ``blocking_factor`` (minimum
    ``blocking_factor``) so that AMReX will accept the grid without errors.

    Args:
        lower_bound: Physical lower bounds per axis [m].
        upper_bound: Physical upper bounds per axis [m].
        dx_target: Target cell size [m] (same for all axes).
        blocking_factor: AMReX blocking_factor (default 8, AMReX default).

    Returns:
        List of integer cell counts, one per axis.
    """
    cells: List[int] = []
    for lo, hi in zip(lower_bound, upper_bound):
        n = max(1, math.ceil((hi - lo) / dx_target))
        n = max(blocking_factor, blocking_factor * math.ceil(n / blocking_factor))
        cells.append(n)
    return cells


def _emit_amr_block(
    amr: AMRSpec,
    n_cell_str: str,
    prob_lo: str,
    prob_hi: str,
    dim: int,
) -> str:
    """Emit the AMR + geometry ParmParse lines.

    When ``amr.max_level == 0`` emits ``amr.blocking_factor = 1`` for backward
    compatibility (arbitrary n_cell is accepted).  When ``max_level >= 1``
    emits the full AMR block including refinement ratio, patch size, and
    tagging directives.

    Returns a string ready for insertion into a ParmParse inputs file.
    """
    # When max_level==0, keep blocking_factor=1 for backward compat
    bf = 1 if amr.max_level == 0 else amr.blocking_factor

    lines: List[str] = [
        f"amr.max_level = {amr.max_level}",
        f"amr.n_cell = {n_cell_str}",
        f"amr.blocking_factor = {bf}",
    ]

    if amr.max_level > 0:
        ref_ratio_str = " ".join(str(amr.ref_ratio) for _ in range(amr.max_level))
        lines.append(f"amr.max_grid_size = {amr.max_grid_size}")
        lines.append(f"amr.ref_ratio = {ref_ratio_str}")
        if amr.tag_by == "box" and amr.fine_tag_lo and amr.fine_tag_hi:
            lo_str = " ".join(f"{x:.17g}" for x in amr.fine_tag_lo)
            hi_str = " ".join(f"{x:.17g}" for x in amr.fine_tag_hi)
            lines.append(f"warpx.fine_tag_lo = {lo_str}")
            lines.append(f"warpx.fine_tag_hi = {hi_str}")
        elif amr.tag_by == "plasma":
            lines.append("warpx.refine_plasma = 1")
        elif amr.tag_by == "function" and amr.ref_patch_function.strip():
            lines.append(
                f"warpx.ref_patch_function(x,y,z) = {amr.ref_patch_function}"
            )

    lines.append("")   # blank line before geometry
    lines.append(f"geometry.dims = {dim}")
    lines.append(f"geometry.prob_lo = {prob_lo}")
    lines.append(f"geometry.prob_hi = {prob_hi}")

    return "\n".join(lines)


def _emit_checkpoint_block(checkpoint_int: Optional[int], checkpoint_file: str = "chk") -> str:
    """Emit AMReX checkpoint ParmParse lines.

    WarpX uses AMReX's checkpoint mechanism via the ``amr.*`` namespace:
      - ``amr.check_int``  — write a checkpoint every N steps
      - ``amr.check_file`` — checkpoint directory basename

    Returns an empty string when ``checkpoint_int`` is None or <= 0.
    """
    if not checkpoint_int or checkpoint_int <= 0:
        return ""
    return (
        f"# --- Checkpoint (AMReX) -----------------------------------------------\n"
        f"amr.check_int = {checkpoint_int}\n"
        f"amr.check_file = {checkpoint_file}"
    )


def _emit_diag_block(diag: DiagSpec, name: str = "diag1") -> str:
    """Emit the full diagnostics ParmParse block for a native (binary) generator.

    Emits the main "Full" field diagnostic block plus any reduced diagnostics
    listed in ``diag.reduced_diags``.

    Returns a string ready for insertion into a ParmParse inputs file.
    """
    fields_to_plot = " ".join(diag.diag_fields)
    write_sp = 1 if diag.write_species else 0

    lines: List[str] = [
        f"diagnostics.diags_names = {name}",
        f"{name}.diag_type = Full",
        f"{name}.intervals = {diag.diag_period}",
        f"{name}.format = {diag.diag_format}",
        f"{name}.fields_to_plot = {fields_to_plot}",
        f"{name}.write_species = {write_sp}",
    ]

    if diag.reduced_diags:
        rd_names = [rd.name or rd.type.lower() for rd in diag.reduced_diags]
        lines.append("")
        lines.append(f"warpx.reduced_diags_names = {' '.join(rd_names)}")
        for rd, rd_name in zip(diag.reduced_diags, rd_names):
            interval = rd.period if rd.period > 0 else diag.diag_period
            lines.append(f"{rd_name}.type = {rd.type}")
            lines.append(f"{rd_name}.intervals = {interval}")
            lines.append(f"{rd_name}.path = {rd.path}")
            if rd.type == "FieldProbe":
                lines.append(f"{rd_name}.probe_geometry = {rd.probe_geometry}")
                lines.append(f"{rd_name}.x_probe = {rd.x_probe}")
                lines.append(f"{rd_name}.y_probe = {rd.y_probe}")
                lines.append(f"{rd_name}.z_probe = {rd.z_probe}")
                if rd.probe_geometry in ("Line", "Plane"):
                    lines.append(f"{rd_name}.z1_probe = {rd.z1_probe}")
                    lines.append(f"{rd_name}.resolution = {rd.resolution}")
                if rd.interp_order != 1:
                    lines.append(f"{rd_name}.interp_order = {rd.interp_order}")
                if rd.integrate:
                    lines.append(f"{rd_name}.integrate = 1")
            elif rd.type == "FieldReduction":
                lines.append(f"{rd_name}.reduction_type = {rd.reduction_type}")
                lines.append(
                    f"{rd_name}.reduced_function(x,y,z,Ex,Ey,Ez,Bx,By,Bz,jx,jy,jz)"
                    f" = {rd.reduced_function}"
                )
            elif rd.type in _KNOWN_REDUCED_SPECIES:
                lines.append(f"{rd_name}.species = {rd.species}")
            elif rd.type == "ParticleHistogram":
                lines.append(f"{rd_name}.species = {rd.species}")
                lines.append(f"{rd_name}.bin_number = {rd.bin_number}")
                lines.append(f"{rd_name}.bin_min = {rd.bin_min}")
                lines.append(f"{rd_name}.bin_max = {rd.bin_max}")
                lines.append(f"{rd_name}.histogram_function(t,x,y,z,ux,uy,uz)"
                              f" = {rd.histogram_function}")
                if rd.normalization:
                    lines.append(f"{rd_name}.normalization = {rd.normalization}")
                if rd.filter_function.strip():
                    lines.append(f"{rd_name}.filter_function(t,x,y,z,ux,uy,uz)"
                                  f" = {rd.filter_function}")
            elif rd.type == "ChargeOnEB" and rd.weighting_function.strip():
                lines.append(f"{rd_name}.weighting_function(x,y,z)"
                              f" = {rd.weighting_function}")

    return "\n".join(lines)


def _emit_picmi_diag_lines(
    diag: DiagSpec,
    species_var_names: Optional[List[str]] = None,
) -> str:
    """Return Python source lines for diagnostics in a generated PICMI script.

    The returned string contains:
    - A ``picmi.FieldDiagnostic`` creation and ``sim.add_diagnostic(...)`` call.
    - If ``diag.write_species`` is True and ``species_var_names`` is provided,
      a ``picmi.ParticleDiagnostic`` block for all species.
    - A ``picmi.ReducedDiagnostic`` block for each entry in ``diag.reduced_diags``.

    The returned code assumes ``grid``, ``sim``, and all species variables are
    already defined in the surrounding script context.
    """
    if species_var_names is None:
        species_var_names = []

    parts: List[str] = ["# -- Diagnostics --"]

    # Main field diagnostic
    parts.append(
        f"field_diag = picmi.FieldDiagnostic(\n"
        f"    name='field_diag',\n"
        f"    grid=grid,\n"
        f"    period={diag.diag_period},\n"
        f"    data_list={diag.diag_fields!r},\n"
        f"    warpx_format={diag.diag_format!r},\n"
        f")"
    )
    parts.append("sim.add_diagnostic(field_diag)")

    # Particle diagnostic (if requested)
    if diag.write_species and species_var_names:
        sp_list = ", ".join(species_var_names)
        parts.append(
            f"ptcl_diag = picmi.ParticleDiagnostic(\n"
            f"    name='ptcl_diag',\n"
            f"    period={diag.diag_period},\n"
            f"    species=[{sp_list}],\n"
            f"    warpx_format={diag.diag_format!r},\n"
            f")"
        )
        parts.append("sim.add_diagnostic(ptcl_diag)")

    # Reduced diagnostics
    for rd in diag.reduced_diags:
        rd_var = rd.name or rd.type.lower()
        rd_period = rd.period if rd.period > 0 else diag.diag_period
        kw_lines = [
            f"    diag_type={rd.type!r},",
            f"    name={rd_var!r},",
            f"    period={rd_period},",
            f"    path={rd.path!r},",
        ]
        if rd.type == "FieldProbe":
            kw_lines.append(f"    probe_geometry={rd.probe_geometry!r},")
            kw_lines.append(f"    x_probe={rd.x_probe!r},")
            kw_lines.append(f"    y_probe={rd.y_probe!r},")
            kw_lines.append(f"    z_probe={rd.z_probe!r},")
            if rd.probe_geometry in ("Line", "Plane"):
                kw_lines.append(f"    z1_probe={rd.z1_probe!r},")
                kw_lines.append(f"    resolution={rd.resolution!r},")
            if rd.interp_order != 1:
                kw_lines.append(f"    interp_order={rd.interp_order!r},")
            if rd.integrate:
                kw_lines.append("    integrate=True,")
        elif rd.type == "FieldReduction":
            kw_lines.append(f"    reduction_type={rd.reduction_type!r},")
            kw_lines.append(f"    reduced_function={rd.reduced_function!r},")
        elif rd.type in _KNOWN_REDUCED_SPECIES:
            kw_lines.append(f"    species={rd.species},  # pass the species object")
        elif rd.type == "ParticleHistogram":
            kw_lines.append(f"    species={rd.species},  # pass the species object")
            kw_lines.append(f"    bin_number={rd.bin_number!r},")
            kw_lines.append(f"    bin_min={rd.bin_min!r},")
            kw_lines.append(f"    bin_max={rd.bin_max!r},")
            kw_lines.append(f"    histogram_function={rd.histogram_function!r},")
            if rd.normalization:
                kw_lines.append(f"    normalization={rd.normalization!r},")
            if rd.filter_function.strip():
                kw_lines.append(f"    filter_function={rd.filter_function!r},")
        elif rd.type == "ChargeOnEB" and rd.weighting_function.strip():
            kw_lines.append(f"    weighting_function={rd.weighting_function!r},")
        kw_str = "\n".join(kw_lines)
        parts.append(
            f"{rd_var} = picmi.ReducedDiagnostic(\n{kw_str}\n)"
        )
        parts.append(f"sim.add_diagnostic({rd_var})")

    return "\n".join(parts)
