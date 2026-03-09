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
        r.add(Severity.WARNING, "em.cfl",
              "cfl > 1.0 is likely unstable for FDTD solvers (Yee/CKC)",
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
