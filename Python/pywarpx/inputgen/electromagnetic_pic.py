from __future__ import annotations

"""Broad-scope electromagnetic PIC (EM-FDTD/spectral) input generation.

Produces native AMReX ParmParse inputs for a general multi-species EM-PIC run.
Supports arbitrary species lists, optional physics blocks (field ionization,
QED, nuclear fusion reactions, Coulomb collisions, classical radiation reaction),
flexible Maxwell solver and particle pusher selection, optional laser injection,
optional moving window, and asymmetric lo/hi boundary conditions.

Use this sim type for:
  - Multi-species plasma (collisionless shocks, TNSA, beam–plasma interaction)
  - Laser-solid/laser-plasma with field ionization
  - High-power laser with QED effects (Breit-Wheeler, quantum synchrotron)
  - Nuclear fusion (D-T, D-He3) in the EM-PIC regime
  - Beam-driven wakefields (PWFA variants) with custom species

Physical constants (CODATA 2018):
    m_e = 9.1093837015e-31 kg
    m_p = 1.67262192369e-27 kg
    q_e = 1.602176634e-19 C
    c   = 299792458.0 m/s
"""

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .blocks import (
    AMRSpec,
    CollisionSpec,
    DiagSpec,
    DomainSpec,
    EBSpec,
    EMSolverSpec,
    ExtBFieldSpec,
    ImplicitSolverSpec,
    LaserSpec,
    SpeciesDefSpec,
    _AMR_KEY_MAP,
    _emit_amr_block,
    _emit_diag_block,
    _emit_picmi_diag_lines,
    suggest_cells,
)

_M_E = 9.1093837015e-31
_M_P = 1.67262192369e-27
_Q_E = 1.602176634e-19
_C   = 299792458.0
_AMU = 1.66053906660e-27

# Electron mass in AMU (used for symbolic mass detection)
_M_E_AMU = _M_E / _AMU    # ≈ 5.486e-4
_M_P_AMU = _M_P / _AMU    # ≈ 1.00728

_DOMAIN_KEYS = {"dim", "number_of_cells", "lower_bound", "upper_bound", "field_bc"}
_EM_SOLVER_KEYS = {
    "max_steps", "cfl", "particle_shape", "maxwell_solver",
    "particle_pusher", "current_deposition",
}
_EB_KEYS = {"eb_implicit_function", "eb_potential", "stl_file"}
_EXT_BFIELD_KEYS = {"Bx_expression", "By_expression", "Bz_expression"}
_LASER_KEYS = {
    "wavelength", "a0", "waist", "duration",
    "focal_position_z", "centroid_position_z",
}
_IMPLICIT_KEY_MAP = {
    "implicit_enabled": "enabled",
    "implicit_theta": "theta",
    "implicit_solver_type": "solver_type",
    "implicit_max_iters": "max_iters",
    "implicit_tolerance": "tolerance",
    "implicit_const_dt": "const_dt",
}


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ElectromagneticPICSpec:
    """Specification for a broad-scope EM-PIC simulation.

    All physics blocks are optional.  The minimal required configuration is
    a domain spec and at least one species in the species list.

    Field BC options (per axis): ``"periodic"``, ``"pml"`` (absorbing),
    ``"pec"`` (perfect electric conductor), ``"pmc"``, ``"open"``.

    ``field_bc_lo`` / ``field_bc_hi`` override ``domain.field_bc`` for
    independent lo/hi boundary conditions (e.g. PML on hi but periodic on lo).
    When ``None`` (default), ``domain.field_bc`` is used for both sides.

    Nuclear fusion example:
      species = [
          {"name":"deuterium","charge":1,"mass_amu":2.014,"density":1e26,"ppc":4},
          {"name":"tritium","charge":1,"mass_amu":3.016,"density":1e26,"ppc":4},
          {"name":"helium4","charge":2,"mass_amu":4.003,"injection_style":"none"},
          {"name":"neutron","charge":0,"mass_amu":1.009,"injection_style":"none"},
      ]
      collisions = [{"name":"dt_fusion","type":"nuclearfusion",
                     "species":["deuterium","tritium"],
                     "product_species":["helium4","neutron"],
                     "event_multiplier":1e13}]
    """
    name: str = "electromagnetic_pic"
    domain: DomainSpec = field(
        default_factory=lambda: DomainSpec(
            dim=2,
            number_of_cells=[128, 256],
            lower_bound=[-50e-6, 0.0],
            upper_bound=[50e-6, 200e-6],
            field_bc=["periodic", "pml"],
        )
    )
    solver: EMSolverSpec = field(default_factory=EMSolverSpec)
    species: List[SpeciesDefSpec] = field(default_factory=list)
    collisions: List[CollisionSpec] = field(default_factory=list)
    diag: DiagSpec = field(
        default_factory=lambda: DiagSpec(
            diag_period=50,
            diag_fields=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
        )
    )
    # Asymmetric BCs: None = use domain.field_bc for both lo and hi
    field_bc_lo: Optional[List[str]] = None
    field_bc_hi: Optional[List[str]] = None
    # Optional laser injection
    laser: Optional[LaserSpec] = None
    laser_direction: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    laser_polarization: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    # Optional moving window (follows driver at v=c along moving_window_direction)
    moving_window: bool = False
    moving_window_direction: str = "z"
    # Optional analytic external B-field
    ext_bfield: Optional[ExtBFieldSpec] = None
    # Optional embedded boundary
    eb: Optional[EBSpec] = None
    # Optional implicit EM solver
    implicit: ImplicitSolverSpec = field(default_factory=ImplicitSolverSpec)
    # AMR / resolution
    amr: AMRSpec = field(default_factory=AMRSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "ElectromagneticPICSpec":
        """Construct from a flat/nested JSON dict (borealis-mcp spec format).

        Scalar/list keys follow the flat convention used by all sim types.
        The ``species`` and ``collisions`` keys are nested lists of dicts.
        """
        amr = AMRSpec(**{attr: d[flat] for flat, attr in _AMR_KEY_MAP.items() if flat in d})
        if "dx_target" in d and "number_of_cells" not in d:
            if "lower_bound" in d and "upper_bound" in d:
                d = dict(d)
                d["number_of_cells"] = suggest_cells(
                    d["lower_bound"], d["upper_bound"], d["dx_target"], amr.blocking_factor
                )
        domain = DomainSpec(**{k: d[k] for k in _DOMAIN_KEYS if k in d})
        solver = EMSolverSpec(**{k: d[k] for k in _EM_SOLVER_KEYS if k in d})
        diag = DiagSpec.from_dict(d)

        species = [SpeciesDefSpec(**e) for e in d.get("species", [])]
        collisions = [CollisionSpec(**e) for e in d.get("collisions", [])]

        imp_kw: dict = {}
        for flat, attr in _IMPLICIT_KEY_MAP.items():
            if flat in d:
                imp_kw[attr] = d[flat]
        implicit = ImplicitSolverSpec(**imp_kw)

        laser = None
        if any(k in d for k in _LASER_KEYS):
            laser = LaserSpec(**{k: d[k] for k in _LASER_KEYS if k in d})

        ext_bfield = None
        if any(k in d for k in _EXT_BFIELD_KEYS):
            ext_bfield = ExtBFieldSpec(**{k: d[k] for k in _EXT_BFIELD_KEYS if k in d})

        eb = None
        if any(k in d for k in _EB_KEYS):
            eb = EBSpec(**{k: d[k] for k in _EB_KEYS if k in d})

        return cls(
            name=d.get("name", "electromagnetic_pic"),
            domain=domain,
            solver=solver,
            species=species,
            collisions=collisions,
            diag=diag,
            field_bc_lo=d.get("field_bc_lo"),
            field_bc_hi=d.get("field_bc_hi"),
            laser=laser,
            laser_direction=d.get("laser_direction", [0.0, 0.0, 1.0]),
            laser_polarization=d.get("laser_polarization", [1.0, 0.0, 0.0]),
            moving_window=d.get("moving_window", False),
            moving_window_direction=d.get("moving_window_direction", "z"),
            ext_bfield=ext_bfield,
            eb=eb,
            implicit=implicit,
            amr=amr,
        )


# ---------------------------------------------------------------------------
# Private generator helpers
# ---------------------------------------------------------------------------

def _charge_str(charge: float) -> str:
    """Return a WarpX-compatible charge expression."""
    if charge == -1.0:
        return "-q_e"
    if charge == 1.0:
        return "q_e"
    if charge == 0.0:
        return "0.0"
    return f"{charge:.17g}*q_e"


def _mass_str(mass_amu: float) -> str:
    """Return a WarpX-compatible mass expression (SI or symbolic)."""
    if abs(mass_amu - _M_E_AMU) / _M_E_AMU < 5e-3:
        return "m_e"
    if abs(mass_amu - _M_P_AMU) / _M_P_AMU < 5e-3:
        return "m_p"
    return f"{mass_amu * _AMU:.17g}"


def _particle_bc_from_field_bc(bc: str) -> str:
    """Map a field BC keyword to the corresponding particle BC keyword."""
    _MAP = {
        "pml": "absorbing",
        "open": "absorbing",
        "absorbing": "absorbing",
        "periodic": "periodic",
        "pec": "reflecting",
        "pmc": "reflecting",
        "none": "none",
        "damped": "absorbing",
    }
    return _MAP.get(bc, bc)


def _emit_species_block(sp: SpeciesDefSpec) -> str:
    """Emit the ParmParse block for a single species."""
    lines: List[str] = []
    n = sp.name

    lines.append(f"{n}.charge = {_charge_str(sp.charge)}")
    lines.append(f"{n}.mass = {_mass_str(sp.mass_amu)}")

    style = sp.injection_style

    if style == "none":
        lines.append(f"{n}.injection_style = none")

    elif style == "gaussian_beam":
        lines.append(f"{n}.injection_style = gaussian_beam")
        lines.append(f"{n}.x_rms = {sp.x_rms:.17g}")
        lines.append(f"{n}.y_rms = {sp.y_rms:.17g}")
        lines.append(f"{n}.z_rms = {sp.z_rms:.17g}")
        lines.append(f"{n}.z_cut = {sp.z_cut:.17g}")
        lines.append(f"{n}.npart = {sp.n_macro}")
        lines.append(f"{n}.q_tot = {sp.q_tot:.17g}")
        lines.append(f"{n}.z_m = {sp.z_mean:.17g}")
        lines.append(f"{n}.momentum_distribution_type = gaussian")
        lines.append(f"{n}.ux_m = {sp.ux_m:.17g}")
        lines.append(f"{n}.uy_m = {sp.uy_m:.17g}")
        lines.append(f"{n}.uz_m = {sp.uz_m:.17g}")
        lines.append(f"{n}.ux_th = {sp.ux_th:.17g}")
        lines.append(f"{n}.uy_th = {sp.uy_th:.17g}")
        lines.append(f"{n}.uz_th = {sp.uz_th:.17g}")

    else:  # NRandomPerCell / NUniformPerCell
        lines.append(f"{n}.injection_style = {style}")
        if style == "NUniformPerCell" and sp.ppc_each_dim is not None:
            dims_str = " ".join(str(x) for x in sp.ppc_each_dim)
            lines.append(f"{n}.num_particles_per_cell_each_dim = {dims_str}")
        else:
            lines.append(f"{n}.num_particles_per_cell = {sp.ppc}")

        if sp.profile == "parse_density_function" and sp.density_function.strip():
            lines.append(f"{n}.profile = parse_density_function")
            lines.append(f"{n}.density_function(x,y,z) = {sp.density_function}")
        else:
            lines.append(f"{n}.profile = constant")
            lines.append(f"{n}.density = {sp.density:.17g}")

        for attr in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            val = getattr(sp, attr)
            if val is not None:
                lines.append(f"{n}.{attr} = {val:.17g}")

        # Momentum distribution
        if sp.temperature_eV > 0:
            mass_kg = sp.mass_amu * _AMU
            u_th = math.sqrt(sp.temperature_eV * _Q_E / mass_kg) / _C
            lines.append(f"{n}.momentum_distribution_type = gaussian")
            lines.append(f"{n}.ux_m = {sp.ux_m:.17g}")
            lines.append(f"{n}.uy_m = {sp.uy_m:.17g}")
            lines.append(f"{n}.uz_m = {sp.uz_m:.17g}")
            lines.append(f"{n}.ux_th = {u_th:.17g}")
            lines.append(f"{n}.uy_th = {u_th:.17g}")
            lines.append(f"{n}.uz_th = {u_th:.17g}")
        elif sp.ux_th or sp.uy_th or sp.uz_th or sp.ux_m or sp.uy_m or sp.uz_m:
            lines.append(f"{n}.momentum_distribution_type = gaussian")
            lines.append(f"{n}.ux_m = {sp.ux_m:.17g}")
            lines.append(f"{n}.uy_m = {sp.uy_m:.17g}")
            lines.append(f"{n}.uz_m = {sp.uz_m:.17g}")
            lines.append(f"{n}.ux_th = {sp.ux_th:.17g}")
            lines.append(f"{n}.uy_th = {sp.uy_th:.17g}")
            lines.append(f"{n}.uz_th = {sp.uz_th:.17g}")
        else:
            lines.append(f"{n}.momentum_distribution_type = constant")
            lines.append(f"{n}.ux_m = 0.0")
            lines.append(f"{n}.uy_m = 0.0")
            lines.append(f"{n}.uz_m = 0.0")

    if sp.do_continuous_injection:
        lines.append(f"{n}.do_continuous_injection = 1")

    # --- Field ionization ---
    if sp.do_field_ionization:
        lines.append(f"{n}.do_field_ionization = 1")
        lines.append(f"{n}.physical_element = {sp.physical_element}")
        lines.append(f"{n}.ionization_initial_level = {sp.ionization_initial_level}")
        lines.append(f"{n}.ionization_product_species = {sp.ionization_product_species}")

    # --- QED ---
    if sp.do_qed_breit_wheeler:
        lines.append(f"{n}.do_qed_breit_wheeler = 1")
        lines.append(f"{n}.qed_breit_wheeler_ele_product_species = {sp.qed_bw_ele_product}")
        lines.append(f"{n}.qed_breit_wheeler_pos_product_species = {sp.qed_bw_pos_product}")

    if sp.do_qed_quantum_sync:
        lines.append(f"{n}.do_qed_quantum_sync = 1")
        lines.append(f"{n}.qed_quantum_sync_phot_product_species = {sp.qed_qs_phot_product}")

    # --- Classical radiation reaction ---
    if sp.do_classical_radiation_reaction:
        lines.append(f"{n}.do_classical_radiation_reaction = 1")

    return "\n".join(lines)


def _emit_collision_block(col: CollisionSpec) -> str:
    """Emit the ParmParse block for a single collision/reaction."""
    lines: List[str] = []
    n = col.name
    species_str = " ".join(col.species)
    lines.append(f"{n}.type = {col.type}")
    lines.append(f"{n}.species = {species_str}")
    if col.type == "nuclearfusion":
        prod_str = " ".join(col.product_species)
        lines.append(f"{n}.product_species = {prod_str}")
        lines.append(f"{n}.event_multiplier = {col.event_multiplier:.17g}")
        if col.probability_target_value > 0:
            lines.append(f"{n}.probability_target_value = {col.probability_target_value:.17g}")
    elif col.type == "coulomb":
        if col.CoulombLog > 0:
            lines.append(f"{n}.CoulombLog = {col.CoulombLog:.17g}")
    return "\n".join(lines)


def _bc_list(spec_bc: Optional[List[str]], domain_bc: List[str]) -> List[str]:
    """Return resolved BC list: spec_bc if given, else domain_bc."""
    return spec_bc if spec_bc is not None else domain_bc


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_inputs_electromagnetic_pic(spec: ElectromagneticPICSpec) -> str:
    """Generate a native WarpX AMReX ParmParse inputs file for EM-PIC.

    All optional blocks are emitted only when configured.  The output is
    deterministic and can be passed directly to the WarpX binary.
    """
    if spec.domain.dim not in (1, 2, 3):
        raise ValueError(
            f"ElectromagneticPICSpec supports dim=1, 2, or 3; got {spec.domain.dim}"
        )

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    n_cell  = " ".join(str(n) for n in spec.domain.number_of_cells)
    prob_lo = " ".join(f"{x:.17g}" for x in spec.domain.lower_bound)
    prob_hi = " ".join(f"{x:.17g}" for x in spec.domain.upper_bound)

    field_lo = _bc_list(spec.field_bc_lo, spec.domain.field_bc)
    field_hi = _bc_list(spec.field_bc_hi, spec.domain.field_bc)
    bc_field_lo_str = " ".join(field_lo)
    bc_field_hi_str = " ".join(field_hi)
    bc_part_lo_str  = " ".join(_particle_bc_from_field_bc(b) for b in field_lo)
    bc_part_hi_str  = " ".join(_particle_bc_from_field_bc(b) for b in field_hi)

    # ------------------------------------------------------------------
    # Solver / numerics
    # ------------------------------------------------------------------
    imp = spec.implicit
    _PSHAPE_MAP = {"linear": 1, "quadratic": 2, "cubic": 3}
    pshape = spec.solver.particle_shape
    if isinstance(pshape, str):
        pshape = _PSHAPE_MAP.get(pshape.lower(), 1)

    if imp.enabled:
        timestep_line = f"warpx.const_dt = {imp.const_dt:.17g}"
        implicit_block = f"""\
# --- Implicit EM solver (theta = {imp.theta}) ---------------------------------
algo.evolve_scheme = theta_implicit_em
warpx.implicit_solver.solver_type = {imp.solver_type}
warpx.implicit_solver.max_iters = {imp.max_iters}
warpx.implicit_solver.relative_tolerance = {imp.tolerance:.17g}
warpx.implicit_solver.theta = {imp.theta:.17g}
"""
    else:
        timestep_line = f"warpx.cfl = {spec.solver.cfl:.17g}"
        implicit_block = ""

    # ------------------------------------------------------------------
    # Species
    # ------------------------------------------------------------------
    species_names = " ".join(sp.name for sp in spec.species)
    species_blocks = "\n\n".join(
        f"# --- Species: {sp.name} ---\n{_emit_species_block(sp)}"
        for sp in spec.species
    )

    # ------------------------------------------------------------------
    # Collisions
    # ------------------------------------------------------------------
    collision_section = ""
    if spec.collisions:
        col_names = " ".join(c.name for c in spec.collisions)
        col_blocks = "\n\n".join(_emit_collision_block(c) for c in spec.collisions)
        collision_section = f"""\

# --- Collisions / reactions --------------------------------------------------
collisions.collision_names = {col_names}

{col_blocks}
"""

    # ------------------------------------------------------------------
    # Laser (optional)
    # ------------------------------------------------------------------
    laser_section = ""
    if spec.laser is not None:
        la = spec.laser
        # Antenna position: first cell above z_lo
        antenna_z = spec.domain.lower_bound[-1] + 1.0e-12
        focal_z   = la.focal_position_z
        # Peak intensity envelope time
        t_peak    = (la.centroid_position_z - spec.domain.lower_bound[-1]) / _C
        omega_0   = 2.0 * math.pi * _C / la.wavelength
        e_max     = la.a0 * _M_E * _C * omega_0 / _Q_E

        dir_str   = " ".join(f"{v:.17g}" for v in spec.laser_direction)
        pol_str   = " ".join(f"{v:.17g}" for v in spec.laser_polarization)

        laser_section = f"""\

# --- Laser injection ---------------------------------------------------------
lasers.names = laser1
laser1.position = 0.0 0.0 {antenna_z:.17g}
laser1.direction = {dir_str}
laser1.polarization = {pol_str}
laser1.profile = Gaussian
laser1.wavelength = {la.wavelength:.17g}
laser1.e_max = {e_max:.17g}
laser1.profile_waist = {la.waist:.17g}
laser1.profile_duration = {la.duration:.17g}
laser1.profile_t_peak = {t_peak:.17g}
laser1.profile_focal_distance = {focal_z - antenna_z:.17g}
"""

    # ------------------------------------------------------------------
    # Moving window (optional)
    # ------------------------------------------------------------------
    moving_window_section = ""
    if spec.moving_window:
        moving_window_section = f"""\

# --- Moving window -----------------------------------------------------------
warpx.do_moving_window = 1
warpx.moving_window_direction = {spec.moving_window_direction}
warpx.moving_window_v = 1.0  # in units of c
"""

    # ------------------------------------------------------------------
    # External B-field (optional)
    # ------------------------------------------------------------------
    ext_bfield_section = ""
    if spec.ext_bfield is not None:
        bf = spec.ext_bfield
        if bf.Bx_expression.strip():
            ext_bfield_section = f"""\

# --- Analytic external B-field -----------------------------------------------
warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = {bf.Bx_expression}
warpx.By_external_grid_function(x,y,z) = {bf.By_expression}
warpx.Bz_external_grid_function(x,y,z) = {bf.Bz_expression}
"""

    # ------------------------------------------------------------------
    # Embedded boundary (optional)
    # ------------------------------------------------------------------
    eb_section = ""
    if spec.eb is not None:
        eb = spec.eb
        if eb.eb_implicit_function.strip():
            eb_section = f"""\

# --- Embedded boundary (implicit function) -----------------------------------
warpx.eb_implicit_function = {eb.eb_implicit_function}
"""
            if eb.eb_potential.strip():
                eb_section += f"warpx.eb_potential(x,y,z,t) = {eb.eb_potential}\n"
        elif eb.stl_file.strip():
            eb_section = f"""\

# --- Embedded boundary (STL) -------------------------------------------------
eb2.geom_type = stl
eb2.stl_file = {eb.stl_file}
"""
            if eb.eb_potential.strip():
                eb_section += f"warpx.eb_potential(x,y,z,t) = {eb.eb_potential}\n"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diag_section = _emit_diag_block(spec.diag)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    d = {
        "name": spec.name,
        "domain": asdict(spec.domain),
        "solver": asdict(spec.solver),
        "species": [asdict(sp) for sp in spec.species],
        "collisions": [asdict(c) for c in spec.collisions],
        "diag": asdict(spec.diag),
        "field_bc_lo": spec.field_bc_lo,
        "field_bc_hi": spec.field_bc_hi,
        "moving_window": spec.moving_window,
    }

    return f"""\
# Auto-generated by pywarpx.inputgen (electromagnetic_pic)
# SPEC = {d!r}

# --- Time --------------------------------------------------------------------
max_step = {spec.solver.max_steps}

# --- AMR / domain ------------------------------------------------------------
{_emit_amr_block(spec.amr, n_cell, prob_lo, prob_hi, spec.domain.dim)}

# --- Boundary conditions -----------------------------------------------------
boundary.field_lo = {bc_field_lo_str}
boundary.field_hi = {bc_field_hi_str}
boundary.particle_lo = {bc_part_lo_str}
boundary.particle_hi = {bc_part_hi_str}
{moving_window_section}
# --- Maxwell / particle solver -----------------------------------------------
warpx.verbose = 1
{timestep_line}
algo.maxwell_solver = {spec.solver.maxwell_solver}
algo.particle_shape = {pshape}
algo.particle_pusher = {spec.solver.particle_pusher}
algo.current_deposition = {spec.solver.current_deposition}

{implicit_block}\
# --- Species -----------------------------------------------------------------
particles.species_names = {species_names}

{species_blocks}
{collision_section}\
{laser_section}\
{ext_bfield_section}\
{eb_section}\
# --- Diagnostics -------------------------------------------------------------
{diag_section}
"""


_GRID_CLASS_EM = {1: "Cartesian1DGrid", 2: "Cartesian2DGrid", 3: "Cartesian3DGrid"}


def generate_picmi_electromagnetic_pic(spec: ElectromagneticPICSpec) -> str:
    """Generate a PICMI script for an EM-PIC simulation.

    Species momentum: if temperature_eV > 0, uses gaussian thermal distribution;
    if injection_style is 'gaussian_beam', uses GaussianBunchDistribution;
    otherwise constant (cold) distribution.
    """
    grid_class = _GRID_CLASS_EM[spec.domain.dim]
    sp_var_names = [sp.name for sp in spec.species if sp.injection_style != "none"]
    diag_code = _emit_picmi_diag_lines(spec.diag, sp_var_names)

    # Build species PICMI code fragment
    species_lines = []
    for sp in spec.species:
        n = sp.name
        if sp.injection_style == "none":
            # Product/secondary species — no initial injection
            species_lines.append(
                f"# {n}: secondary species (no initial injection)\n"
                f"{n} = picmi.Species(name={n!r}, charge_state={sp.charge!r}, mass={sp.mass_amu * _AMU!r})"
            )
        elif sp.injection_style == "gaussian_beam":
            dist_var = f"{n}_dist"
            species_lines.append(
                f"{dist_var} = picmi.GaussianBunchDistribution(\n"
                f"    n_physical_particles=abs({sp.q_tot!r} / picmi.constants.q_e),\n"
                f"    rms_bunch_size=[{sp.x_rms!r}, {sp.y_rms!r}, {sp.z_rms!r}],\n"
                f"    bunch_rms_velocity=[0.0, 0.0, {sp.uz_th!r}],\n"
                f"    directed_velocity=[0.0, 0.0, {sp.uz_m!r}],\n"
                f"    centroid_position=[0.0, 0.0, {sp.z_mean!r}],\n"
                f")\n"
                f"{n} = picmi.Species(name={n!r}, charge_state={sp.charge!r}, mass={sp.mass_amu * _AMU!r},\n"
                f"    initial_distribution={dist_var})"
            )
        else:
            dist_var = f"{n}_dist"
            if sp.temperature_eV > 0:
                mass_kg = sp.mass_amu * _AMU
                u_th = math.sqrt(sp.temperature_eV * _Q_E / mass_kg) / _C
                species_lines.append(
                    f"{dist_var} = picmi.UniformDistribution(\n"
                    f"    density={sp.density!r}, fill_in=True,\n"
                    f"    rms_velocity=[{u_th!r}, {u_th!r}, {u_th!r}],\n"
                    f"    directed_velocity=[{sp.ux_m!r}, {sp.uy_m!r}, {sp.uz_m!r}],\n"
                    f")\n"
                    f"{n} = picmi.Species(name={n!r}, charge_state={sp.charge!r}, mass={sp.mass_amu * _AMU!r},\n"
                    f"    initial_distribution={dist_var})"
                )
            else:
                species_lines.append(
                    f"{dist_var} = picmi.UniformDistribution(density={sp.density!r}, fill_in=True)\n"
                    f"{n} = picmi.Species(name={n!r}, charge_state={sp.charge!r}, mass={sp.mass_amu * _AMU!r},\n"
                    f"    initial_distribution={dist_var})"
                )

    species_code = "\n\n".join(species_lines)

    add_species_lines = "\n".join(
        f"sim.add_species({sp.name}, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[{sp.ppc}]))"
        for sp in spec.species if sp.injection_style not in ("none", "gaussian_beam")
    ) + "\n" + "\n".join(
        f"sim.add_species({sp.name}, layout=picmi.PseudoRandomLayout(n_macroparticles={sp.n_macro}))"
        for sp in spec.species if sp.injection_style == "gaussian_beam"
    )

    d = {"name": spec.name, "domain": asdict(spec.domain), "solver": asdict(spec.solver)}
    maxwell_method = spec.solver.maxwell_solver.upper() if spec.solver.maxwell_solver != "yee" else "Yee"

    script = f"""#!/usr/bin/env python3
# Auto-generated by warpx-inputgen (electromagnetic_pic PICMI).
from pywarpx import picmi

SPEC = {d!r}

grid = picmi.{grid_class}(
    number_of_cells={spec.domain.number_of_cells!r},
    lower_bound={spec.domain.lower_bound!r},
    upper_bound={spec.domain.upper_bound!r},
    lower_boundary_conditions={spec.domain.field_bc!r},
    upper_boundary_conditions={spec.domain.field_bc!r},
)

solver = picmi.ElectromagneticSolver(grid=grid, method={maxwell_method!r}, cfl={spec.solver.cfl!r})

# --- Species ---
{species_code}

sim = picmi.Simulation(
    solver=solver,
    max_steps={spec.solver.max_steps!r},
    particle_shape={spec.solver.particle_shape!r},
)

{add_species_lines}

{diag_code}

sim.initialize_inputs()
sim.initialize_warpx()
sim.step({spec.solver.max_steps!r})
"""
    return script.lstrip()
