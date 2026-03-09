from __future__ import annotations

"""Broad-scope electrostatic PIC (ES-PIC) input generation.

Produces native AMReX ParmParse inputs for a general multi-species ES-PIC run.
Supports arbitrary species lists, optional physics blocks (field ionization,
Coulomb collisions), flexible Poisson solver and particle pusher selection,
and asymmetric lo/hi boundary conditions.

Use this sim type for:
  - Multi-species electrostatic plasma (sheaths, double layers, instabilities)
  - Field ionization with Poisson self-field
  - Ion beam propagation in background plasma
  - Debye-scale electrostatic structures

Unlike electromagnetic_pic, this sim type uses a Poisson solver for the
electric field only; there is no Maxwell wave equation, no laser injection,
and no moving window.  The timestep is always specified explicitly via
``const_dt`` (no CFL constraint from the speed of light, but stability
requires ``dt * omega_pe < 2``).

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
    CollisionSpec,
    DiagSpec,
    DomainSpec,
    EBSpec,
    ESSolverSpec,
    ExtBFieldSpec,
    SpeciesDefSpec,
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
_ES_SOLVER_KEYS = {
    "max_steps", "particle_shape", "poisson_solver",
    "particle_pusher", "poisson_precision", "const_dt", "electrostatic_mode",
}
_DIAG_KEYS = {"diag_period", "diag_fields"}
_EB_KEYS = {"eb_implicit_function", "eb_potential", "stl_file"}
_EXT_BFIELD_KEYS = {"Bx_expression", "By_expression", "Bz_expression"}


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ElectrostaticPICSpec:
    """Specification for a broad-scope ES-PIC simulation.

    All physics blocks are optional.  The minimal required configuration is
    a domain spec and at least one species in the species list.

    Field BC options (per axis): ``"periodic"``, ``"pec"`` (perfect electric
    conductor), ``"pmc"``, ``"open"``.  Note: ``"pml"`` requires EM fields and
    is not meaningful for ES-PIC; use ``"pec"`` or ``"open"`` for wall-like BCs.

    ``field_bc_lo`` / ``field_bc_hi`` override ``domain.field_bc`` for
    independent lo/hi boundary conditions.

    Field ionization example:
      species = [
          {"name":"electrons","charge":-1,"mass_amu":5.486e-4,"density":1e18,"ppc":16,
           "temperature_eV":10.0},
          {"name":"nitrogen","charge":7,"mass_amu":14.003,"density":1e18,"ppc":4,
           "do_field_ionization":True,"physical_element":"N",
           "ionization_initial_level":0,"ionization_product_species":"electrons"},
      ]
    """
    name: str = "electrostatic_pic"
    domain: DomainSpec = field(
        default_factory=lambda: DomainSpec(
            dim=1,
            number_of_cells=[200],
            lower_bound=[0.0],
            upper_bound=[0.01],
            field_bc=["periodic"],
        )
    )
    solver: ESSolverSpec = field(default_factory=ESSolverSpec)
    species: List[SpeciesDefSpec] = field(default_factory=list)
    collisions: List[CollisionSpec] = field(default_factory=list)
    diag: DiagSpec = field(
        default_factory=lambda: DiagSpec(
            diag_period=50,
            diag_fields=["Ex", "rho"],
        )
    )
    # Asymmetric BCs: None = use domain.field_bc for both lo and hi
    field_bc_lo: Optional[List[str]] = None
    field_bc_hi: Optional[List[str]] = None
    # Optional analytic external B-field (for magnetised ES-PIC)
    ext_bfield: Optional[ExtBFieldSpec] = None
    # Optional embedded boundary
    eb: Optional[EBSpec] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ElectrostaticPICSpec":
        """Construct from a flat/nested JSON dict (borealis-mcp spec format).

        Scalar/list keys follow the flat convention used by all sim types.
        The ``species`` and ``collisions`` keys are nested lists of dicts.
        """
        domain = DomainSpec(**{k: d[k] for k in _DOMAIN_KEYS if k in d})
        solver = ESSolverSpec(**{k: d[k] for k in _ES_SOLVER_KEYS if k in d})
        diag = DiagSpec(**{k: d[k] for k in _DIAG_KEYS if k in d})

        species = [SpeciesDefSpec(**e) for e in d.get("species", [])]
        collisions = [CollisionSpec(**e) for e in d.get("collisions", [])]

        ext_bfield = None
        if any(k in d for k in _EXT_BFIELD_KEYS):
            ext_bfield = ExtBFieldSpec(**{k: d[k] for k in _EXT_BFIELD_KEYS if k in d})

        eb = None
        if any(k in d for k in _EB_KEYS):
            eb = EBSpec(**{k: d[k] for k in _EB_KEYS if k in d})

        return cls(
            name=d.get("name", "electrostatic_pic"),
            domain=domain,
            solver=solver,
            species=species,
            collisions=collisions,
            diag=diag,
            field_bc_lo=d.get("field_bc_lo"),
            field_bc_hi=d.get("field_bc_hi"),
            ext_bfield=ext_bfield,
            eb=eb,
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

    # --- QED (unusual in ES-PIC but technically allowed) ---
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

def generate_inputs_electrostatic_pic(spec: ElectrostaticPICSpec) -> str:
    """Generate a native WarpX AMReX ParmParse inputs file for ES-PIC.

    All optional blocks are emitted only when configured.  The output is
    deterministic and can be passed directly to the WarpX binary.
    """
    if spec.domain.dim not in (1, 2, 3):
        raise ValueError(
            f"ElectrostaticPICSpec supports dim=1, 2, or 3; got {spec.domain.dim}"
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
    _PSHAPE_MAP = {"linear": 1, "quadratic": 2, "cubic": 3}
    pshape = spec.solver.particle_shape
    if isinstance(pshape, str):
        pshape = _PSHAPE_MAP.get(pshape.lower(), 1)

    # MLMG precision line only for multigrid solver
    if spec.solver.poisson_solver == "multigrid":
        precision_line = (
            f"warpx.self_fields_required_precision = {spec.solver.poisson_precision:.17g}\n"
            f"warpx.self_fields_max_iters = 200"
        )
    else:
        precision_line = ""

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
    # External B-field (optional — for magnetised ES-PIC)
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
    fields_to_plot = " ".join(spec.diag.diag_fields)
    diag_section = f"""\
diagnostics.diags_names = diag1
diag1.diag_type = Full
diag1.intervals = {spec.diag.diag_period}
diag1.format = openpmd
diag1.fields_to_plot = {fields_to_plot}
diag1.write_species = 0"""

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
    }

    precision_str = f"\n{precision_line}" if precision_line else ""

    return f"""\
# Auto-generated by pywarpx.inputgen (electrostatic_pic)
# SPEC = {d!r}

# --- Time --------------------------------------------------------------------
max_step = {spec.solver.max_steps}

# --- AMR / domain ------------------------------------------------------------
amr.max_level = 0
amr.n_cell = {n_cell}
amr.blocking_factor = 1

geometry.dims = {spec.domain.dim}
geometry.prob_lo = {prob_lo}
geometry.prob_hi = {prob_hi}

# --- Boundary conditions -----------------------------------------------------
boundary.field_lo = {bc_field_lo_str}
boundary.field_hi = {bc_field_hi_str}
boundary.particle_lo = {bc_part_lo_str}
boundary.particle_hi = {bc_part_hi_str}

# --- Electrostatic solver (Poisson) ------------------------------------------
warpx.do_electrostatic = {spec.solver.electrostatic_mode}
warpx.poisson_solver = {spec.solver.poisson_solver}{precision_str}
warpx.const_dt = {spec.solver.const_dt:.17g}

# --- Numerics ----------------------------------------------------------------
warpx.verbose = 1
algo.particle_shape = {pshape}
algo.particle_pusher = {spec.solver.particle_pusher}

# --- Species -----------------------------------------------------------------
particles.species_names = {species_names}

{species_blocks}
{collision_section}\
{ext_bfield_section}\
{eb_section}\
# --- Diagnostics -------------------------------------------------------------
{diag_section}
"""
