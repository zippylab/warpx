from __future__ import annotations

"""Electrostatic PIC (ES-PIC) input generation.

Produces native AMReX ParmParse inputs files for purely electrostatic
simulations (Poisson solver, no EM waves).

WarpX ES-PIC key flags:
  - warpx.do_electrostatic = labframe (or relativistic)
  - warpx.self_fields_required_precision: MLMG tolerance
  - warpx.const_dt: fixed timestep (Boris leapfrog)
  - Maxwell solver is implicitly skipped in electrostatic mode

Physical constants (CODATA 2018):
    m_p = 1.67262192369e-27 kg   (proton mass)
    m_e = 9.1093837015e-31  kg   (electron mass)
    c   = 299792458.0 m/s
    q_e = 1.602176634e-19   C
"""

import math
from dataclasses import asdict, dataclass, field
from typing import List

from .blocks import DiagSpec, DomainSpec, EBSpec, SolverSpec

# CODATA 2018 values
_M_P = 1.67262192369e-27  # kg  (proton mass)
_M_E = 9.1093837015e-31   # kg  (electron mass)
_C   = 299792458.0          # m/s
_Q_E = 1.602176634e-19    # C

# Flat JSON key → block mapping
_DOMAIN_KEYS = {"dim", "number_of_cells", "lower_bound", "upper_bound", "field_bc"}
_SOLVER_KEYS = {"max_steps", "particle_shape"}
_DIAG_KEYS   = {"diag_period", "diag_fields"}
_EB_KEYS     = {"eb_implicit_function", "eb_potential", "stl_file"}


@dataclass
class ElectrostaticPlasmaSpec:
    """Spec for an electrostatic (Poisson-based) PIC run.

    Default values correspond to a simple 1D electron-proton plasma:
      n0 = 1e16 m^-3, Te = 1 eV, Ti = 0 eV (cold ions)
      Debye length λ_De ≈ 74.3 μm; dx = 50 μm (Nc=100, L=5mm) → dx/λ_De ≈ 0.67
      Plasma frequency ω_pe ≈ 5.64e9 rad/s; dt*ω_pe ≈ 0.056 (stable and accurate)
    """
    name: str = "electrostatic_plasma"
    domain: DomainSpec = field(
        default_factory=lambda: DomainSpec(
            dim=1,
            number_of_cells=[100],
            lower_bound=[0.0],
            upper_bound=[5e-3],
            field_bc=["periodic"],
        )
    )
    solver: SolverSpec = field(
        default_factory=lambda: SolverSpec(max_steps=200, particle_shape="linear")
    )
    # Fixed timestep in seconds. Default ≈ 0.056 / ω_pe for n0=1e16.
    const_dt: float = 1e-11
    # Plasma parameters
    n0: float = 1e16               # Reference density [m^-3] (electrons + ions)
    Te: float = 1.0                # Electron temperature [eV]
    Ti: float = 0.0                # Ion temperature [eV] (0 = cold ions)
    ion_mass_amu: float = 1.0      # Ion mass in proton masses (1.0 = proton)
    ppc: int = 16                  # Particles per cell (each species)
    include_ions: bool = True      # Include a neutralizing singly-charged ion species
    electrostatic_solver: str = "labframe"   # "labframe" or "relativistic"
    poisson_precision: float = 1e-11         # MLMG relative precision
    diag: DiagSpec = field(
        default_factory=lambda: DiagSpec(
            diag_period=50,
            diag_fields=["Ex", "Ey", "Ez", "rho"],
        )
    )
    eb: EBSpec = field(default_factory=EBSpec)   # optional embedded boundary

    @classmethod
    def from_dict(cls, d: dict) -> "ElectrostaticPlasmaSpec":
        """Construct from a flat JSON dict (backward-compatible external format)."""
        domain = DomainSpec(**{k: d[k] for k in _DOMAIN_KEYS if k in d})
        solver = SolverSpec(**{k: d[k] for k in _SOLVER_KEYS if k in d})
        diag = DiagSpec(**{k: d[k] for k in _DIAG_KEYS if k in d})
        eb = EBSpec(**{k: d[k] for k in _EB_KEYS if k in d})
        return cls(
            name=d.get("name", "electrostatic_plasma"),
            domain=domain,
            solver=solver,
            const_dt=d.get("const_dt", 1e-11),
            n0=d.get("n0", 1e16),
            Te=d.get("Te", 1.0),
            Ti=d.get("Ti", 0.0),
            ion_mass_amu=d.get("ion_mass_amu", 1.0),
            ppc=d.get("ppc", 16),
            include_ions=d.get("include_ions", True),
            electrostatic_solver=d.get("electrostatic_solver", "labframe"),
            poisson_precision=d.get("poisson_precision", 1e-11),
            diag=diag,
            eb=eb,
        )


def generate_inputs_electrostatic_plasma(spec: ElectrostaticPlasmaSpec) -> str:
    """Generate a native WarpX AMReX ParmParse inputs file for an ES-PIC run.

    Supports dim=1, 2, or 3. Uses the Poisson (MLMG) solver for the electric
    field. No EM solver; Maxwell equations are skipped entirely.

    The ion section is only emitted when ``include_ions=True`` (default). With
    cold ions (Ti=0) a constant momentum distribution is used; otherwise a
    Gaussian thermal distribution is used.
    """

    if spec.domain.dim not in (1, 2, 3):
        raise ValueError(
            f"ElectrostaticPlasmaSpec supports dim=1, 2, or 3; got dim={spec.domain.dim}"
        )

    d = {
        "name": spec.name,
        "domain": asdict(spec.domain),
        "solver": asdict(spec.solver),
        "const_dt": spec.const_dt,
        "n0": spec.n0,
        "Te": spec.Te,
        "Ti": spec.Ti,
        "ion_mass_amu": spec.ion_mass_amu,
        "ppc": spec.ppc,
        "include_ions": spec.include_ions,
        "electrostatic_solver": spec.electrostatic_solver,
        "poisson_precision": spec.poisson_precision,
        "diag": asdict(spec.diag),
    }

    # ------------------------------------------------------------------
    # Particle shape integer
    # ------------------------------------------------------------------
    _PSHAPE_MAP = {"linear": 1, "quadratic": 2, "cubic": 3}
    pshape = spec.solver.particle_shape
    if isinstance(pshape, str):
        pshape = _PSHAPE_MAP[pshape.lower()]

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    n_cell   = " ".join(str(n) for n in spec.domain.number_of_cells)
    prob_lo  = " ".join(f"{x:.17g}" for x in spec.domain.lower_bound)
    prob_hi  = " ".join(f"{x:.17g}" for x in spec.domain.upper_bound)
    field_lo = " ".join(bc for bc in spec.domain.field_bc)
    field_hi = field_lo  # symmetric
    # AMReX requires n_cell divisible by blocking_factor (default 8).
    # For simple non-AMR ES-PIC runs, set blocking_factor=1 for full flexibility.
    blocking_factor = 1

    # ------------------------------------------------------------------
    # Electron thermal velocity
    # ------------------------------------------------------------------
    # v_te = sqrt(Te_J / m_e); normalised momentum u_te = v_te / c
    Te_J    = spec.Te * _Q_E
    e_v_th  = math.sqrt(Te_J / _M_E)
    e_u_th  = e_v_th / _C

    # ------------------------------------------------------------------
    # Ion species section (optional)
    # ------------------------------------------------------------------
    species_names = "electrons"
    ion_section = ""

    if spec.include_ions:
        species_names = "electrons ions"
        ion_mass_kg = spec.ion_mass_amu * _M_P

        if spec.Ti > 0.0:
            Ti_J      = spec.Ti * _Q_E
            ion_v_th  = math.sqrt(Ti_J / ion_mass_kg)
            ion_u_th  = ion_v_th / _C
            ion_mom_type = "gaussian"
        else:
            ion_u_th  = 0.0
            ion_mom_type = "constant"

        ion_section = f"""
ions.charge = q_e
ions.mass = {ion_mass_kg:.17g}
ions.injection_style = NRandomPerCell
ions.num_particles_per_cell = {spec.ppc}
ions.profile = constant
ions.density = {spec.n0:.17g}
ions.momentum_distribution_type = {ion_mom_type}
ions.ux_m = 0.0
ions.uy_m = 0.0
ions.uz_m = 0.0
ions.ux_th = {ion_u_th:.17g}
ions.uy_th = {ion_u_th:.17g}
ions.uz_th = {ion_u_th:.17g}"""

    # ------------------------------------------------------------------
    # Embedded boundary (optional)
    # ------------------------------------------------------------------
    eb_section = ""
    if spec.eb.eb_implicit_function.strip():
        eb_section = f"\n# --- Embedded boundary (implicit function) ---------------------------\nwarpx.eb_implicit_function = {spec.eb.eb_implicit_function}\n"
        if spec.eb.eb_potential.strip():
            eb_section += f"warpx.eb_potential(x,y,z,t) = {spec.eb.eb_potential}\n"
    elif spec.eb.stl_file.strip():
        eb_section = f"\n# --- Embedded boundary (STL) -----------------------------------------\neb2.geom_type = stl\neb2.stl_file = {spec.eb.stl_file}\n"
        if spec.eb.eb_potential.strip():
            eb_section += f"warpx.eb_potential(x,y,z,t) = {spec.eb.eb_potential}\n"

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
    text = f"""\
# Auto-generated by pywarpx.inputgen (electrostatic_plasma)
# SPEC = {d!r}

# --- Time -------------------------------------------------------------------
max_step = {spec.solver.max_steps}

# --- AMR / domain -----------------------------------------------------------
amr.max_level = 0
amr.n_cell = {n_cell}
amr.blocking_factor = {blocking_factor}

geometry.dims = {spec.domain.dim}
geometry.prob_lo = {prob_lo}
geometry.prob_hi = {prob_hi}

# --- Boundary conditions ----------------------------------------------------
boundary.field_lo = {field_lo}
boundary.field_hi = {field_hi}
boundary.particle_lo = {field_lo}
boundary.particle_hi = {field_hi}
{eb_section}
# --- Electrostatic solver (Poisson / MLMG) ----------------------------------
warpx.do_electrostatic = {spec.electrostatic_solver}
warpx.self_fields_required_precision = {spec.poisson_precision:.17g}
warpx.self_fields_max_iters = 200
warpx.const_dt = {spec.const_dt:.17g}

# --- Numerics ---------------------------------------------------------------
warpx.verbose = 1
algo.particle_shape = {pshape}

# --- Species ----------------------------------------------------------------
particles.species_names = {species_names}

electrons.charge = -q_e
electrons.mass = m_e
electrons.injection_style = NRandomPerCell
electrons.num_particles_per_cell = {spec.ppc}
electrons.profile = constant
electrons.density = {spec.n0:.17g}
electrons.momentum_distribution_type = gaussian
electrons.ux_m = 0.0
electrons.uy_m = 0.0
electrons.uz_m = 0.0
electrons.ux_th = {e_u_th:.17g}
electrons.uy_th = {e_u_th:.17g}
electrons.uz_th = {e_u_th:.17g}{ion_section}

# --- Diagnostics ------------------------------------------------------------
{diag_section}
"""

    return text
