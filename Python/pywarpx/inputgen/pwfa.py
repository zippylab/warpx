from __future__ import annotations

"""Plasma wakefield acceleration (PWFA) input generation.

Produces native AMReX ParmParse inputs for a beam-driven PWFA run:
  - Background electron-ion plasma (EM FDTD solver)
  - Driver electron beam (Gaussian bunch)
  - Optional witness electron beam
  - Moving window following the driver at velocity c

Physical setup:
  - Driver beam propagates along z at relativistic momentum uz_m >> 1
  - Beam launches a plasma wake that accelerates the witness (if present)
  - Background plasma is neutral (electrons + ions)

Reference:
  Blumenfeld et al. (2007), Nature 445, 741.
  WarpX PWFA example: Examples/Physics_applications/plasma_acceleration/

Physical constants (CODATA 2018):
    m_e = 9.1093837015e-31 kg
    q_e = 1.602176634e-19 C
    c   = 299792458.0 m/s
    m_p = 1.67262192369e-27 kg
"""

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .blocks import DiagSpec, DomainSpec, ParticleBeamSpec, SolverSpec, SpeciesSpec

_M_E = 9.1093837015e-31
_M_P = 1.67262192369e-27
_C   = 299792458.0
_Q_E = 1.602176634e-19

_DOMAIN_KEYS  = {"dim", "number_of_cells", "lower_bound", "upper_bound", "field_bc"}
_SOLVER_KEYS  = {"max_steps", "cfl", "particle_shape"}
_SPECIES_KEYS = {"plasma_density", "plasma_zmin", "plasma_zmax"}
_DIAG_KEYS    = {"diag_period", "diag_fields"}


def _beam_from_prefix(d: dict, prefix: str) -> ParticleBeamSpec:
    """Extract ParticleBeamSpec fields from a flat dict with a given prefix."""
    kw: dict = {}
    mapping = {
        f"{prefix}_x_rms": "x_rms",
        f"{prefix}_y_rms": "y_rms",
        f"{prefix}_z_rms": "z_rms",
        f"{prefix}_z_cut": "z_cut",
        f"{prefix}_uz_m":  "uz_m",
        f"{prefix}_uz_th": "uz_th",
        f"{prefix}_q_tot": "q_tot",
        f"{prefix}_z_mean": "z_mean",
        f"{prefix}_n_macro": "n_macro",
    }
    for flat_key, attr in mapping.items():
        if flat_key in d:
            kw[attr] = d[flat_key]
    return ParticleBeamSpec(**kw)


@dataclass
class PWFASpec:
    """Spec for a beam-driven plasma wakefield acceleration (PWFA) run.

    Default values correspond to a simple 2D lab-frame simulation with
    a sub-GeV driver and background density n = 1e22 m⁻³.

    The driver propagates along +z.  A moving window follows the driver
    at velocity c when ``moving_window=True`` (default).
    """
    name: str = "pwfa"
    domain: DomainSpec = field(
        default_factory=lambda: DomainSpec(
            dim=2,
            number_of_cells=[128, 512],
            lower_bound=[-150e-6, -200e-6],
            upper_bound=[150e-6,    0.0],
            field_bc=["open", "open"],
        )
    )
    solver: SolverSpec = field(
        default_factory=lambda: SolverSpec(max_steps=1000, cfl=0.99,
                                           particle_shape="linear")
    )
    # Background plasma: electrons + ions (quasi-neutral)
    plasma: SpeciesSpec = field(
        default_factory=lambda: SpeciesSpec(
            plasma_density=1e22,
            plasma_zmin=-200e-6,
            plasma_zmax=0.0,
        )
    )
    # Driver beam
    driver: ParticleBeamSpec = field(
        default_factory=lambda: ParticleBeamSpec(
            x_rms=2e-6, y_rms=2e-6, z_rms=4e-6,
            uz_m=2000.0, uz_th=20.0,
            q_tot=-1e-9, z_mean=-50e-6, n_macro=1000,
        )
    )
    # Optional witness beam (None = no witness)
    witness: Optional[ParticleBeamSpec] = None
    # Moving window: follows driver at velocity c along z
    moving_window: bool = True
    diag: DiagSpec = field(
        default_factory=lambda: DiagSpec(
            diag_period=50,
            diag_fields=["Ex", "Ey", "Ez", "Bx", "By", "Bz", "rho"],
        )
    )

    @classmethod
    def from_dict(cls, d: dict) -> "PWFASpec":
        domain = DomainSpec(**{k: d[k] for k in _DOMAIN_KEYS if k in d})
        solver = SolverSpec(**{k: d[k] for k in _SOLVER_KEYS if k in d})
        plasma = SpeciesSpec(**{k: d[k] for k in _SPECIES_KEYS if k in d})
        driver = _beam_from_prefix(d, "driver")
        witness = _beam_from_prefix(d, "witness") if any(
            k.startswith("witness_") for k in d
        ) else None
        diag = DiagSpec(**{k: d[k] for k in _DIAG_KEYS if k in d})
        return cls(
            name=d.get("name", "pwfa"),
            domain=domain, solver=solver, plasma=plasma,
            driver=driver, witness=witness,
            moving_window=d.get("moving_window", True),
            diag=diag,
        )


def generate_inputs_pwfa(spec: PWFASpec) -> str:
    """Generate native WarpX AMReX ParmParse inputs for a PWFA run.

    Uses the standard FDTD Maxwell solver with CFL-limited timestep.
    A moving window follows the driver bunch at velocity c.

    Both background species (electrons + protons) are injected uniformly
    throughout the plasma region.  The driver (and optional witness) are
    Gaussian bunches injected via ``gaussian_beam`` injection style.
    """
    if spec.domain.dim not in (2, 3):
        raise ValueError(
            f"PWFASpec supports dim=2 or 3; got {spec.domain.dim}"
        )

    _PSHAPE_MAP = {"linear": 1, "quadratic": 2, "cubic": 3}
    pshape = spec.solver.particle_shape
    if isinstance(pshape, str):
        pshape = _PSHAPE_MAP[pshape.lower()]

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    n_cell  = " ".join(str(n) for n in spec.domain.number_of_cells)
    prob_lo = " ".join(f"{x:.17g}" for x in spec.domain.lower_bound)
    prob_hi = " ".join(f"{x:.17g}" for x in spec.domain.upper_bound)
    bc      = " ".join(spec.domain.field_bc)

    # ------------------------------------------------------------------
    # Background plasma thermal velocity (room-temperature; cold plasma default)
    # ------------------------------------------------------------------
    # Electrons: Te ~ 1 eV → v_th/c ~ 0.002
    Te_J = 1.0 * _Q_E    # 1 eV background electron temperature
    e_u_th = math.sqrt(Te_J / _M_E) / _C

    # ------------------------------------------------------------------
    # Species list
    # ------------------------------------------------------------------
    species_names = "electrons protons driver_beam"
    if spec.witness is not None:
        species_names += " witness_beam"

    # ------------------------------------------------------------------
    # Moving window
    # ------------------------------------------------------------------
    moving_window_block = ""
    if spec.moving_window:
        moving_window_block = """\

# --- Moving window ----------------------------------------------------------
warpx.do_moving_window = 1
warpx.moving_window_direction = z
warpx.moving_window_v = 1.0  # in units of c
"""

    # ------------------------------------------------------------------
    # Driver beam section
    # ------------------------------------------------------------------
    def _beam_section(name: str, b: ParticleBeamSpec) -> str:
        return f"""\
{name}.charge = -q_e
{name}.mass = m_e
{name}.injection_style = gaussian_beam
{name}.x_rms = {b.x_rms:.17g}
{name}.y_rms = {b.y_rms:.17g}
{name}.z_rms = {b.z_rms:.17g}
{name}.z_cut = {b.z_cut:.17g}
{name}.npart = {b.n_macro}
{name}.q_tot = {b.q_tot:.17g}
{name}.z_m = {b.z_mean:.17g}
{name}.momentum_distribution_type = gaussian
{name}.ux_m = 0.0
{name}.uy_m = 0.0
{name}.uz_m = {b.uz_m:.17g}
{name}.ux_th = 0.0
{name}.uy_th = 0.0
{name}.uz_th = {b.uz_th:.17g}"""

    driver_section = _beam_section("driver_beam", spec.driver)
    witness_section = (
        "\n" + _beam_section("witness_beam", spec.witness)
        if spec.witness is not None
        else ""
    )

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

    d = {
        "name": spec.name,
        "domain": asdict(spec.domain),
        "solver": asdict(spec.solver),
        "plasma": asdict(spec.plasma),
        "driver": asdict(spec.driver),
        "witness": asdict(spec.witness) if spec.witness else None,
        "moving_window": spec.moving_window,
        "diag": asdict(spec.diag),
    }

    text = f"""\
# Auto-generated by pywarpx.inputgen (pwfa)
# SPEC = {d!r}

# --- Time -------------------------------------------------------------------
max_step = {spec.solver.max_steps}

# --- AMR / domain -----------------------------------------------------------
amr.max_level = 0
amr.n_cell = {n_cell}
amr.blocking_factor = 1

geometry.dims = {spec.domain.dim}
geometry.prob_lo = {prob_lo}
geometry.prob_hi = {prob_hi}

# --- Boundary conditions ----------------------------------------------------
boundary.field_lo = {bc}
boundary.field_hi = {bc}
boundary.particle_lo = {bc}
boundary.particle_hi = {bc}
{moving_window_block}
# --- Maxwell solver ---------------------------------------------------------
warpx.verbose = 1
algo.maxwell_solver = yee
algo.particle_shape = {pshape}
warpx.cfl = {spec.solver.cfl:.17g}

# --- Species ----------------------------------------------------------------
particles.species_names = {species_names}

# Background electrons
electrons.charge = -q_e
electrons.mass = m_e
electrons.injection_style = NRandomPerCell
electrons.num_particles_per_cell = 2
electrons.profile = constant
electrons.density = {spec.plasma.plasma_density:.17g}
electrons.zmin = {spec.plasma.plasma_zmin:.17g}
electrons.zmax = {spec.plasma.plasma_zmax:.17g}
electrons.momentum_distribution_type = gaussian
electrons.ux_m = 0.0
electrons.uy_m = 0.0
electrons.uz_m = 0.0
electrons.ux_th = {e_u_th:.17g}
electrons.uy_th = {e_u_th:.17g}
electrons.uz_th = {e_u_th:.17g}

# Background protons (neutralising ions)
protons.charge = q_e
protons.mass = m_p
protons.injection_style = NRandomPerCell
protons.num_particles_per_cell = 2
protons.profile = constant
protons.density = {spec.plasma.plasma_density:.17g}
protons.zmin = {spec.plasma.plasma_zmin:.17g}
protons.zmax = {spec.plasma.plasma_zmax:.17g}
protons.momentum_distribution_type = constant
protons.ux_m = 0.0
protons.uy_m = 0.0
protons.uz_m = 0.0

# Driver beam
{driver_section}{witness_section}

# --- Diagnostics ------------------------------------------------------------
{diag_section}
"""
    return text
