from __future__ import annotations

"""Ion beam instability (R-instability) hybrid-PIC input generation.

Produces native AMReX ParmParse inputs for a two-species hybrid-PIC run
consisting of a background ion population and a drifting ion beam, both
moving through Ohm's-law fluid electrons.

Physical setup:
  - Background ions at rest (or with a small counter-drift for momentum
    conservation)
  - Beam ions drifting along z with velocity beam_drift_velocity [m/s]
  - Uniform background B0 field (typically along z)
  - Hybrid-PIC solver: kinetic ions + Ohm's law electrons

Reference: Weidl et al. (2016), PRL; Gary et al. (1984), JGR.

Physical constants (CODATA 2018):
    m_p = 1.67262192369e-27 kg
    c   = 299792458.0 m/s
    q_e = 1.602176634e-19 C
"""

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .blocks import (
    AMRSpec,
    DiagSpec,
    DomainSpec,
    HybridIonSpec,
    OhmSolverSpec,
    SolverSpec,
    _AMR_KEY_MAP,
    _emit_amr_block,
    _emit_checkpoint_block,
    _emit_diag_block,
    _emit_picmi_diag_lines,
    suggest_cells,
)

# CODATA 2018 values
_M_P = 1.67262192369e-27
_C   = 299792458.0
_Q_E = 1.602176634e-19

_DOMAIN_KEYS = {"dim", "number_of_cells", "lower_bound", "upper_bound", "field_bc"}
_SOLVER_KEYS = {"max_steps", "cfl", "particle_shape"}
_OHM_KEYS    = {"Te", "n0_ref", "gamma", "resistivity", "hyper_resistivity",
                 "substeps", "n_floor"}


@dataclass
class IonBeamInstabilitySpec:
    """Spec for a two-species hybrid-PIC ion beam instability run.

    Default parameters reproduce the 1D R-instability benchmark:
      n0 = 3.3e22 m⁻³, B0 = 0.25 T (along z), m_ion = 1 amu (proton)
      beam fraction n_beam/n_core = 0.1
      beam drift U = 10 vA where vA ≈ 2.4e5 m/s

    Total z-momentum is conserved: core_drift = -(n_beam/n_core) * beam_drift
    when core_drift_velocity = 0 (auto-compute flag).
    """
    name: str = "ion_beam_instability"
    domain: DomainSpec = field(
        default_factory=lambda: DomainSpec(
            dim=1,
            number_of_cells=[1024],
            lower_bound=[0.0],
            upper_bound=[0.256],          # ≈ 204 ion skin depths for n0=3.3e22, proton
            field_bc=["periodic"],
        )
    )
    solver: SolverSpec = field(
        default_factory=lambda: SolverSpec(max_steps=2000, particle_shape="linear")
    )
    ohm: OhmSolverSpec = field(default_factory=OhmSolverSpec)
    # Background (core) ions
    core: HybridIonSpec = field(
        default_factory=lambda: HybridIonSpec(
            density=2.97e22,             # n_core = 0.9 * n0
            mass_amu=1.0,
            temperature_eV=0.05,
            ppc=256,
        )
    )
    # Beam ions
    beam: HybridIonSpec = field(
        default_factory=lambda: HybridIonSpec(
            density=3.3e21,              # n_beam = 0.1 * n0
            mass_amu=1.0,
            temperature_eV=0.05,
            ppc=64,
        )
    )
    # Beam drift velocity along z [m/s]; positive = +z direction
    beam_drift_velocity: float = 2.4e6    # ≈ 10 vA for default parameters
    # Core counter-drift [m/s]; if 0.0, auto-compute for momentum conservation
    core_drift_velocity: float = 0.0
    # Fixed timestep [s]; default ≈ 0.01 T_ci for B0=0.25T, proton
    const_dt: float = 1.3e-10
    # Uniform background B field [Bx, By, Bz] in Tesla
    B0: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.25])
    diag: DiagSpec = field(
        default_factory=lambda: DiagSpec(
            diag_period=100,
            diag_fields=["Bx", "By", "Bz", "Ex", "Ey", "Ez",
                         "rho", "jx_displacement", "jy_displacement", "jz_displacement"],
        )
    )
    # AMR / resolution
    amr: AMRSpec = field(default_factory=AMRSpec)
    # Checkpoint: write AMReX checkpoint every N steps; None = no checkpoint
    checkpoint_int: Optional[int] = None
    checkpoint_file: str = "chk"

    @classmethod
    def from_dict(cls, d: dict) -> "IonBeamInstabilitySpec":
        amr = AMRSpec(**{attr: d[flat] for flat, attr in _AMR_KEY_MAP.items() if flat in d})
        if "dx_target" in d and "number_of_cells" not in d:
            if "lower_bound" in d and "upper_bound" in d:
                d = dict(d)
                d["number_of_cells"] = suggest_cells(
                    d["lower_bound"], d["upper_bound"], d["dx_target"], amr.blocking_factor
                )
        domain = DomainSpec(**{k: d[k] for k in _DOMAIN_KEYS if k in d})
        solver = SolverSpec(**{k: d[k] for k in _SOLVER_KEYS if k in d})
        ohm = OhmSolverSpec(**{k: d[k] for k in _OHM_KEYS if k in d})

        core_kw: dict = {}
        if "core_density" in d:     core_kw["density"]        = d["core_density"]
        if "core_mass_amu" in d:    core_kw["mass_amu"]       = d["core_mass_amu"]
        if "core_temperature_eV" in d: core_kw["temperature_eV"] = d["core_temperature_eV"]
        if "core_ppc" in d:         core_kw["ppc"]            = d["core_ppc"]
        core = HybridIonSpec(**core_kw)

        beam_kw: dict = {}
        if "beam_density" in d:     beam_kw["density"]        = d["beam_density"]
        if "beam_mass_amu" in d:    beam_kw["mass_amu"]       = d["beam_mass_amu"]
        if "beam_temperature_eV" in d: beam_kw["temperature_eV"] = d["beam_temperature_eV"]
        if "beam_ppc" in d:         beam_kw["ppc"]            = d["beam_ppc"]
        beam = HybridIonSpec(**beam_kw)

        diag = DiagSpec.from_dict(d)
        return cls(
            name=d.get("name", "ion_beam_instability"),
            domain=domain, solver=solver, ohm=ohm, core=core, beam=beam,
            beam_drift_velocity=d.get("beam_drift_velocity", 2.4e6),
            core_drift_velocity=d.get("core_drift_velocity", 0.0),
            const_dt=d.get("const_dt", 1.3e-10),
            B0=d.get("B0", [0.0, 0.0, 0.25]),
            diag=diag,
            amr=amr,
            checkpoint_int=d.get("checkpoint_int"),
            checkpoint_file=d.get("checkpoint_file", "chk"),
        )


def generate_inputs_ion_beam_instability(spec: IonBeamInstabilitySpec) -> str:
    """Generate a native WarpX AMReX ParmParse inputs file for a two-species
    hybrid-PIC ion beam instability run.

    Two ion species are used: ``core_ions`` (background) and ``beam_ions``
    (drifting beam).  The core counter-drift is computed automatically for
    total z-momentum conservation when ``spec.core_drift_velocity == 0``.
    """
    if spec.domain.dim not in (1, 2, 3):
        raise ValueError(
            f"IonBeamInstabilitySpec supports dim=1, 2, or 3; got {spec.domain.dim}"
        )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    _PSHAPE_MAP = {"linear": 1, "quadratic": 2, "cubic": 3}
    pshape = spec.solver.particle_shape
    if isinstance(pshape, str):
        pshape = _PSHAPE_MAP[pshape.lower()]

    core_mass_kg = spec.core.mass_amu * _M_P
    beam_mass_kg = spec.beam.mass_amu * _M_P

    # Thermal velocities → normalised momenta
    core_v_th = math.sqrt(spec.core.temperature_eV * _Q_E / core_mass_kg)
    core_u_th = core_v_th / _C
    beam_v_th = math.sqrt(spec.beam.temperature_eV * _Q_E / beam_mass_kg)
    beam_u_th = beam_v_th / _C

    # Drift velocities in normalised units
    beam_uz_m = spec.beam_drift_velocity / _C

    if spec.core_drift_velocity != 0.0:
        core_uz_m = spec.core_drift_velocity / _C
    else:
        # Auto-compute for total momentum conservation
        core_uz_m = -(spec.beam.density / spec.core.density) * beam_uz_m

    # Domain
    n_cell  = " ".join(str(n) for n in spec.domain.number_of_cells)
    prob_lo = " ".join(f"{x:.17g}" for x in spec.domain.lower_bound)
    prob_hi = " ".join(f"{x:.17g}" for x in spec.domain.upper_bound)
    bc      = " ".join(spec.domain.field_bc)

    # B0
    bx, by, bz = spec.B0[0], spec.B0[1], spec.B0[2]

    # Hyper-resistivity
    hyper_line = (
        f"hybrid_pic_model.plasma_hyper_resistivity(rho,B) = "
        f"{spec.ohm.hyper_resistivity:.17g}"
        if spec.ohm.hyper_resistivity != 0.0
        else "# hybrid_pic_model.plasma_hyper_resistivity = 0 (omitted)"
    )

    # Diagnostics
    diag_section = _emit_diag_block(spec.diag)

    _chk = _emit_checkpoint_block(spec.checkpoint_int, spec.checkpoint_file)
    checkpoint_section = ("\n" + _chk + "\n") if _chk else ""

    d = {
        "name": spec.name,
        "domain": asdict(spec.domain),
        "solver": asdict(spec.solver),
        "ohm": asdict(spec.ohm),
        "core": asdict(spec.core),
        "beam": asdict(spec.beam),
        "beam_drift_velocity": spec.beam_drift_velocity,
        "core_drift_velocity": spec.core_drift_velocity,
        "const_dt": spec.const_dt,
        "B0": spec.B0,
        "diag": asdict(spec.diag),
    }

    text = f"""\
# Auto-generated by pywarpx.inputgen (ion_beam_instability)
# SPEC = {d!r}

# --- Time -------------------------------------------------------------------
max_step = {spec.solver.max_steps}

# --- AMR / domain -----------------------------------------------------------
{_emit_amr_block(spec.amr, n_cell, prob_lo, prob_hi, spec.domain.dim)}

# --- Boundary conditions ----------------------------------------------------
boundary.field_lo = {bc}
boundary.field_hi = {bc}
boundary.particle_lo = {bc}
boundary.particle_hi = {bc}

# --- Numerics ---------------------------------------------------------------
warpx.verbose = 1
warpx.const_dt = {spec.const_dt:.17g}

algo.maxwell_solver = hybrid
algo.particle_shape = {pshape}
algo.current_deposition = direct

# --- Hybrid-PIC model (Ohm's law) -------------------------------------------
hybrid_pic_model.substeps = {spec.ohm.substeps}
hybrid_pic_model.gamma = {spec.ohm.gamma:.17g}
hybrid_pic_model.elec_temp = {spec.ohm.Te:.17g}
hybrid_pic_model.n0_ref = {spec.ohm.n0_ref:.17g}
hybrid_pic_model.plasma_resistivity(rho,J) = {spec.ohm.resistivity:.17g}
{hyper_line}
hybrid_pic_model.n_floor = {spec.ohm.n_floor:.17g}

# --- Applied background magnetic field [T] ----------------------------------
warpx.B_ext_grid_init_style = constant
warpx.B_external_grid = {bx:.17g} {by:.17g} {bz:.17g}

# --- Ion species (kinetic) --------------------------------------------------
particles.species_names = core_ions beam_ions

# Background (core) ions
core_ions.charge = q_e
core_ions.mass = {core_mass_kg:.17g}
core_ions.injection_style = NRandomPerCell
core_ions.num_particles_per_cell = {spec.core.ppc}
core_ions.profile = constant
core_ions.density = {spec.core.density:.17g}
core_ions.momentum_distribution_type = gaussian
core_ions.ux_m = 0.0
core_ions.uy_m = 0.0
core_ions.uz_m = {core_uz_m:.17g}
core_ions.ux_th = {core_u_th:.17g}
core_ions.uy_th = {core_u_th:.17g}
core_ions.uz_th = {core_u_th:.17g}

# Drifting beam ions
beam_ions.charge = q_e
beam_ions.mass = {beam_mass_kg:.17g}
beam_ions.injection_style = NRandomPerCell
beam_ions.num_particles_per_cell = {spec.beam.ppc}
beam_ions.profile = constant
beam_ions.density = {spec.beam.density:.17g}
beam_ions.momentum_distribution_type = gaussian
beam_ions.ux_m = 0.0
beam_ions.uy_m = 0.0
beam_ions.uz_m = {beam_uz_m:.17g}
beam_ions.ux_th = {beam_u_th:.17g}
beam_ions.uy_th = {beam_u_th:.17g}
beam_ions.uz_th = {beam_u_th:.17g}

# --- Diagnostics ------------------------------------------------------------
{diag_section}
{checkpoint_section}"""
    return text


_GRID_CLASS_IBI = {1: "Cartesian1DGrid", 2: "Cartesian2DGrid", 3: "Cartesian3DGrid"}


def generate_picmi_ion_beam_instability(spec: IonBeamInstabilitySpec) -> str:
    """Generate a PICMI script for a two-species hybrid-PIC ion beam instability run."""
    from dataclasses import asdict

    grid_class = _GRID_CLASS_IBI[spec.domain.dim]
    core_mass_kg = spec.core.mass_amu * _M_P
    beam_mass_kg = spec.beam.mass_amu * _M_P

    core_v_th = math.sqrt(spec.core.temperature_eV * _Q_E / core_mass_kg)
    core_u_th = core_v_th / _C
    beam_v_th = math.sqrt(spec.beam.temperature_eV * _Q_E / beam_mass_kg)
    beam_u_th = beam_v_th / _C

    beam_uz_m = spec.beam_drift_velocity / _C
    if spec.core_drift_velocity != 0.0:
        core_uz_m = spec.core_drift_velocity / _C
    else:
        core_uz_m = -(spec.beam.density / spec.core.density) * beam_uz_m

    bx, by, bz = spec.B0[0], spec.B0[1], spec.B0[2]

    diag_code = _emit_picmi_diag_lines(spec.diag, ["core_ions", "beam_ions"])

    d = {
        "name": spec.name,
        "domain": asdict(spec.domain),
        "ohm": asdict(spec.ohm),
        "core": asdict(spec.core),
        "beam": asdict(spec.beam),
        "beam_drift_velocity": spec.beam_drift_velocity,
        "const_dt": spec.const_dt,
        "B0": spec.B0,
    }

    script = f"""#!/usr/bin/env python3
# Auto-generated by warpx-inputgen (ion_beam_instability PICMI).
from pywarpx import picmi

SPEC = {d!r}

grid = picmi.{grid_class}(
    number_of_cells={spec.domain.number_of_cells!r},
    lower_bound={spec.domain.lower_bound!r},
    upper_bound={spec.domain.upper_bound!r},
    lower_boundary_conditions={spec.domain.field_bc!r},
    upper_boundary_conditions={spec.domain.field_bc!r},
)

# Hybrid-PIC (Ohm's law) solver
solver = picmi.HybridPICSolver(
    grid=grid,
    Te={spec.ohm.Te!r},
    density={spec.ohm.n0_ref!r},
    gamma={spec.ohm.gamma!r},
    plasma_resistivity={spec.ohm.resistivity!r},
    substeps={spec.ohm.substeps!r},
)

# Core ions (background, slight counter-drift for momentum conservation)
core_dist = picmi.UniformDistribution(
    density={spec.core.density!r}, fill_in=True,
    rms_velocity=[{core_u_th!r}, {core_u_th!r}, {core_u_th!r}],
    directed_velocity=[0.0, 0.0, {core_uz_m!r}],
)
core_ions = picmi.Species(
    name='core_ions', charge_state=1, mass={core_mass_kg!r},
    initial_distribution=core_dist,
)

# Beam ions (drifting at beam_drift_velocity)
beam_dist = picmi.UniformDistribution(
    density={spec.beam.density!r}, fill_in=True,
    rms_velocity=[{beam_u_th!r}, {beam_u_th!r}, {beam_u_th!r}],
    directed_velocity=[0.0, 0.0, {beam_uz_m!r}],
)
beam_ions = picmi.Species(
    name='beam_ions', charge_state=1, mass={beam_mass_kg!r},
    initial_distribution=beam_dist,
)

sim = picmi.Simulation(
    solver=solver,
    max_steps={spec.solver.max_steps!r},
    particle_shape={spec.solver.particle_shape!r},
    warpx_const_dt={spec.const_dt!r},
    # Background uniform magnetic field B0
    warpx_B_ext_grid_init_style='constant',
    warpx_B_external_grid=[{bx!r}, {by!r}, {bz!r}],
)

sim.add_species(core_ions, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[{spec.core.ppc}]))
sim.add_species(beam_ions, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[{spec.beam.ppc}]))

{diag_code}

sim.initialize_inputs()
sim.initialize_warpx()
sim.step({spec.solver.max_steps!r})
"""
    return script.lstrip()
