from __future__ import annotations

"""Harris-sheet magnetic reconnection hybrid-PIC input generation.

Produces native AMReX ParmParse inputs for a 2D hybrid-PIC Harris-sheet
reconnection run.  The initial magnetic field is set analytically via
WarpX's ``warpx.B_ext_grid_init_style = parse_B_ext_grid_function``.

Physical setup (x-z plane, WarpX 2D convention):
  - Bx = B0 * tanh(z / delta)          (reconnecting field)
  - By = sqrt(Bg**2 + B0**2 - Bx**2)  (guide field, force-free balance)
  - Bz = dB * sin(2π x/Lx) cos(π z/Lz)(perturbation to seed reconnection)

The current sheet is at z = 0 (centre of domain).  Two sheets are formed
naturally by the periodic boundary conditions when the domain spans
(-Lz/2, Lz/2).

Reference: Birn et al. (2001), GEM challenge; WarpX reconnection example
  Examples/Tests/ohm_solver_magnetic_reconnection/

Physical constants (CODATA 2018):
    m_p = 1.67262192369e-27 kg
    c   = 299792458.0 m/s
    q_e = 1.602176634e-19 C
"""

import math
from dataclasses import asdict, dataclass, field
from typing import List

from .blocks import (
    AMRSpec,
    DiagSpec,
    DomainSpec,
    HybridIonSpec,
    OhmSolverSpec,
    SolverSpec,
    _AMR_KEY_MAP,
    _emit_amr_block,
    suggest_cells,
)

_M_P = 1.67262192369e-27
_C   = 299792458.0
_Q_E = 1.602176634e-19
_EPS0 = 8.854187817e-12
_MU0  = 4.0 * math.pi * 1e-7

_DOMAIN_KEYS = {"dim", "number_of_cells", "lower_bound", "upper_bound", "field_bc"}
_SOLVER_KEYS = {"max_steps", "cfl", "particle_shape"}
_OHM_KEYS    = {"Te", "n0_ref", "gamma", "resistivity", "hyper_resistivity",
                 "substeps", "n_floor"}
_ION_KEYS    = {"ion_density", "ion_mass_amu", "ion_temperature_eV", "ppc"}
_DIAG_KEYS   = {"diag_period", "diag_fields"}


@dataclass
class MagneticReconnectionSpec:
    """Spec for a 2D hybrid-PIC Harris-sheet magnetic reconnection run.

    Default values are inspired by the GEM reconnection challenge:
      B0 = 0.1 T, m_ion/m_e = 400 (light ions for fast reconnection),
      T_i/T_e = 5, beta_e = 0.1.
    The current sheet half-width delta is set to the ion skin depth l_i.

    Domain: 40 l_i × 20 l_i centred on z = 0 (two Harris sheets from PBCs).
    """
    name: str = "magnetic_reconnection"
    # 2D: x spans [0, Lx], z spans [-Lz/2, Lz/2]
    domain: DomainSpec = field(
        default_factory=lambda: DomainSpec(
            dim=2,
            number_of_cells=[512, 256],
            lower_bound=[0.0, -0.025],   # ~20 l_i each side at default params
            upper_bound=[0.050, 0.025],
            field_bc=["periodic", "periodic"],
        )
    )
    solver: SolverSpec = field(
        default_factory=lambda: SolverSpec(max_steps=2000, particle_shape="linear")
    )
    ohm: OhmSolverSpec = field(
        default_factory=lambda: OhmSolverSpec(
            Te=0.05,
            n0_ref=1.0e20,
            gamma=1.0,
            resistivity=1e-7,
            hyper_resistivity=0.0,
            substeps=40,
            n_floor=1.0e17,
        )
    )
    ions: HybridIonSpec = field(
        default_factory=lambda: HybridIonSpec(
            density=1.0e20,
            mass_amu=0.22,      # ~400 m_e for fast GEM-style reconnection
            temperature_eV=0.25,
            ppc=100,
        )
    )
    const_dt: float = 1e-11     # Fixed timestep [s]
    # Harris sheet B-field parameters
    B0: float = 0.1             # Asymptotic reconnecting field [T]
    Bg: float = 0.0             # Guide field [T]; 0 = anti-parallel (no guide)
    delta: float = 1.25e-3      # Current sheet half-width [m]; default ≈ l_i
    dB_fraction: float = 0.01   # Perturbation amplitude as fraction of B0
    diag: DiagSpec = field(
        default_factory=lambda: DiagSpec(
            diag_period=100,
            diag_fields=["Bx", "By", "Bz", "Ex", "Ey", "Ez",
                         "rho", "jx_displacement", "jy_displacement", "jz_displacement"],
        )
    )
    # AMR / resolution
    amr: AMRSpec = field(default_factory=AMRSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "MagneticReconnectionSpec":
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

        ion_kwargs: dict = {}
        if "ion_density" in d:        ion_kwargs["density"]        = d["ion_density"]
        if "ion_mass_amu" in d:       ion_kwargs["mass_amu"]       = d["ion_mass_amu"]
        if "ion_temperature_eV" in d: ion_kwargs["temperature_eV"] = d["ion_temperature_eV"]
        if "ppc" in d:                ion_kwargs["ppc"]            = d["ppc"]
        ions = HybridIonSpec(**ion_kwargs)

        diag = DiagSpec(**{k: d[k] for k in _DIAG_KEYS if k in d})
        return cls(
            name=d.get("name", "magnetic_reconnection"),
            domain=domain, solver=solver, ohm=ohm, ions=ions,
            const_dt=d.get("const_dt", 1e-11),
            B0=d.get("B0", 0.1),
            Bg=d.get("Bg", 0.0),
            delta=d.get("delta", 1.25e-3),
            dB_fraction=d.get("dB_fraction", 0.01),
            diag=diag,
            amr=amr,
        )


def generate_inputs_magnetic_reconnection(spec: MagneticReconnectionSpec) -> str:
    """Generate native WarpX ParmParse inputs for Harris-sheet reconnection.

    The analytic B-field expressions are set via WarpX's parser functions
    using ``my_constants`` for the physical parameters.  The perturbation
    is a standard single-mode cos/sin pattern that breaks the equilibrium
    symmetry and seeds fast reconnection at the X-point.
    """
    if spec.domain.dim not in (1, 2):
        raise ValueError(
            f"MagneticReconnectionSpec supports dim=1 or 2; got {spec.domain.dim}"
        )

    _PSHAPE_MAP = {"linear": 1, "quadratic": 2, "cubic": 3}
    pshape = spec.solver.particle_shape
    if isinstance(pshape, str):
        pshape = _PSHAPE_MAP[pshape.lower()]

    ion_mass_kg = spec.ions.mass_amu * _M_P
    ion_v_th = math.sqrt(spec.ions.temperature_eV * _Q_E / ion_mass_kg)
    ion_u_th = ion_v_th / _C

    # Domain lengths for perturbation wavevectors
    Lx = spec.domain.upper_bound[0]  - spec.domain.lower_bound[0]
    Lz = spec.domain.upper_bound[-1] - spec.domain.lower_bound[-1]

    dB = spec.dB_fraction * spec.B0

    n_cell  = " ".join(str(n) for n in spec.domain.number_of_cells)
    prob_lo = " ".join(f"{x:.17g}" for x in spec.domain.lower_bound)
    prob_hi = " ".join(f"{x:.17g}" for x in spec.domain.upper_bound)
    bc      = " ".join(spec.domain.field_bc)

    # Analytic B-field expressions using my_constants
    # In WarpX 2D (x-z), the z axis is the last coordinate.
    bx_expr = "B0 * tanh(z / delta)"
    if spec.Bg > 0.0:
        by_expr = "sqrt(Bg**2 + B0**2 - (B0*tanh(z/delta))**2)"
    else:
        by_expr = "0.0"
    if dB > 0.0:
        bz_expr = "dB * sin(2*pi*x/Lx) * cos(pi*z/Lz)"
    else:
        bz_expr = "0.0"

    hyper_line = (
        f"hybrid_pic_model.plasma_hyper_resistivity(rho,B) = "
        f"{spec.ohm.hyper_resistivity:.17g}"
        if spec.ohm.hyper_resistivity != 0.0
        else "# hybrid_pic_model.plasma_hyper_resistivity = 0 (omitted)"
    )

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
        "ohm": asdict(spec.ohm),
        "ions": asdict(spec.ions),
        "const_dt": spec.const_dt,
        "B0": spec.B0, "Bg": spec.Bg, "delta": spec.delta,
        "dB_fraction": spec.dB_fraction,
        "diag": asdict(spec.diag),
    }

    text = f"""\
# Auto-generated by pywarpx.inputgen (magnetic_reconnection)
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

# --- Harris-sheet magnetic field (analytic) ---------------------------------
# Bx = B0*tanh(z/delta), By = guide field, Bz = seed perturbation
warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = {bx_expr}
warpx.By_external_grid_function(x,y,z) = {by_expr}
warpx.Bz_external_grid_function(x,y,z) = {bz_expr}

my_constants.B0 = {spec.B0:.17g}
my_constants.Bg = {spec.Bg:.17g}
my_constants.delta = {spec.delta:.17g}
my_constants.dB = {dB:.17g}
my_constants.Lx = {Lx:.17g}
my_constants.Lz = {Lz:.17g}

# --- Ion species (kinetic) --------------------------------------------------
particles.species_names = ions

ions.charge = q_e
ions.mass = {ion_mass_kg:.17g}
ions.injection_style = NRandomPerCell
ions.num_particles_per_cell = {spec.ions.ppc}
ions.profile = constant
ions.density = {spec.ions.density:.17g}
ions.momentum_distribution_type = gaussian
ions.ux_m = 0.0
ions.uy_m = 0.0
ions.uz_m = 0.0
ions.ux_th = {ion_u_th:.17g}
ions.uy_th = {ion_u_th:.17g}
ions.uz_th = {ion_u_th:.17g}

# --- Diagnostics ------------------------------------------------------------
{diag_section}
"""
    return text
