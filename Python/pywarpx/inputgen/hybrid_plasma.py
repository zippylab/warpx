from __future__ import annotations

"""Hybrid-PIC (kinetic ions + Ohm's law fluid electrons) input generation.

Produces native AMReX ParmParse inputs files that can be run with the WarpX
C++ binary:
    mpiexec -n N warpx.Xd inputs [key=value overrides ...]

Reference: Munoz et al. (2018), kinetic-fluid hybrid model in WarpX.

Physical constants (CODATA 2018):
    m_p = 1.67262192369e-27 kg   (proton mass)
    m_e = 9.1093837015e-31  kg   (electron mass)
    c   = 299792458.0 m/s
    q_e = 1.602176634e-19   C
"""

import math
from dataclasses import asdict, dataclass, field
from typing import List

from .blocks import (
    DiagSpec,
    DomainSpec,
    HybridIonSpec,
    OhmSolverSpec,
    SolverSpec,
)

# CODATA 2018 values
_M_P = 1.67262192369e-27  # kg  (proton mass)
_M_E = 9.1093837015e-31   # kg  (electron mass)
_C   = 299792458.0          # m/s
_Q_E = 1.602176634e-19    # C

# Flat JSON key → block mapping (mirrors laser_acceleration.py convention)
_DOMAIN_KEYS  = {"dim", "number_of_cells", "lower_bound", "upper_bound", "field_bc"}
_SOLVER_KEYS  = {"max_steps", "cfl", "particle_shape"}
_OHM_KEYS     = {"Te", "n0_ref", "gamma", "resistivity", "hyper_resistivity",
                  "substeps", "n_floor"}
_ION_KEYS     = {"ion_density", "ion_mass_amu", "ion_temperature_eV", "ppc"}
_DIAG_KEYS    = {"diag_period", "diag_fields"}


@dataclass
class HybridPlasmaSpec:
    """Spec for a hybrid-PIC run: kinetic ions + Ohm's-law fluid electrons.

    Default values correspond to the 1D EM-modes test case from Munoz et al.
    (2018) with B0 along z, B0 = 0.25 T, beta = 0.01, m_ion/m_e = 100.
    """
    name: str = "hybrid_plasma"
    # Default domain: 512 cells × 1.25e-4 m = 0.064 m (≈51 ion skin depths)
    # Self-consistent with n0_ref=3.3e22, B0=0.25 T (vA/c=1e-4, l_i≈1.25 mm).
    domain: DomainSpec = field(
        default_factory=lambda: DomainSpec(
            dim=1,
            number_of_cells=[512],
            lower_bound=[0.0],
            upper_bound=[0.064],
            field_bc=["periodic"],
        )
    )
    solver: SolverSpec = field(
        default_factory=lambda: SolverSpec(max_steps=1000, particle_shape="linear")
    )
    ohm: OhmSolverSpec = field(default_factory=OhmSolverSpec)
    ions: HybridIonSpec = field(default_factory=HybridIonSpec)
    # Fixed timestep in seconds (required by the hybrid-PIC solver; warpx.const_dt).
    # Default 1.3 ns ≈ 5e-3 * T_ci for B0=0.25 T, proton mass.
    const_dt: float = 1.3e-9
    # Applied uniform background magnetic field [Bx, By, Bz] in Tesla.
    B0: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.25])
    diag: DiagSpec = field(
        default_factory=lambda: DiagSpec(
            diag_period=100,
            diag_fields=["Bx", "By", "Bz", "Ex", "Ey", "Ez",
                         "rho", "jx_displacement", "jy_displacement", "jz_displacement"],
        )
    )

    @classmethod
    def from_dict(cls, d: dict) -> "HybridPlasmaSpec":
        """Construct from a flat JSON dict (backward-compatible external format)."""
        domain = DomainSpec(**{k: d[k] for k in _DOMAIN_KEYS if k in d})
        solver = SolverSpec(**{k: d[k] for k in _SOLVER_KEYS if k in d})

        ohm_kwargs = {k: d[k] for k in _OHM_KEYS if k in d}
        ohm = OhmSolverSpec(**ohm_kwargs)

        # Ion keys have an "ion_" prefix in the flat JSON to avoid clashes
        ion_kwargs: dict = {}
        if "ion_density" in d:
            ion_kwargs["density"] = d["ion_density"]
        if "ion_mass_amu" in d:
            ion_kwargs["mass_amu"] = d["ion_mass_amu"]
        if "ion_temperature_eV" in d:
            ion_kwargs["temperature_eV"] = d["ion_temperature_eV"]
        if "ppc" in d:
            ion_kwargs["ppc"] = d["ppc"]
        ions = HybridIonSpec(**ion_kwargs)

        diag_kwargs = {k: d[k] for k in _DIAG_KEYS if k in d}
        diag = DiagSpec(**diag_kwargs)

        return cls(
            name=d.get("name", "hybrid_plasma"),
            domain=domain, solver=solver, ohm=ohm, ions=ions,
            const_dt=d.get("const_dt", 1.3e-9),
            B0=d.get("B0", [0.0, 0.0, 0.25]),
            diag=diag,
        )


def generate_inputs_hybrid_plasma(spec: HybridPlasmaSpec) -> str:
    """Generate a native WarpX AMReX ParmParse inputs file for a hybrid-PIC run.

    Supports dim=1, 2, or 3.  The Ohm's law solver (HybridPIC) requires:
      - algo.maxwell_solver = hybrid
      - algo.current_deposition = direct
      - algo.particle_shape = 1 (linear; higher-order may introduce noise)

    The applied background magnetic field B0 is set as a constant external
    field (warpx.B_ext_grid_init_style = constant), which initialises the
    B-field grid and remains as a background field throughout the simulation.
    """

    if spec.domain.dim not in (1, 2, 3):
        raise ValueError(
            f"HybridPlasmaSpec supports dim=1, 2, or 3; got dim={spec.domain.dim}"
        )

    d = {
        "name": spec.name,
        "domain": asdict(spec.domain),
        "solver": asdict(spec.solver),
        "ohm": asdict(spec.ohm),
        "ions": asdict(spec.ions),
        "const_dt": spec.const_dt,
        "B0": spec.B0,
        "diag": asdict(spec.diag),
    }

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    n_cell   = " ".join(str(n) for n in spec.domain.number_of_cells)
    prob_lo  = " ".join(f"{x:.17g}" for x in spec.domain.lower_bound)
    prob_hi  = " ".join(f"{x:.17g}" for x in spec.domain.upper_bound)
    field_lo = " ".join(bc for bc in spec.domain.field_bc)
    field_hi = field_lo   # symmetric

    # ------------------------------------------------------------------
    # Particle shape integer
    # ------------------------------------------------------------------
    _PSHAPE_MAP = {"linear": 1, "quadratic": 2, "cubic": 3}
    pshape = spec.solver.particle_shape
    if isinstance(pshape, str):
        pshape = _PSHAPE_MAP[pshape.lower()]

    # ------------------------------------------------------------------
    # Ion species: derived quantities
    # ------------------------------------------------------------------
    ion_mass_kg = spec.ions.mass_amu * _M_P
    # Thermal velocity in m/s from temperature in eV
    ion_v_th = math.sqrt(spec.ions.temperature_eV * _Q_E / ion_mass_kg)
    # Normalised momentum u_th = v_th / c  (WarpX convention for ux_th etc.)
    ion_u_th = ion_v_th / _C

    # ------------------------------------------------------------------
    # Background B field
    # ------------------------------------------------------------------
    B0 = spec.B0
    bx, by, bz = B0[0], B0[1], B0[2]

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
    # Hyper-resistivity line (omit entirely when zero to keep file clean)
    # ------------------------------------------------------------------
    hyper_line = (
        f"hybrid_pic_model.plasma_hyper_resistivity(rho,B) = "
        f"{spec.ohm.hyper_resistivity:.17g}"
        if spec.ohm.hyper_resistivity != 0.0
        else "# hybrid_pic_model.plasma_hyper_resistivity = 0 (omitted)"
    )

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    text = f"""\
# Auto-generated by pywarpx.inputgen (hybrid_plasma)
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
boundary.field_lo = {field_lo}
boundary.field_hi = {field_hi}
boundary.particle_lo = {field_lo}
boundary.particle_hi = {field_hi}

# --- Numerics ---------------------------------------------------------------
warpx.verbose = 1
warpx.const_dt = {spec.const_dt:.17g}

algo.maxwell_solver = hybrid
algo.particle_shape = {pshape}
algo.current_deposition = direct

# --- Hybrid-PIC model (Ohm's law) -------------------------------------------
# Electrons: isothermal (gamma=1) or adiabatic (gamma=5/3) fluid
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
