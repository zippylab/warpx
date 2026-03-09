from __future__ import annotations

"""Native AMReX ParmParse input generation for laser acceleration (LWFA-style).

Produces inputs files that can be fed directly to the WarpX C++ binary:
    mpiexec -n N warpx.Xd inputs [key=value overrides ...]

No Python bindings or PICMI required at run time.

Physical constants (CODATA 2018, matching scipy.constants / picmi):
    m_e = 9.1093837015e-31 kg
    c   = 299792458.0 m/s
    q_e = 1.602176634e-19 C
"""

import math
from dataclasses import asdict

from .blocks import _emit_amr_block, _emit_diag_block
from .laser_acceleration import LaserAccelerationSpec

# CODATA 2018 values (matches pywarpx / PICMI constants)
_M_E = 9.1093837015e-31   # kg
_C   = 299792458.0         # m/s
_Q_E = 1.602176634e-19    # C

_PARTICLE_SHAPE_MAP = {
    "linear":    1,
    "quadratic": 2,
    "cubic":     3,
}


def _particle_shape_int(shape) -> int:
    if isinstance(shape, int):
        return shape
    return _PARTICLE_SHAPE_MAP[shape.lower()]


def _field_bc_to_native(bc: str) -> str:
    """Map PICMI-style field BC name to native WarpX name."""
    return "pml" if bc == "open" else bc


def generate_inputs_laser_acceleration(spec: LaserAccelerationSpec) -> str:
    """Generate a native WarpX AMReX ParmParse inputs file for an LWFA-style run.

    The output is designed to match what PICMI's ``sim.write_input_file()`` would
    produce for the same :class:`LaserAccelerationSpec`, as validated against the
    reference file ``airun_laser_sunspot/inputs_laser``.

    Supports dim=2 and dim=3.
    """

    if spec.domain.dim not in (2, 3):
        raise ValueError(
            f"laser_acceleration_native supports dim=2 or dim=3, got dim={spec.domain.dim}"
        )

    d = {
        "name": spec.name,
        "domain": asdict(spec.domain),
        "solver": asdict(spec.solver),
        "species": asdict(spec.species),
        "laser": asdict(spec.laser),
        "diag": asdict(spec.diag),
    }

    # ------------------------------------------------------------------
    # Derived laser quantities
    # ------------------------------------------------------------------
    # Antenna sits just inside the lower z boundary to avoid floating-point
    # edge comparisons — same offset as the PICMI generator.
    # z is always the last axis.
    _antenna_offset = 1.0e-12
    antenna_z = spec.domain.lower_bound[-1] + _antenna_offset

    # Peak field amplitude from normalised vector potential a0.
    # Formula: e_max = a0 * m_e * c * omega / q_e  where omega = 2π c / λ
    e_max = spec.laser.a0 * _M_E * _C**2 * 2.0 * math.pi / (_Q_E * spec.laser.wavelength)

    # Time at which the pulse peak reaches the antenna plane.
    # centroid_position_z is the z-position of the pulse centre at t=0.
    t_peak = (antenna_z - spec.laser.centroid_position_z) / _C

    # Focal distance measured from the antenna plane.
    focal_distance = spec.laser.focal_position_z - antenna_z

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    # Field BCs: "open" → "pml"; everything else passed through.
    # Particle BCs: use the spec names directly ("open", "periodic", …).
    field_lo = " ".join(_field_bc_to_native(bc) for bc in spec.domain.field_bc)
    field_hi = field_lo   # symmetric
    particle_lo = " ".join(spec.domain.field_bc)
    particle_hi = particle_lo

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------
    pshape = _particle_shape_int(spec.solver.particle_shape)

    n_cell = " ".join(str(n) for n in spec.domain.number_of_cells)
    prob_lo = " ".join(f"{x:.17g}" for x in spec.domain.lower_bound)
    prob_hi = " ".join(f"{x:.17g}" for x in spec.domain.upper_bound)

    # Implicit solver section (only emitted when enabled)
    imp = spec.implicit
    if imp.enabled:
        timestep_line = f"warpx.const_dt = {imp.const_dt:.17g}"
        implicit_block = f"""\

# --- Implicit EM solver (theta = {imp.theta}) --------------------------------
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
    # Diagnostics
    # ------------------------------------------------------------------
    # Use openPMD format for production-quality output.  If libopenPMD is
    # not available on the target system, change format to "plotfile".
    diag_section = _emit_diag_block(spec.diag)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    text = f"""\
# Auto-generated by pywarpx.inputgen (laser_acceleration_native)
# SPEC = {d!r}

# --- Time -------------------------------------------------------------------
max_step = {spec.solver.max_steps}

# --- AMR / domain -----------------------------------------------------------
{_emit_amr_block(spec.amr, n_cell, prob_lo, prob_hi, spec.domain.dim)}

# --- Boundary conditions ----------------------------------------------------
boundary.field_lo = {field_lo}
boundary.field_hi = {field_hi}
boundary.particle_lo = {particle_lo}
boundary.particle_hi = {particle_hi}

# --- Numerics ---------------------------------------------------------------
warpx.verbose = 1
{timestep_line}
warpx.use_filter = 0

algo.maxwell_solver = CKC
algo.particle_shape = {pshape}{implicit_block}

# --- Species ----------------------------------------------------------------
particles.species_names = electrons ions

electrons.charge = -q_e
electrons.mass = m_e
electrons.injection_style = NUniformPerCell
electrons.num_particles_per_cell_each_dim = 2 2 2
electrons.profile = constant
electrons.density = {spec.species.plasma_density:.17g}
electrons.zmin = {spec.species.plasma_zmin:.17g}
electrons.zmax = {spec.species.plasma_zmax:.17g}
electrons.momentum_distribution_type = constant
electrons.ux = 0.0
electrons.uy = 0.0
electrons.uz = 0.0
electrons.initialize_self_fields = 0
electrons.do_continuous_injection = 1

ions.charge = q_e
ions.mass = m_p
ions.injection_style = NUniformPerCell
ions.num_particles_per_cell_each_dim = 2 2 2
ions.profile = constant
ions.density = {spec.species.plasma_density:.17g}
ions.zmin = {spec.species.plasma_zmin:.17g}
ions.zmax = {spec.species.plasma_zmax:.17g}
ions.momentum_distribution_type = constant
ions.ux = 0.0
ions.uy = 0.0
ions.uz = 0.0
ions.initialize_self_fields = 0
ions.do_continuous_injection = 1

# --- Laser ------------------------------------------------------------------
lasers.names = laser1

laser1.profile = Gaussian
laser1.position = 0.0 0.0 {antenna_z:.17g}
laser1.direction = 0 0 1
laser1.polarization = 1 0 0
laser1.wavelength = {spec.laser.wavelength:.17g}
laser1.e_max = {e_max:.17g}
laser1.profile_waist = {spec.laser.waist:.17g}
laser1.profile_duration = {spec.laser.duration:.17g}
laser1.profile_t_peak = {t_peak:.17g}
laser1.profile_focal_distance = {focal_distance:.17g}
laser1.do_continuous_injection = 0

# --- Diagnostics ------------------------------------------------------------
{diag_section}
"""

    return text
