from __future__ import annotations

from dataclasses import asdict
from textwrap import dedent

from .spec import UniformPlasmaSpec


def _grid_class(dim: int) -> str:
    if dim == 1:
        return "Cartesian1DGrid"
    if dim == 2:
        return "Cartesian2DGrid"
    if dim == 3:
        return "Cartesian3DGrid"
    raise ValueError(f"Unsupported dim={dim}")


def generate_picmi_uniform_plasma(spec: UniformPlasmaSpec) -> str:
    """Render a minimal PICMI script for a uniform plasma test.

    Notes:
    - This is intentionally conservative and uses a simple EM solver.
    - We keep the script small and deterministic to support validation and templating.
    """

    d = asdict(spec)
    grid_cls = _grid_class(spec.dim)

    warpx_max_grid_size_line = ""
    if spec.warpx_max_grid_size is not None:
        warpx_max_grid_size_line = f"    warpx_max_grid_size={spec.warpx_max_grid_size},\n"

    amr_lines = ""
    if spec.amr_max_level > 0:
        amr_lines = (
            f"    warpx_max_level={spec.amr_max_level},\n"
            f"    warpx_blocking_factor={spec.amr_blocking_factor},\n"
        )

    # If time_step_size is None, PICMI will compute dt from solver cfl.
    dt_line = ""
    if spec.time_step_size is not None:
        dt_line = f"    time_step_size={spec.time_step_size},\n"

    script = f"""#!/usr/bin/env python3

from pywarpx import picmi

constants = picmi.constants

# Spec (for provenance)
SPEC = {d!r}

# Grid
grid = picmi.{grid_cls}(
    number_of_cells={spec.number_of_cells!r},
    lower_bound={spec.lower_bound!r},
    upper_bound={spec.upper_bound!r},
    lower_boundary_conditions={spec.field_bc!r},
    upper_boundary_conditions={spec.field_bc!r},
{warpx_max_grid_size_line.rstrip()}
)

# Solver (electromagnetic)
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="Yee",
    cfl={spec.cfl!r},
)

# Species
uniform = picmi.UniformDistribution(
    density={spec.density!r},
    fill_in=True,
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=uniform,
)

# Diagnostics
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period={spec.diag_period!r},
    data_list={spec.diag_fields!r},
)

# Simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps={spec.max_steps!r},
{dt_line.rstrip()}
{amr_lines.rstrip()}
    particle_shape={spec.particle_shape!r},
    warpx_current_deposition_algo={spec.current_deposition_algo!r},
    warpx_use_filter=0,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[1]*{spec.dim}),
)

sim.add_diagnostic(field_diag)

# Init + run
sim.initialize_inputs()
sim.initialize_warpx()
sim.step({spec.max_steps!r})
"""

    # Dedent to avoid blank indentation issues in optional lines
    return dedent(script).lstrip()
