from __future__ import annotations

import ast
import math
from typing import List, Tuple

from .spec import Severity, UniformPlasmaSpec, ValidationReport


def _cell_sizes(spec: UniformPlasmaSpec) -> List[float]:
    dx = []
    for n, lo, hi in zip(spec.number_of_cells, spec.lower_bound, spec.upper_bound):
        dx.append((hi - lo) / float(n))
    return dx


def _debye_length_m(temperature_eV: float, density_m3: float) -> float:
    # lambda_D = sqrt(eps0 * kT / (n e^2))
    # kT in Joules: T_eV * q_e
    eps0 = 8.8541878188e-12
    q_e = 1.602176634e-19
    kT = temperature_eV * q_e
    return math.sqrt(eps0 * kT / (density_m3 * q_e * q_e))


def validate_uniform_plasma_spec(spec: UniformPlasmaSpec) -> ValidationReport:
    r = ValidationReport()

    # Basic schema checks
    if spec.dim not in (1, 2, 3):
        r.add(Severity.ERROR, "spec.dim.invalid", "dim must be 1, 2, or 3", dim=spec.dim)
        return r

    def _len_ok(name: str, v: List[float | int | str]) -> None:
        if len(v) != spec.dim:
            r.add(
                Severity.ERROR,
                f"spec.{name}.len",
                f"{name} must have length dim={spec.dim}",
                length=len(v),
                dim=spec.dim,
            )

    _len_ok("number_of_cells", spec.number_of_cells)
    _len_ok("lower_bound", spec.lower_bound)
    _len_ok("upper_bound", spec.upper_bound)
    _len_ok("field_bc", spec.field_bc)

    if not r.ok:
        return r

    if any(n <= 0 for n in spec.number_of_cells):
        r.add(Severity.ERROR, "spec.number_of_cells.nonpositive", "All cell counts must be > 0")

    if any(hi <= lo for lo, hi in zip(spec.lower_bound, spec.upper_bound)):
        r.add(Severity.ERROR, "spec.bounds.invalid", "Each upper_bound must be > lower_bound")

    if spec.max_steps <= 0:
        r.add(Severity.ERROR, "spec.max_steps.nonpositive", "max_steps must be > 0", max_steps=spec.max_steps)

    if not (0.0 < spec.cfl <= 1.0):
        r.add(Severity.WARNING, "spec.cfl.suspicious", "cfl should be in (0, 1] for FDTD stability", cfl=spec.cfl)

    if spec.density <= 0:
        r.add(Severity.ERROR, "spec.density.nonpositive", "density must be > 0", density=spec.density)

    if spec.temperature_eV <= 0:
        r.add(Severity.WARNING, "spec.temperature.nonpositive", "temperature_eV should be > 0", temperature_eV=spec.temperature_eV)

    # Physics sanity checks
    if r.ok:
        dx = _cell_sizes(spec)
        dx_min = min(dx)

        # Simple Debye-length resolution heuristic.
        # This is not universally required (e.g., implicit/hybrid), but it's a strong warning
        # for explicit EM plasma simulations.
        try:
            lamD = _debye_length_m(spec.temperature_eV, spec.density)
            if dx_min > 10.0 * lamD:
                r.add(
                    Severity.WARNING,
                    "physics.debye.underresolved",
                    "Grid spacing appears much larger than Debye length; results may be unphysical/noisy",
                    dx_min=dx_min,
                    debye_length=lamD,
                    ratio=dx_min / lamD,
                )
        except Exception as e:
            r.add(Severity.WARNING, "physics.debye.compute_failed", "Failed to compute Debye length", error=str(e))

    # Time step check: if user provided dt, compare to CFL estimate for Yee.
    if r.ok and spec.time_step_size is not None:
        c = 299792458.0
        dx = _cell_sizes(spec)
        inv_dx2 = sum((1.0 / d) ** 2 for d in dx)
        dt_cfl = 1.0 / (c * math.sqrt(inv_dx2))
        if spec.time_step_size > spec.cfl * dt_cfl * 1.01:
            r.add(
                Severity.WARNING,
                "physics.cfl.exceeded",
                "Provided time_step_size exceeds CFL-based estimate for Yee; simulation may be unstable",
                dt=spec.time_step_size,
                dt_cfl=dt_cfl,
                cfl=spec.cfl,
            )

    # AMR blocking_factor divisibility
    bf = spec.amr_blocking_factor
    if spec.amr_max_level < 0:
        r.add(Severity.ERROR, "amr.max_level",
              "amr_max_level must be >= 0", amr_max_level=spec.amr_max_level)
    elif spec.amr_max_level > 0:
        if bf < 1 or (bf & (bf - 1)) != 0:
            r.add(Severity.ERROR, "amr.blocking_factor",
                  "amr_blocking_factor must be a power of 2", amr_blocking_factor=bf)
        else:
            bad = [n for n in spec.number_of_cells if n % bf != 0]
            if bad:
                r.add(Severity.ERROR, "amr.blocking_factor.divisibility",
                      f"number_of_cells must be divisible by amr_blocking_factor={bf}; "
                      f"offending counts: {bad}",
                      amr_blocking_factor=bf, bad_cells=bad)

    # Diagnostics
    if spec.diag_period <= 0:
        r.add(Severity.WARNING, "spec.diag_period.nonpositive", "diag_period should be > 0", diag_period=spec.diag_period)

    if not spec.diag_fields:
        r.add(Severity.WARNING, "spec.diag_fields.empty", "No field diagnostics selected; run may produce little output")

    return r


def validate_picmi_syntax(picmi_script_text: str) -> Tuple[ValidationReport, ast.AST | None]:
    """Cheap validation: parse the generated PICMI script.

    This catches trivial syntax errors before running on an HPC system.
    """

    r = ValidationReport()
    try:
        tree = ast.parse(picmi_script_text)
        return r, tree
    except SyntaxError as e:
        r.add(
            Severity.ERROR,
            "picmi.syntax",
            "Generated PICMI script is not valid Python",
            lineno=getattr(e, "lineno", None),
            offset=getattr(e, "offset", None),
            text=getattr(e, "text", None),
            msg=str(e),
        )
        return r, None
