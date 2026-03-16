from __future__ import annotations

"""Validator for ElectrostaticPICSpec."""

import math

from .blocks import (
    Severity,
    ValidationReport,
    check_boris_stability,
    check_debye_resolution,
    check_fft_requires_periodic,
    check_periodic_bc_symmetry,
    validate_amr,
    validate_collision,
    validate_diag,
    validate_domain,
    validate_eb,
    validate_es_solver,
    validate_species_def,
)
from .electrostatic_pic import ElectrostaticPICSpec


def _check_bc_lengths(spec: ElectrostaticPICSpec, r: ValidationReport) -> None:
    dim = spec.domain.dim
    for attr, val in [("field_bc_lo", spec.field_bc_lo), ("field_bc_hi", spec.field_bc_hi)]:
        if val is not None and len(val) != dim:
            r.add(
                Severity.ERROR, f"es.{attr}.len",
                f"{attr} must have length {dim} (one entry per axis)",
                attr=attr, got=len(val), expected=dim,
            )


def _check_pec_potentials(spec: ElectrostaticPICSpec, r: ValidationReport) -> None:
    """Check PEC boundary-potential consistency.

    Two sub-checks:
    1. PEC-without-potential NOTE: a PEC axis that has no corresponding
       boundary_potential entry will default to 0 V in WarpX — flag it so
       the user knows and can confirm intent.
    2. All-equal-potential DEGENERATE WARNING: if all PEC walls share the same
       potential AND the plasma is quasi-neutral (|net q| / max density < 1%),
       the Poisson solution is trivially φ=const with E=0 everywhere — no
       sheath will form.
    """
    if spec.solver.poisson_solver != "multigrid":
        return

    # Resolved per-axis BC lists
    field_lo = spec.field_bc_lo if spec.field_bc_lo is not None else spec.domain.field_bc
    field_hi = spec.field_bc_hi if spec.field_bc_hi is not None else spec.domain.field_bc
    dim = spec.domain.dim

    # --- Sub-check 1: PEC without explicit potential -------------------------
    pot_lo = spec.boundary_potential_lo or []
    pot_hi = spec.boundary_potential_hi or []
    for axis_idx, (bc_lo, bc_hi) in enumerate(zip(field_lo, field_hi)):
        pot_lo_val = pot_lo[axis_idx] if axis_idx < len(pot_lo) else None
        pot_hi_val = pot_hi[axis_idx] if axis_idx < len(pot_hi) else None
        if bc_lo in ("pec", "dirichlet") and pot_lo_val is None:
            r.add(
                Severity.WARNING, "es.potential.unspecified",
                f"field_bc_lo[{axis_idx}]='pec' but boundary_potential_lo[{axis_idx}] "
                f"is not set; WarpX defaults to 0 V.  Set boundary_potential_lo to "
                f"confirm intent.",
                axis=axis_idx,
            )
        if bc_hi in ("pec", "dirichlet") and pot_hi_val is None:
            r.add(
                Severity.WARNING, "es.potential.unspecified",
                f"field_bc_hi[{axis_idx}]='pec' but boundary_potential_hi[{axis_idx}] "
                f"is not set; WarpX defaults to 0 V.  Set boundary_potential_hi to "
                f"confirm intent.",
                axis=axis_idx,
            )

    # --- Sub-check 2: all-equal-potential degenerate case -------------------
    # Collect every potential that applies to a PEC wall.
    # Unspecified PEC potentials default to 0 V in WarpX.
    pec_potentials: list[float] = []
    for axis_idx, (bc_lo, bc_hi) in enumerate(zip(field_lo, field_hi)):
        if bc_lo in ("pec", "dirichlet"):
            v = pot_lo[axis_idx] if axis_idx < len(pot_lo) and pot_lo[axis_idx] is not None else 0.0
            pec_potentials.append(v)
        if bc_hi in ("pec", "dirichlet"):
            v = pot_hi[axis_idx] if axis_idx < len(pot_hi) and pot_hi[axis_idx] is not None else 0.0
            pec_potentials.append(v)

    if len(pec_potentials) < 2:
        return  # need at least two PEC walls for the check to be meaningful

    all_equal = all(abs(v - pec_potentials[0]) < 1e-12 * (abs(pec_potentials[0]) + 1.0)
                    for v in pec_potentials)
    if not all_equal:
        return

    # Check quasi-neutrality of species
    densities = [sp.density for sp in spec.species if sp.density > 0]
    if not densities:
        return
    max_density = max(densities)
    net_charge_density = sum(
        sp.density * sp.charge for sp in spec.species if sp.density > 0
    )
    quasi_neutral = abs(net_charge_density) / max_density < 0.01

    if quasi_neutral:
        r.add(
            Severity.WARNING,
            "es.potential.degenerate",
            (
                f"All {len(pec_potentials)} PEC walls are at the same potential "
                f"({pec_potentials[0]:.4g} V) and the plasma is quasi-neutral "
                f"(|net charge|/max_density = "
                f"{abs(net_charge_density)/max_density:.2e} < 1%).  "
                f"The Poisson solution is trivially φ=const everywhere with E=0 "
                f"— no sheath or electric field will form.  "
                f"For a sheath simulation, set different potentials on opposing "
                f"walls, e.g. boundary_potential_lo=[0.0] and "
                f"boundary_potential_hi=[-30.0] along the sheath axis."
            ),
            pec_potential=pec_potentials[0],
            net_charge_density_rel=abs(net_charge_density) / max_density,
        )


def validate_electrostatic_pic_spec(spec: ElectrostaticPICSpec) -> ValidationReport:
    """Validate an ElectrostaticPICSpec.

    Runs per-block validators then cross-block checks (species product
    references, collision species references, BC length consistency,
    FFT/periodic BC compatibility, Debye resolution, plasma frequency stability).
    """
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_es_solver(spec.solver))
    r.merge(validate_diag(spec.diag))

    if spec.eb is not None:
        r.merge(validate_eb(spec.eb))

    # Build name set for cross-reference checks
    all_names: set = {sp.name for sp in spec.species}

    for sp in spec.species:
        r.merge(validate_species_def(sp, all_names))

    for col in spec.collisions:
        r.merge(validate_collision(col, all_names))

    _check_bc_lengths(spec, r)
    _check_pec_potentials(spec, r)
    check_fft_requires_periodic(
        spec.solver.poisson_solver, spec.domain.field_bc, r, code_prefix="es"
    )
    domain_bc = spec.domain.field_bc
    lo_bcs = spec.field_bc_lo if spec.field_bc_lo is not None else domain_bc
    hi_bcs = spec.field_bc_hi if spec.field_bc_hi is not None else domain_bc
    check_periodic_bc_symmetry(spec.domain.dim, lo_bcs, hi_bcs, r, code_prefix="es")

    # All-Neumann singularity check: multigrid Poisson is singular without any
    # Dirichlet (PEC) boundary.  AMReX MLMG enforces solvability by subtracting
    # the mean of rho; for a uniform charge distribution this zeroes the RHS
    # silently, giving phi=0 and E=0 at every step.  The 1D tridiagonal solver
    # handles Neumann correctly, so the check only applies to dim >= 2.
    if (
        spec.domain.dim >= 2
        and spec.solver.poisson_solver == "multigrid"
    ):
        all_bcs = list(spec.domain.field_bc)
        if spec.field_bc_lo:
            all_bcs += list(spec.field_bc_lo)
        if spec.field_bc_hi:
            all_bcs += list(spec.field_bc_hi)
        has_dirichlet = any(b in ("pec", "dirichlet") for b in all_bcs)
        has_neumann = any(b == "neumann" for b in all_bcs)
        if not has_dirichlet and has_neumann:
            r.add(
                Severity.ERROR,
                "es.multigrid.singular",
                (
                    "All boundary conditions are Neumann (or periodic) — the "
                    "3D/2D multigrid Poisson problem is singular.  AMReX MLMG "
                    "enforces solvability by subtracting the mean of rho, which "
                    "for a spatially uniform charge distribution produces a zero "
                    "right-hand side and phi=0 at every step (silent physics "
                    "failure).  Add at least one PEC (Dirichlet) boundary, e.g. "
                    "field_bc=['periodic','periodic','pec'] for a sheath geometry."
                ),
                bcs=all_bcs,
            )

    # Debye resolution and plasma-frequency stability checks (electron species only)
    electron_specs = [
        sp for sp in spec.species
        if sp.charge == -1.0 and sp.injection_style != "none"
    ]
    if electron_specs:
        dx_max = max(
            (hi - lo) / nc
            for lo, hi, nc in zip(
                spec.domain.lower_bound,
                spec.domain.upper_bound,
                spec.domain.number_of_cells,
            )
        )
        # Debye check: use highest-density electron with known temperature
        e_with_temp = [sp for sp in electron_specs if sp.density > 0 and sp.temperature_eV > 0]
        if e_with_temp:
            ref = max(e_with_temp, key=lambda sp: sp.density)
            check_debye_resolution(
                dx_max, ref.density, ref.temperature_eV, r,
                code_prefix="es", species_name=ref.name,
            )
        # Boris stability: use highest-density electron species
        highest_n = max(sp.density for sp in electron_specs)
        check_boris_stability(spec.solver.const_dt, highest_n, r, code_prefix="es")

    return r
