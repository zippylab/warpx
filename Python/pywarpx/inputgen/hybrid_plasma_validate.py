from __future__ import annotations

import math

from .blocks import (
    validate_amr,
    validate_diag,
    validate_domain,
    validate_hybrid_ion,
    validate_ohm_solver,
    validate_solver,
)
from .hybrid_plasma import HybridPlasmaSpec
from .spec import Severity, ValidationReport


def validate_hybrid_plasma_spec(spec: HybridPlasmaSpec) -> ValidationReport:
    r = ValidationReport()

    r.merge(validate_domain(spec.domain, allowed_dims=(1, 2, 3)))
    if not r.ok:
        return r

    r.merge(validate_amr(spec.amr, spec.domain))
    r.merge(validate_solver(spec.solver))
    r.merge(validate_ohm_solver(spec.ohm))
    r.merge(validate_hybrid_ion(spec.ions))
    r.merge(validate_diag(spec.diag))

    _check_const_dt(r, spec)
    _check_b0(r, spec)
    _check_density_consistency(r, spec)
    _check_whistler_cfl(r, spec)

    return r


def _check_const_dt(r: ValidationReport, spec: HybridPlasmaSpec) -> None:
    if spec.const_dt <= 0:
        r.add(Severity.ERROR, "hybrid.const_dt",
              "const_dt must be > 0 (required by the hybrid-PIC solver)",
              const_dt=spec.const_dt)


def _check_b0(r: ValidationReport, spec: HybridPlasmaSpec) -> None:
    if len(spec.B0) != 3:
        r.add(Severity.ERROR, "hybrid.B0.len",
              "B0 must be a 3-element list [Bx, By, Bz]", got=len(spec.B0))
        return
    if all(b == 0.0 for b in spec.B0):
        r.add(Severity.WARNING, "hybrid.B0.zero",
              "B0 = [0,0,0]: a zero background field may cause numerical issues "
              "in the Ohm's law solver")


def _check_whistler_cfl(r: ValidationReport, spec: HybridPlasmaSpec) -> None:
    """Warn if sub-inertial whistler modes at the Nyquist wavenumber are unstable.

    For explicit RK4 subcycling of B, the whistler wave at k_max = π/dx must satisfy:

        (k_max * l_i)² * ω_ci * dt_sub  <  2√2  ≈ 2.83

    where l_i = c/ω_pi is the ion skin depth and dt_sub = const_dt / substeps.
    When dx ≪ l_i the Nyquist-scale whistler frequency is enormous and the
    simulation will produce E-field NaN even with many substeps.  The fix is to
    set dx ≈ 0.1 * l_i (i.e. choose n and domain size self-consistently).
    """
    _EPS0 = 8.854187817e-12
    _M_P  = 1.67262192369e-27
    _Q_E  = 1.602176634e-19
    _C    = 299792458.0
    _RK4_LIMIT = 2.0 * math.sqrt(2.0)   # ≈ 2.828

    B = math.sqrt(sum(b ** 2 for b in spec.B0))
    if B == 0.0:
        return  # zero-B handled elsewhere

    n     = spec.ohm.n0_ref
    m_i   = spec.ions.mass_amu * _M_P
    omega_ci = _Q_E * B / m_i
    omega_pi = math.sqrt(n * _Q_E ** 2 / (m_i * _EPS0))
    l_i      = _C / omega_pi

    # minimum cell size across all axes
    dx = min(
        (hi - lo) / nc
        for lo, hi, nc in zip(
            spec.domain.lower_bound,
            spec.domain.upper_bound,
            spec.domain.number_of_cells,
        )
    )

    dt_sub   = spec.const_dt / spec.ohm.substeps
    k_nyq_li = math.pi * l_i / dx          # k_Nyquist × l_i
    z_max    = k_nyq_li ** 2 * omega_ci * dt_sub

    if z_max > _RK4_LIMIT:
        substeps_min = int(
            math.ceil(spec.const_dt * omega_ci * k_nyq_li ** 2 / _RK4_LIMIT)
        )
        r.add(
            Severity.WARNING,
            "hybrid.cfl.whistler",
            (
                f"Whistler CFL z={z_max:.3g} > {_RK4_LIMIT:.3g}: "
                f"sub-inertial Nyquist modes are numerically unstable and will "
                f"cause E-field blow-up regardless of substep count. "
                f"Ion skin depth l_i={l_i:.3e} m; dx={dx:.3e} m (dx/l_i={dx/l_i:.4f}). "
                f"Required substeps>={substeps_min} (likely impractical). "
                f"Recommended fix: set dx <= l_i/10 = {l_i/10:.3e} m by using "
                f"n0_ref >= {(math.pi * _C / (10 * dx)) ** 2 * m_i * _EPS0 / _Q_E ** 2:.2e} m^-3 "
                f"or a coarser grid."
            ),
            z_whistler=round(z_max, 3),
            substeps_min=substeps_min,
            l_i=round(l_i, 6),
            dx=round(dx, 8),
            dx_over_l_i=round(dx / l_i, 6),
        )


def _check_density_consistency(r: ValidationReport, spec: HybridPlasmaSpec) -> None:
    """Warn if ion density and Ohm solver reference density differ by more than 10×."""
    ratio = spec.ions.density / spec.ohm.n0_ref
    if ratio < 0.1 or ratio > 10.0:
        r.add(
            Severity.WARNING,
            "hybrid.density_mismatch",
            "Ion density and Ohm solver n0_ref differ by more than 10×; "
            "check that n0_ref is representative of the plasma density",
            ion_density=spec.ions.density,
            n0_ref=spec.ohm.n0_ref,
            ratio=ratio,
        )
