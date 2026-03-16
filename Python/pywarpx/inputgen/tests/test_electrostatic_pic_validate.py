"""Tests for ElectrostaticPICSpec validation and input generation."""

from pywarpx.inputgen.electrostatic_pic import (
    ElectrostaticPICSpec,
    generate_inputs_electrostatic_pic,
)
from pywarpx.inputgen.electrostatic_pic_validate import validate_electrostatic_pic_spec
from pywarpx.inputgen.spec import Severity


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TWO_SPECIES = [
    {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e16, "ppc": 16,
     "temperature_eV": 1.0},
    {"name": "ions", "charge": 1, "mass_amu": 1.00728, "density": 1e16, "ppc": 16},
]


def _spec(**overrides) -> ElectrostaticPICSpec:
    d = {"species": _TWO_SPECIES, "const_dt": 1e-11}
    d.update(overrides)
    return ElectrostaticPICSpec.from_dict(d)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_es_pic_validate_ok():
    """Default two-species ES-PIC spec passes validation."""
    spec = _spec()
    r = validate_electrostatic_pic_spec(spec)
    assert r.ok, r.issues


def test_es_pic_bad_poisson_solver():
    """Invalid poisson_solver → ERROR."""
    spec = _spec(poisson_solver="cg")
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "es.poisson_solver" for i in r.issues)


def test_es_pic_higuera_pusher_error():
    """higuera particle_pusher in ES-PIC → ERROR (it's for implicit EM only)."""
    spec = _spec(particle_pusher="higuera")
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "es.particle_pusher" for i in r.issues)


def test_es_pic_plasma_freq_unstable():
    """dt * omega_pe >= 2 → ERROR."""
    # omega_pe(n=1e20) ≈ 1.78e10 rad/s; dt=2e-10 → dt*omega_pe ≈ 3.56 → ERROR
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e20,
         "ppc": 16, "temperature_eV": 1.0},
        {"name": "ions", "charge": 1, "mass_amu": 1.00728, "density": 1e20, "ppc": 16},
    ]
    spec = ElectrostaticPICSpec.from_dict({"species": species, "const_dt": 2e-10})
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "es.debye.dt_omega_pe" and i.severity == Severity.ERROR
               for i in r.issues)


def test_es_pic_plasma_freq_marginal():
    """dt * omega_pe > 0.2 but < 2 → WARNING."""
    # n=1e16, omega_pe ≈ 5.6e9 rad/s; dt=4e-11 → dt*omega_pe ≈ 0.225
    spec = _spec(const_dt=4e-11)
    r = validate_electrostatic_pic_spec(spec)
    assert any(i.code == "es.debye.dt_omega_pe" and i.severity == Severity.WARNING
               for i in r.issues)


def test_es_pic_ionization_product_missing():
    """Ionization product species not in list → ERROR."""
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e18, "ppc": 16},
        {"name": "argon", "charge": 18, "mass_amu": 39.948, "density": 1e18, "ppc": 4,
         "do_field_ionization": True, "physical_element": "Ar",
         "ionization_product_species": "ionized_e"},  # not in list
    ]
    spec = ElectrostaticPICSpec.from_dict({"species": species, "const_dt": 1e-11})
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "species.ionization.product" for i in r.issues)


def test_es_pic_collision_bad_type():
    """Collision with unrecognised type → ERROR."""
    spec = _spec(collisions=[{
        "name": "col1", "type": "nuclear",  # invalid; should be "nuclearfusion"
        "species": ["electrons", "ions"],
    }])
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "collision.type" for i in r.issues)


def test_es_pic_debye_resolution_warns():
    """1 < dx/λ_De ≤ 2 → WARNING only (not error)."""
    # electrons: n=1e16, Te=1eV → λ_De ≈ 74.3 μm; 10 cells over 1.115mm → dx ≈ 111.5 μm ≈ 1.5 λ_De
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e16,
         "ppc": 16, "temperature_eV": 1.0},
        {"name": "ions", "charge": 1, "mass_amu": 1.00728, "density": 1e16, "ppc": 16},
    ]
    spec = ElectrostaticPICSpec.from_dict({
        "species": species, "const_dt": 1e-12,
        "dim": 1, "number_of_cells": [10], "lower_bound": [0.0], "upper_bound": [1.115e-3],
        "field_bc": ["periodic"],
    })
    r = validate_electrostatic_pic_spec(spec)
    assert r.ok  # warning only
    issues = [i for i in r.issues if i.code == "es.debye_resolution"]
    assert issues and issues[0].severity == Severity.WARNING


def test_es_pic_debye_resolution_error():
    """dx/λ_De > 2 → ERROR (finite-grid instability guaranteed)."""
    # 3D 64^3, 10cm box, n=1e16, Te=1eV → dx=1.5625mm, λ_De=74μm, dx/λ_De≈21 → ERROR
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e16,
         "ppc": 8, "temperature_eV": 1.0},
        {"name": "protons", "charge": 1, "mass_amu": 1.00728, "density": 1e16, "ppc": 8},
    ]
    spec = ElectrostaticPICSpec.from_dict({
        "species": species, "const_dt": 2e-11,
        "dim": 3, "number_of_cells": [64, 64, 64],
        "lower_bound": [0, 0, 0], "upper_bound": [0.1, 0.1, 0.1],
        "field_bc": ["pec", "pec", "pec"],
    })
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    issues = [i for i in r.issues if i.code == "es.debye_resolution"]
    assert issues and issues[0].severity == Severity.ERROR


def test_es_pic_field_bc_lo_wrong_length():
    """field_bc_lo length != dim → ERROR."""
    # Use dataclass constructor directly to avoid from_dict domain defaults
    spec = ElectrostaticPICSpec(field_bc_lo=["periodic", "pec"])  # dim=1, but len=2
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any("field_bc_lo.len" in i.code for i in r.issues)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

def test_es_pic_generate_basic():
    """Basic two-species ES-PIC generate: key ParmParse lines present."""
    spec = _spec()
    text = generate_inputs_electrostatic_pic(spec)
    assert "warpx.do_electrostatic = labframe" in text
    assert "warpx.const_dt" in text
    assert "electrons.charge = -q_e" in text
    assert "algo.maxwell_solver" not in text


def test_es_pic_generate_with_collision():
    """Generator emits Coulomb collision block."""
    collisions = [{"name": "coul_ei", "type": "coulomb",
                   "species": ["electrons", "ions"]}]
    spec = _spec(collisions=collisions)
    text = generate_inputs_electrostatic_pic(spec)
    assert "collisions.collision_names = coul_ei" in text
    assert "coul_ei.type = coulomb" in text


def test_es_pic_generate_fft_solver():
    """Generator emits correct poisson_solver line for FFT."""
    spec = _spec(poisson_solver="fft", field_bc=["periodic"])
    text = generate_inputs_electrostatic_pic(spec)
    assert "warpx.poisson_solver = fft" in text
    # Multigrid precision lines should not appear for FFT
    assert "self_fields_required_precision" not in text


def test_es_pic_generate_with_eb():
    """EB section is emitted when eb_implicit_function is set."""
    spec = _spec(eb_implicit_function="x^2 + y^2 - 0.01^2")
    r = validate_electrostatic_pic_spec(spec)
    assert r.ok, r.issues
    text = generate_inputs_electrostatic_pic(spec)
    assert "warpx.eb_implicit_function" in text
    assert "x^2 + y^2" in text


def test_es_pic_generate_asymmetric_bc():
    """Asymmetric field_bc_lo/hi produce distinct boundary lines."""
    spec = ElectrostaticPICSpec.from_dict({
        "species": _TWO_SPECIES,
        "const_dt": 1e-11,
        "dim": 2,
        "number_of_cells": [100, 100],
        "lower_bound": [0.0, 0.0],
        "upper_bound": [0.01, 0.01],
        "field_bc": ["periodic", "periodic"],
        "field_bc_lo": ["pec", "periodic"],
        "field_bc_hi": ["periodic", "pec"],
    })
    text = generate_inputs_electrostatic_pic(spec)
    lo_line = [l for l in text.splitlines() if "boundary.field_lo" in l][0]
    hi_line = [l for l in text.splitlines() if "boundary.field_hi" in l][0]
    assert lo_line != hi_line
    assert "pec periodic" in lo_line
    assert "periodic pec" in hi_line


# ---------------------------------------------------------------------------
# FFT/periodic BC tests
# ---------------------------------------------------------------------------

def test_es_pic_fft_requires_periodic_error():
    """poisson_solver='fft' with non-periodic BC → ERROR."""
    spec = _spec(
        dim=1, number_of_cells=[200], lower_bound=[0.0], upper_bound=[1e-2],
        field_bc=["pec"], poisson_solver="fft",
    )
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "es.fft_requires_periodic" for i in r.issues)


def test_es_pic_fft_with_periodic_ok():
    """poisson_solver='fft' with all-periodic BC → no FFT error."""
    spec = _spec(poisson_solver="fft", field_bc=["periodic"])
    r = validate_electrostatic_pic_spec(spec)
    assert not any(i.code == "es.fft_requires_periodic" for i in r.issues)


# ---------------------------------------------------------------------------
# AMR tests
# ---------------------------------------------------------------------------

def test_es_pic_amr_plasma_ok():
    """max_level=1 with tag_by='plasma' → no AMR errors (default n_cell=[200] div by 8)."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma")
    r = validate_electrostatic_pic_spec(spec)
    amr_errors = [i for i in r.issues if i.code.startswith("amr.") and i.severity == Severity.ERROR]
    assert not amr_errors, amr_errors


def test_es_pic_amr_bad_blocking():
    """n_cell=[100, 512] (100 not div by 8) with max_level=1 → ERROR."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma",
                 number_of_cells=[100, 512])
    r = validate_electrostatic_pic_spec(spec)
    assert any("amr.blocking_factor.divisibility" in i.code for i in r.issues)


def test_es_pic_dx_target():
    """dx_target in from_dict auto-computes number_of_cells rounded to blocking_factor."""
    spec = _spec(lower_bound=[0.0, 0.0], upper_bound=[1e-3, 2e-3],
                 dx_target=5e-6, amr_blocking_factor=8)
    assert len(spec.domain.number_of_cells) == 2
    for n in spec.domain.number_of_cells:
        assert n % 8 == 0
        assert n > 0


# ---------------------------------------------------------------------------
# PICMI generator smoke tests
# ---------------------------------------------------------------------------

def test_es_pic_picmi_generates_script():
    from pywarpx.inputgen.electrostatic_pic import generate_picmi_electrostatic_pic
    spec = _spec()
    script = generate_picmi_electrostatic_pic(spec)
    assert isinstance(script, str) and script
    assert "from pywarpx import picmi" in script


def test_es_pic_picmi_correct_solver():
    from pywarpx.inputgen.electrostatic_pic import generate_picmi_electrostatic_pic
    spec = _spec()
    script = generate_picmi_electrostatic_pic(spec)
    assert "ElectrostaticSolver" in script


def test_es_pic_picmi_has_diagnostics():
    from pywarpx.inputgen.electrostatic_pic import generate_picmi_electrostatic_pic
    spec = _spec()
    script = generate_picmi_electrostatic_pic(spec)
    assert "sim.add_diagnostic" in script
    assert "sim.initialize_inputs" in script


def test_es_pic_all_neumann_3d_singular():
    """All-neumann 3D multigrid is singular → ERROR."""
    spec = ElectrostaticPICSpec.from_dict({
        "dim": 3,
        "number_of_cells": [32, 32, 32],
        "lower_bound": [0, 0, 0],
        "upper_bound": [0.001, 0.001, 0.001],
        "field_bc": ["neumann", "neumann", "neumann"],
        "const_dt": 1e-12,
        "species": _TWO_SPECIES,
    })
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "es.multigrid.singular" for i in r.issues)


def test_es_pic_pec_z_not_singular():
    """periodic/pec 3D multigrid is non-singular → no singularity error."""
    spec = ElectrostaticPICSpec.from_dict({
        "dim": 3,
        "number_of_cells": [16, 16, 64],
        "lower_bound": [0, 0, 0],
        "upper_bound": [0.001, 0.001, 0.004],
        "field_bc": ["periodic", "periodic", "pec"],
        "const_dt": 5e-12,
        "species": _TWO_SPECIES,
    })
    r = validate_electrostatic_pic_spec(spec)
    assert all(i.code != "es.multigrid.singular" for i in r.issues)


def test_es_pic_neumann_periodic_mix_singular():
    """neumann + periodic 2D multigrid is also singular → ERROR."""
    spec = ElectrostaticPICSpec.from_dict({
        "dim": 2,
        "number_of_cells": [32, 64],
        "lower_bound": [0, 0],
        "upper_bound": [0.001, 0.004],
        "field_bc": ["periodic", "neumann"],
        "const_dt": 5e-12,
        "species": _TWO_SPECIES,
    })
    r = validate_electrostatic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "es.multigrid.singular" for i in r.issues)


# ---------------------------------------------------------------------------
# boundary_potential: PEC-without-potential NOTE and degenerate WARNING
# ---------------------------------------------------------------------------

_3D_PEC_BASE = {
    "dim": 3,
    "number_of_cells": [64, 64, 64],
    "lower_bound": [0.0, 0.0, 0.0],
    "upper_bound": [0.1, 0.1, 0.1],
    "field_bc": ["pec", "pec", "pec"],
    "const_dt": 2e-9,
    "species": _TWO_SPECIES,   # quasi-neutral: equal densities, charges ±1
}


def test_pec_without_potential_note():
    """PEC wall without boundary_potential_lo/hi emits NOTE."""
    spec = ElectrostaticPICSpec.from_dict(_3D_PEC_BASE)
    r = validate_electrostatic_pic_spec(spec)
    notes = [i for i in r.issues if i.code == "es.potential.unspecified"]
    assert notes, "expected NOTE for PEC walls without explicit potential"


def test_pec_with_explicit_potential_no_note():
    """PEC walls with all potentials explicitly set → no unspecified NOTE."""
    d = dict(_3D_PEC_BASE)
    d["boundary_potential_lo"] = [1.0, 1.0, 1.0]
    d["boundary_potential_hi"] = [1.0, 1.0, 1.0]
    spec = ElectrostaticPICSpec.from_dict(d)
    r = validate_electrostatic_pic_spec(spec)
    notes = [i for i in r.issues if i.code == "es.potential.unspecified"]
    assert not notes, f"unexpected unspecified NOTE: {notes}"


def test_degenerate_all_equal_potential_warns():
    """All PEC walls at same potential + quasi-neutral plasma → WARNING."""
    d = dict(_3D_PEC_BASE)
    d["boundary_potential_lo"] = [1.0, 1.0, 1.0]
    d["boundary_potential_hi"] = [1.0, 1.0, 1.0]
    spec = ElectrostaticPICSpec.from_dict(d)
    r = validate_electrostatic_pic_spec(spec)
    warnings = [i for i in r.issues if i.code == "es.potential.degenerate"]
    assert warnings, "expected WARNING for all-equal-potential quasi-neutral case"


def test_degenerate_all_zero_potential_warns():
    """Unspecified PEC potentials (all default 0 V) + quasi-neutral → WARNING."""
    # No boundary_potential keys at all — validator should still detect the
    # degenerate case since WarpX defaults unspecified PEC walls to 0 V.
    spec = ElectrostaticPICSpec.from_dict(_3D_PEC_BASE)
    r = validate_electrostatic_pic_spec(spec)
    warnings = [i for i in r.issues if i.code == "es.potential.degenerate"]
    assert warnings, "expected WARNING when all PEC walls implicitly at 0 V"


def test_asymmetric_potential_no_degenerate_warning():
    """Asymmetric potentials (different lo/hi) → no degenerate WARNING."""
    d = dict(_3D_PEC_BASE)
    d["field_bc"] = ["periodic", "periodic", "pec"]
    d["boundary_potential_lo"] = [None, None, 0.0]
    d["boundary_potential_hi"] = [None, None, -30.0]
    spec = ElectrostaticPICSpec.from_dict(d)
    r = validate_electrostatic_pic_spec(spec)
    warnings = [i for i in r.issues if i.code == "es.potential.degenerate"]
    assert not warnings, f"unexpected degenerate WARNING: {warnings}"


def test_non_neutral_plasma_no_degenerate_warning():
    """All-equal PEC potentials but strongly non-neutral plasma → no WARNING."""
    d = dict(_3D_PEC_BASE)
    d["boundary_potential_lo"] = [0.0, 0.0, 0.0]
    d["boundary_potential_hi"] = [0.0, 0.0, 0.0]
    # Make plasma non-neutral: 10× more electrons than ions
    d["species"] = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4,
         "density": 1e17, "ppc": 16, "temperature_eV": 1.0},
        {"name": "ions", "charge": 1, "mass_amu": 1.00728, "density": 1e16, "ppc": 16},
    ]
    spec = ElectrostaticPICSpec.from_dict(d)
    r = validate_electrostatic_pic_spec(spec)
    warnings = [i for i in r.issues if i.code == "es.potential.degenerate"]
    assert not warnings, f"unexpected degenerate WARNING for non-neutral plasma: {warnings}"


# ---------------------------------------------------------------------------
# boundary_potential in native inputs generation
# ---------------------------------------------------------------------------

def test_generate_emits_boundary_potential():
    """Native generator emits boundary.potential_lo/hi lines when potentials set."""
    d = dict(_3D_PEC_BASE)
    d["boundary_potential_lo"] = [0.0, 0.0, 0.0]
    d["boundary_potential_hi"] = [0.0, 0.0, -30.0]
    spec = ElectrostaticPICSpec.from_dict(d)
    text = generate_inputs_electrostatic_pic(spec)
    assert "boundary.potential_lo_x = 0" in text
    assert "boundary.potential_hi_z = -30" in text


def test_generate_emits_abs_tol():
    """Native generator always emits self_fields_absolute_tolerance for multigrid."""
    d = dict(_3D_PEC_BASE)
    d["boundary_potential_lo"] = [1.0, 1.0, 1.0]
    d["boundary_potential_hi"] = [1.0, 1.0, 1.0]
    spec = ElectrostaticPICSpec.from_dict(d)
    text = generate_inputs_electrostatic_pic(spec)
    assert "warpx.self_fields_absolute_tolerance" in text


def test_generate_abs_tol_scales_with_potential():
    """abs_tol should be larger when wall potential is larger."""
    base = dict(_3D_PEC_BASE)

    def _abs_tol(v_scale: float) -> float:
        d = dict(base)
        d["boundary_potential_lo"] = [v_scale] * 3
        d["boundary_potential_hi"] = [v_scale] * 3
        text = generate_inputs_electrostatic_pic(ElectrostaticPICSpec.from_dict(d))
        for line in text.splitlines():
            if "self_fields_absolute_tolerance" in line:
                return float(line.split("=")[1].strip())
        raise AssertionError("abs_tol line not found")

    assert _abs_tol(100.0) > _abs_tol(1.0), "abs_tol should scale with potential magnitude"


def test_generate_no_potential_no_potential_lines():
    """Native generator omits potential lines when boundary_potential not set."""
    spec = ElectrostaticPICSpec.from_dict(_3D_PEC_BASE)
    text = generate_inputs_electrostatic_pic(spec)
    assert "boundary.potential_lo" not in text
    assert "boundary.potential_hi" not in text


def test_generate_partial_potential():
    """None entries in boundary_potential are silently skipped."""
    d = dict(_3D_PEC_BASE)
    d["field_bc"] = ["periodic", "periodic", "pec"]
    d["boundary_potential_lo"] = [None, None, 0.0]
    d["boundary_potential_hi"] = [None, None, -20.0]
    spec = ElectrostaticPICSpec.from_dict(d)
    text = generate_inputs_electrostatic_pic(spec)
    assert "boundary.potential_lo_z = 0" in text
    assert "boundary.potential_hi_z = -20" in text
    assert "boundary.potential_lo_x" not in text
    assert "boundary.potential_hi_y" not in text
