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
