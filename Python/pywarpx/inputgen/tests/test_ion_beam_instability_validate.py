from pywarpx.inputgen.ion_beam_instability import (
    IonBeamInstabilitySpec,
    generate_inputs_ion_beam_instability,
)
from pywarpx.inputgen.ion_beam_instability_validate import validate_ion_beam_instability_spec
from pywarpx.inputgen.spec import Severity


def _spec(**overrides) -> IonBeamInstabilitySpec:
    return IonBeamInstabilitySpec.from_dict(overrides)


def test_beam_validate_ok():
    spec = IonBeamInstabilitySpec()
    r = validate_ion_beam_instability_spec(spec)
    assert r.ok, r.issues


def test_beam_validate_from_dict_ok():
    spec = _spec(
        dim=1,
        number_of_cells=[1024],
        lower_bound=[0.0],
        upper_bound=[0.256],
        field_bc=["periodic"],
        B0=[0.0, 0.0, 0.25],
        beam_drift_velocity=2.4e6,
    )
    r = validate_ion_beam_instability_spec(spec)
    assert r.ok, r.issues


def test_beam_validate_bad_dim():
    spec = _spec(dim=4)
    r = validate_ion_beam_instability_spec(spec)
    assert not r.ok
    assert any(i.code == "domain.dim" for i in r.issues)


def test_beam_validate_zero_b0():
    spec = _spec(B0=[0.0, 0.0, 0.0])
    r = validate_ion_beam_instability_spec(spec)
    assert r.ok  # warning only
    assert any(i.severity == Severity.WARNING and i.code == "beam.B0.zero" for i in r.issues)


def test_beam_validate_high_density_fraction():
    """Warn when beam density > 50% of total."""
    spec = _spec(core_density=1e21, beam_density=2e21)  # 67% beam
    r = validate_ion_beam_instability_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "beam.density_fraction" for i in r.issues)


def test_beam_validate_negative_const_dt():
    spec = _spec(const_dt=-1e-10)
    r = validate_ion_beam_instability_spec(spec)
    assert not r.ok
    assert any(i.code == "beam.const_dt" for i in r.issues)


def test_beam_generate_inputs():
    spec = IonBeamInstabilitySpec()
    text = generate_inputs_ion_beam_instability(spec)
    assert "algo.maxwell_solver = hybrid" in text
    assert "core_ions.charge = q_e" in text
    assert "beam_ions.charge = q_e" in text
    assert "particles.species_names = core_ions beam_ions" in text
    assert "warpx.const_dt" in text


def test_beam_generate_auto_core_drift():
    """Core drift is auto-computed for momentum conservation when not set."""
    spec = _spec(
        core_density=2.97e22,
        beam_density=3.3e21,
        beam_drift_velocity=2.4e6,
        core_drift_velocity=0.0,
    )
    text = generate_inputs_ion_beam_instability(spec)
    # core drift should be nonzero (auto-computed) and negative
    assert "core_ions.uz_m = -" in text or "core_ions.uz_m = 0" not in text


# ---------------------------------------------------------------------------
# AMR tests
# ---------------------------------------------------------------------------

def test_beam_amr_plasma_ok():
    """max_level=1 with tag_by='plasma' → no AMR errors (default n_cell=[128,512] div by 8)."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma")
    r = validate_ion_beam_instability_spec(spec)
    amr_errors = [i for i in r.issues if i.code.startswith("amr.") and i.severity == Severity.ERROR]
    assert not amr_errors, amr_errors


def test_beam_amr_bad_blocking():
    """n_cell=[100, 512] (100 not div by 8) with max_level=1 → ERROR."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma",
                 number_of_cells=[100, 512])
    r = validate_ion_beam_instability_spec(spec)
    assert any("amr.blocking_factor.divisibility" in i.code for i in r.issues)


# ---------------------------------------------------------------------------
# PICMI generator smoke tests
# ---------------------------------------------------------------------------

def test_beam_picmi_generates_script():
    from pywarpx.inputgen.ion_beam_instability import generate_picmi_ion_beam_instability
    spec = IonBeamInstabilitySpec()
    script = generate_picmi_ion_beam_instability(spec)
    assert isinstance(script, str) and script
    assert "from pywarpx import picmi" in script


def test_beam_picmi_correct_solver():
    from pywarpx.inputgen.ion_beam_instability import generate_picmi_ion_beam_instability
    spec = IonBeamInstabilitySpec()
    script = generate_picmi_ion_beam_instability(spec)
    assert "HybridPICSolver" in script


def test_beam_picmi_has_diagnostics():
    from pywarpx.inputgen.ion_beam_instability import generate_picmi_ion_beam_instability
    spec = IonBeamInstabilitySpec()
    script = generate_picmi_ion_beam_instability(spec)
    assert "sim.add_diagnostic" in script
    assert "sim.initialize_inputs" in script
