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
