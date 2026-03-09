from pywarpx.inputgen.pwfa import PWFASpec, generate_inputs_pwfa
from pywarpx.inputgen.pwfa_validate import validate_pwfa_spec
from pywarpx.inputgen.blocks import ParticleBeamSpec
from pywarpx.inputgen.spec import Severity


def _spec(**overrides) -> PWFASpec:
    return PWFASpec.from_dict(overrides)


def test_pwfa_validate_ok():
    spec = PWFASpec()
    r = validate_pwfa_spec(spec)
    assert r.ok, r.issues


def test_pwfa_validate_from_dict_ok():
    spec = _spec(
        dim=2,
        number_of_cells=[128, 512],
        lower_bound=[-150e-6, -200e-6],
        upper_bound=[150e-6, 0.0],
        field_bc=["open", "open"],
        plasma_density=1e22,
        plasma_zmin=-200e-6,
        plasma_zmax=0.0,
        driver_uz_m=2000.0,
        driver_q_tot=-1e-9,
    )
    r = validate_pwfa_spec(spec)
    assert r.ok, r.issues


def test_pwfa_validate_bad_dim():
    spec = _spec(dim=1)
    r = validate_pwfa_spec(spec)
    assert not r.ok
    assert any(i.code == "domain.dim" for i in r.issues)


def test_pwfa_validate_nonrelativistic_driver_warns():
    """Warn when driver uz_m < 10."""
    spec = _spec(driver_uz_m=5.0)
    r = validate_pwfa_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "pwfa.driver.uz_m" for i in r.issues)


def test_pwfa_validate_skin_depth_resolution_warns():
    """Warn when grid cannot resolve plasma skin depth."""
    # n=1e23: kp_inv ≈ 53 μm; dx = 300μm/10 = 30μm; ratio = 0.57 > 0.5
    spec = _spec(
        dim=2,
        number_of_cells=[10, 10],
        lower_bound=[-150e-6, -300e-6],
        upper_bound=[150e-6, 0.0],
        field_bc=["open", "open"],
        plasma_density=1e23,
    )
    r = validate_pwfa_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "pwfa.skin_depth" for i in r.issues)


def test_pwfa_generate_inputs():
    spec = PWFASpec()
    text = generate_inputs_pwfa(spec)
    assert "algo.maxwell_solver = yee" in text
    assert "driver_beam.injection_style = gaussian_beam" in text
    assert "electrons.charge = -q_e" in text
    assert "protons.charge = q_e" in text
    assert "warpx.do_moving_window = 1" in text


def test_pwfa_generate_with_witness():
    """Witness beam section is only emitted when specified."""
    spec = PWFASpec()
    spec.witness = ParticleBeamSpec(
        x_rms=0.5e-6, y_rms=0.5e-6, z_rms=1e-6,
        uz_m=500.0, uz_th=5.0,
        q_tot=-1e-10, z_mean=-100e-6, n_macro=500,
    )
    text = generate_inputs_pwfa(spec)
    assert "witness_beam.injection_style = gaussian_beam" in text
    assert "particles.species_names = electrons protons driver_beam witness_beam" in text


def test_pwfa_generate_no_moving_window():
    spec = _spec(moving_window=False)
    text = generate_inputs_pwfa(spec)
    assert "warpx.do_moving_window" not in text


def test_pwfa_generate_from_dict_witness():
    """from_dict recognises witness_* prefix keys."""
    spec = _spec(
        witness_uz_m=500.0,
        witness_q_tot=-5e-10,
        witness_x_rms=0.5e-6,
        witness_y_rms=0.5e-6,
        witness_z_rms=1e-6,
        witness_z_mean=-100e-6,
    )
    assert spec.witness is not None
    assert spec.witness.uz_m == 500.0
    text = generate_inputs_pwfa(spec)
    assert "witness_beam.injection_style = gaussian_beam" in text


# ---------------------------------------------------------------------------
# AMR tests
# ---------------------------------------------------------------------------

def test_pwfa_amr_plasma_ok():
    """max_level=1 with tag_by='plasma' → no AMR errors (default n_cell=[128,512] div by 8)."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma")
    r = validate_pwfa_spec(spec)
    amr_errors = [i for i in r.issues if i.code.startswith("amr.") and i.severity == Severity.ERROR]
    assert not amr_errors, amr_errors


def test_pwfa_amr_bad_blocking():
    """n_cell=[100, 512] not divisible by blocking_factor=8 with max_level=1 → ERROR."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma",
                 number_of_cells=[100, 512])
    r = validate_pwfa_spec(spec)
    assert any("amr.blocking_factor.divisibility" in i.code for i in r.issues)
