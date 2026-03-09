import pytest

from pywarpx.inputgen.spec import UniformPlasmaSpec, Severity
from pywarpx.inputgen.validate import validate_picmi_syntax, validate_uniform_plasma_spec
from pywarpx.inputgen.generate import generate_picmi_uniform_plasma


def _base_spec(**overrides):
    data = dict(
        dim=2,
        number_of_cells=[16, 32],
        lower_bound=[0.0, 0.0],
        upper_bound=[1.0, 2.0],
        field_bc=["periodic", "periodic"],
        max_steps=10,
        time_step_size=None,
        cfl=0.99,
        density=1e20,
        temperature_eV=1.0,
        particle_shape=1,
        current_deposition_algo="direct",
        diag_period=10,
        diag_fields=["E", "B", "J"],
        warpx_max_grid_size=16,
        name="uniform_plasma",
    )
    data.update(overrides)
    return UniformPlasmaSpec(**data)


def test_validate_ok():
    spec = _base_spec()
    r = validate_uniform_plasma_spec(spec)
    assert r.ok


@pytest.mark.parametrize("dim", [0, 4])
def test_validate_bad_dim(dim):
    spec = _base_spec(dim=dim)
    r = validate_uniform_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "spec.dim.invalid" and i.severity == Severity.ERROR for i in r.issues)


def test_validate_bad_lengths():
    spec = _base_spec(number_of_cells=[16])
    r = validate_uniform_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "spec.number_of_cells.len" for i in r.issues)


def test_validate_negative_density():
    spec = _base_spec(density=-1.0)
    r = validate_uniform_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "spec.density.nonpositive" for i in r.issues)


def test_debye_underresolved_warns():
    # Very cold/high density -> tiny Debye length; coarse grid should warn.
    spec = _base_spec(temperature_eV=0.1, density=1e25, upper_bound=[1.0, 1.0], number_of_cells=[8, 8])
    r = validate_uniform_plasma_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "physics.debye.underresolved" and i.severity == Severity.WARNING for i in r.issues)


def test_generate_script_parses():
    spec = _base_spec()
    script = generate_picmi_uniform_plasma(spec)
    r, _ = validate_picmi_syntax(script)
    assert r.ok


# ---------------------------------------------------------------------------
# AMR tests
# ---------------------------------------------------------------------------

def test_uniform_plasma_amr_ok():
    """amr_max_level=1 with n_cell=[16,32] divisible by amr_blocking_factor=16 → no AMR errors."""
    spec = _base_spec(amr_max_level=1, amr_blocking_factor=16)
    r = validate_uniform_plasma_spec(spec)
    amr_errors = [i for i in r.issues if i.code.startswith("amr.") and i.severity == Severity.ERROR]
    assert not amr_errors, amr_errors


def test_uniform_plasma_amr_bad_blocking():
    """n_cell=[15, 32] not divisible by amr_blocking_factor=8 with max_level=1 → ERROR."""
    spec = _base_spec(amr_max_level=1, amr_blocking_factor=8, number_of_cells=[15, 32])
    r = validate_uniform_plasma_spec(spec)
    assert any("amr.blocking_factor" in i.code for i in r.issues)
