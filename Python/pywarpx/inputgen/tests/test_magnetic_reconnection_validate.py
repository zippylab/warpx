from pywarpx.inputgen.magnetic_reconnection import (
    MagneticReconnectionSpec,
    generate_inputs_magnetic_reconnection,
)
from pywarpx.inputgen.magnetic_reconnection_validate import validate_magnetic_reconnection_spec
from pywarpx.inputgen.spec import Severity


def _spec(**overrides) -> MagneticReconnectionSpec:
    return MagneticReconnectionSpec.from_dict(overrides)


def test_reconnect_validate_ok():
    spec = MagneticReconnectionSpec()
    r = validate_magnetic_reconnection_spec(spec)
    assert r.ok, r.issues


def test_reconnect_validate_from_dict_ok():
    spec = _spec(
        dim=2,
        number_of_cells=[512, 256],
        lower_bound=[0.0, -0.025],
        upper_bound=[0.050, 0.025],
        field_bc=["periodic", "periodic"],
        B0=0.1,
        Bg=0.0,
        delta=1.25e-3,
        dB_fraction=0.01,
        const_dt=1e-11,
    )
    r = validate_magnetic_reconnection_spec(spec)
    assert r.ok, r.issues


def test_reconnect_validate_bad_dim():
    spec = _spec(dim=3)
    r = validate_magnetic_reconnection_spec(spec)
    assert not r.ok
    assert any(i.code == "domain.dim" for i in r.issues)


def test_reconnect_validate_bad_B0():
    spec = _spec(B0=-0.1)
    r = validate_magnetic_reconnection_spec(spec)
    assert not r.ok
    assert any(i.code == "reconnect.B0" for i in r.issues)


def test_reconnect_validate_bad_delta():
    spec = _spec(delta=-1e-3)
    r = validate_magnetic_reconnection_spec(spec)
    assert not r.ok
    assert any(i.code == "reconnect.delta" for i in r.issues)


def test_reconnect_validate_delta_unresolved():
    """Warn when current sheet width < cell size."""
    # Very fine delta relative to coarse grid
    spec = _spec(
        dim=1,
        number_of_cells=[10],
        lower_bound=[-0.1],
        upper_bound=[0.1],
        field_bc=["periodic"],
        delta=1e-5,    # << dx = 0.02 m
        B0=0.1,
    )
    r = validate_magnetic_reconnection_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "reconnect.delta_unresolved" for i in r.issues)


def test_reconnect_generate_inputs():
    spec = MagneticReconnectionSpec()
    text = generate_inputs_magnetic_reconnection(spec)
    assert "algo.maxwell_solver = hybrid" in text
    assert "warpx.B_ext_grid_init_style = parse_B_ext_grid_function" in text
    assert "warpx.Bx_external_grid_function(x,y,z)" in text
    assert "my_constants.B0" in text
    assert "my_constants.delta" in text
    assert "ions.charge = q_e" in text


def test_reconnect_generate_guide_field():
    """With Bg > 0, By expression should be non-trivial (force-free)."""
    spec = _spec(B0=0.1, Bg=0.03)
    text = generate_inputs_magnetic_reconnection(spec)
    assert "sqrt(Bg**2" in text


def test_reconnect_generate_no_guide_field():
    """With Bg = 0, By expression is 0.0."""
    spec = _spec(B0=0.1, Bg=0.0)
    text = generate_inputs_magnetic_reconnection(spec)
    assert "warpx.By_external_grid_function(x,y,z) = 0.0" in text
