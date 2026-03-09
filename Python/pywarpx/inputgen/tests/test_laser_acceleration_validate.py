from pywarpx.inputgen.laser_acceleration import LaserAccelerationSpec
from pywarpx.inputgen.laser_acceleration_validate import validate_laser_acceleration_spec
from pywarpx.inputgen.spec import Severity


def _spec(**overrides) -> LaserAccelerationSpec:
    """Build a spec from flat-dict overrides (mirrors CLI JSON format)."""
    return LaserAccelerationSpec.from_dict(overrides)


def test_laser_validate_ok():
    spec = LaserAccelerationSpec()
    r = validate_laser_acceleration_spec(spec)
    assert r.ok


def test_laser_validate_bad_dim():
    spec = _spec(dim=4)
    r = validate_laser_acceleration_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and i.code == "domain.dim" for i in r.issues)


def test_laser_validate_3d_ok():
    # Provide explicit 3-element lists so __post_init__ expansion is not needed
    spec = _spec(
        dim=3,
        number_of_cells=[64, 128, 512],
        lower_bound=[-20e-6, -20e-6, 0.0],
        upper_bound=[20e-6, 20e-6, 200e-6],
        field_bc=["periodic", "periodic", "open"],
    )
    r = validate_laser_acceleration_spec(spec)
    assert r.ok, r.issues


def test_laser_validate_3d_auto_expand():
    # Setting only dim=3 should trigger __post_init__ default expansion
    spec = _spec(dim=3)
    r = validate_laser_acceleration_spec(spec)
    assert r.ok, r.issues
    assert spec.domain.dim == 3
    assert len(spec.domain.number_of_cells) == 3


def test_laser_plasma_slab_invalid():
    spec = _spec(plasma_zmin=1e-4, plasma_zmax=1e-5)
    r = validate_laser_acceleration_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and i.code == "species.slab" for i in r.issues)


def test_laser_implicit_validate_ok():
    """Implicit solver with valid parameters passes validation."""
    spec = _spec(
        implicit_enabled=True,
        implicit_theta=0.5,
        implicit_solver_type="picard",
        implicit_max_iters=30,
        implicit_tolerance=1e-3,
        implicit_const_dt=1e-13,
    )
    r = validate_laser_acceleration_spec(spec)
    assert r.ok, r.issues


def test_laser_implicit_validate_missing_const_dt():
    """Implicit enabled but const_dt=0 → error."""
    spec = _spec(implicit_enabled=True, implicit_const_dt=0.0)
    r = validate_laser_acceleration_spec(spec)
    assert not r.ok
    assert any(i.code == "implicit.const_dt" for i in r.issues)


def test_laser_implicit_validate_bad_solver_type():
    """Invalid solver_type → error."""
    spec = _spec(implicit_enabled=True, implicit_const_dt=1e-13,
                 implicit_solver_type="gmres")
    r = validate_laser_acceleration_spec(spec)
    assert not r.ok
    assert any(i.code == "implicit.solver_type" for i in r.issues)


def test_laser_implicit_generate_inputs():
    """Native generator emits implicit solver block when enabled."""
    from pywarpx.inputgen.laser_acceleration_native import generate_inputs_laser_acceleration
    spec = _spec(implicit_enabled=True, implicit_const_dt=1e-13, implicit_theta=0.5)
    text = generate_inputs_laser_acceleration(spec)
    assert "algo.evolve_scheme = theta_implicit_em" in text
    assert "warpx.const_dt" in text
    assert "warpx.cfl" not in text


def test_laser_explicit_generate_inputs():
    """Native generator emits CFL line (not implicit) by default."""
    from pywarpx.inputgen.laser_acceleration_native import generate_inputs_laser_acceleration
    spec = LaserAccelerationSpec()
    text = generate_inputs_laser_acceleration(spec)
    assert "warpx.cfl" in text
    assert "algo.evolve_scheme" not in text
