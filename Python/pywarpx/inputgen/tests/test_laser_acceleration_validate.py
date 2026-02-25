from pywarpx.inputgen.laser_acceleration import LaserAccelerationSpec
from pywarpx.inputgen.laser_acceleration_validate import validate_laser_acceleration_spec
from pywarpx.inputgen.spec import Severity


def test_laser_validate_ok():
    spec = LaserAccelerationSpec()
    r = validate_laser_acceleration_spec(spec)
    assert r.ok


def test_laser_validate_bad_dim():
    spec = LaserAccelerationSpec(dim=3)
    r = validate_laser_acceleration_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and i.code == "laser.dim" for i in r.issues)


def test_laser_plasma_slab_invalid():
    spec = LaserAccelerationSpec(plasma_zmin=1e-4, plasma_zmax=1e-5)
    r = validate_laser_acceleration_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and i.code == "laser.plasma_slab" for i in r.issues)
