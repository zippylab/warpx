from pywarpx.inputgen.hybrid_plasma import HybridPlasmaSpec
from pywarpx.inputgen.hybrid_plasma_validate import validate_hybrid_plasma_spec
from pywarpx.inputgen.spec import Severity


def _spec(**overrides) -> HybridPlasmaSpec:
    return HybridPlasmaSpec.from_dict(overrides)


def test_hybrid_validate_ok():
    spec = HybridPlasmaSpec()
    r = validate_hybrid_plasma_spec(spec)
    assert r.ok, r.issues


def test_hybrid_validate_from_dict_ok():
    import json, pathlib
    import pywarpx.inputgen as _pkg
    example = pathlib.Path(_pkg.__file__).parent / "_example_hybrid_plasma_spec.json"
    spec = HybridPlasmaSpec.from_dict(json.loads(example.read_text()))
    r = validate_hybrid_plasma_spec(spec)
    assert r.ok, r.issues


def test_hybrid_validate_bad_dim():
    spec = _spec(dim=5)
    r = validate_hybrid_plasma_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and i.code == "domain.dim" for i in r.issues)


def test_hybrid_validate_bad_substeps():
    spec = _spec(substeps=3)  # odd — must be divisible by 2
    r = validate_hybrid_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "ohm.substeps" for i in r.issues)


def test_hybrid_validate_zero_B0_warns():
    spec = _spec(B0=[0.0, 0.0, 0.0])
    r = validate_hybrid_plasma_spec(spec)
    assert r.ok  # warning, not error
    assert any(i.severity == Severity.WARNING and i.code == "hybrid.B0.zero" for i in r.issues)


def test_hybrid_validate_density_mismatch_warns():
    spec = _spec(ion_density=1.0e20, n0_ref=1.0e25)  # factor 1e5 off
    r = validate_hybrid_plasma_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "hybrid.density_mismatch" for i in r.issues)
