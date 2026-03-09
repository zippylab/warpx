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


def test_hybrid_validate_whistler_cfl_warns():
    """Old-style parameters (n=1e20, dx=1e-5 m) have l_i >> dx: whistler unstable."""
    spec = _spec(
        dim=1, number_of_cells=[512], lower_bound=[0.0], upper_bound=[0.00512],
        field_bc=["periodic"],
        n0_ref=1.0e20, substeps=400, const_dt=2e-9,
        B0=[0.0, 0.0, 0.25], ion_density=1.0e20, max_steps=20,
    )
    r = validate_hybrid_plasma_spec(spec)
    assert r.ok  # warning, not error
    assert any(i.code == "hybrid.cfl.whistler" for i in r.issues)


def test_hybrid_validate_whistler_cfl_ok():
    """Self-consistent parameters (n=3.3e22, dx≈0.1*l_i): whistler CFL stable."""
    spec = _spec(
        dim=1, number_of_cells=[512], lower_bound=[0.0], upper_bound=[0.064],
        field_bc=["periodic"],
        n0_ref=3.3e22, substeps=40, const_dt=1.3e-9,
        B0=[0.0, 0.0, 0.25], ion_density=3.3e22, Te=0.05,
        ion_temperature_eV=0.05, max_steps=200,
    )
    r = validate_hybrid_plasma_spec(spec)
    assert not any(i.code == "hybrid.cfl.whistler" for i in r.issues)


# ---------------------------------------------------------------------------
# AMR tests
# ---------------------------------------------------------------------------

def test_hybrid_amr_plasma_ok():
    """max_level=1 with tag_by='plasma' → no AMR errors (default n_cell=[1000] div by 8)."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma")
    r = validate_hybrid_plasma_spec(spec)
    amr_errors = [i for i in r.issues if i.code.startswith("amr.") and i.severity == Severity.ERROR]
    assert not amr_errors, amr_errors


def test_hybrid_amr_bad_blocking():
    """n_cell=[100, 512] (100 not div by 8) with max_level=1 → ERROR."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma",
                 number_of_cells=[100, 512])
    r = validate_hybrid_plasma_spec(spec)
    assert any("amr.blocking_factor.divisibility" in i.code for i in r.issues)
