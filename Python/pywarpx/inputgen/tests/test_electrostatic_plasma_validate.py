from pywarpx.inputgen.electrostatic_plasma import ElectrostaticPlasmaSpec
from pywarpx.inputgen.electrostatic_plasma_validate import validate_electrostatic_plasma_spec
from pywarpx.inputgen.spec import Severity


def _spec(**overrides) -> ElectrostaticPlasmaSpec:
    return ElectrostaticPlasmaSpec.from_dict(overrides)


def test_es_validate_ok():
    spec = ElectrostaticPlasmaSpec()
    r = validate_electrostatic_plasma_spec(spec)
    assert r.ok, r.issues


def test_es_validate_from_dict_ok():
    # Explicit clean parameters: dx=50μm < λ_De≈74μm, dt*ω_pe≈0.056
    spec = _spec(
        dim=1, number_of_cells=[100], lower_bound=[0.0], upper_bound=[5e-3],
        field_bc=["periodic"],
        n0=1e16, Te=1.0, Ti=0.0, ppc=16, const_dt=1e-11,
        max_steps=100, diag_period=10,
    )
    r = validate_electrostatic_plasma_spec(spec)
    assert r.ok, r.issues


def test_es_validate_bad_dim():
    spec = _spec(dim=5)
    r = validate_electrostatic_plasma_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and i.code == "domain.dim" for i in r.issues)


def test_es_validate_bad_density():
    spec = _spec(n0=-1.0)
    r = validate_electrostatic_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "es.n0" for i in r.issues)


def test_es_validate_bad_Te():
    spec = _spec(Te=0.0)
    r = validate_electrostatic_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "es.Te" for i in r.issues)


def test_es_validate_bad_solver_type():
    spec = _spec(electrostatic_solver="gyrokinetic")
    r = validate_electrostatic_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "es.solver_type" for i in r.issues)


def test_es_validate_debye_resolution_warns():
    """dx >> λ_De → Debye resolution warning (not error)."""
    # n0=1e16, Te=1 eV: λ_De ≈ 74.3 μm; dx=50mm >> λ_De
    spec = _spec(
        dim=1, number_of_cells=[10], lower_bound=[0.0], upper_bound=[0.5],
        field_bc=["periodic"],
        n0=1e16, Te=1.0, const_dt=1e-12,
    )
    r = validate_electrostatic_plasma_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "es.debye_resolution" for i in r.issues)


def test_es_validate_debye_resolution_ok():
    """dx < λ_De: no Debye warning."""
    # n0=1e16, Te=1 eV: λ_De ≈ 74.3 μm; dx=50μm < λ_De
    spec = _spec(
        dim=1, number_of_cells=[100], lower_bound=[0.0], upper_bound=[5e-3],
        field_bc=["periodic"],
        n0=1e16, Te=1.0, const_dt=1e-11,
    )
    r = validate_electrostatic_plasma_spec(spec)
    assert not any(i.code == "es.debye_resolution" for i in r.issues), r.issues


def test_es_validate_plasma_frequency_unstable():
    """dt * ω_pe >= 2 → ERROR (Boris pusher unstable)."""
    # n0=1e20: ω_pe ≈ 5.638e11 rad/s; dt=2e-10 → dt*ω_pe ≈ 113 >> 2
    # domain: Nc=100, L=5e-4 → dx=5μm < λ_De≈7.4μm (avoids Debye warning)
    spec = _spec(
        dim=1, number_of_cells=[100], lower_bound=[0.0], upper_bound=[5e-4],
        field_bc=["periodic"],
        n0=1e20, Te=100.0, const_dt=2e-10,
    )
    r = validate_electrostatic_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "es.plasma_frequency" for i in r.issues)


def test_es_validate_plasma_frequency_accuracy_warns():
    """0.2 < dt * ω_pe < 2 → accuracy WARNING (not error)."""
    # n0=1e20: ω_pe ≈ 5.638e11 rad/s; dt=9e-13 → dt*ω_pe ≈ 0.507
    # domain: Nc=100, L=1e-4 → dx=1μm < λ_De≈7.4μm (avoids Debye warning)
    spec = _spec(
        dim=1, number_of_cells=[100], lower_bound=[0.0], upper_bound=[1e-4],
        field_bc=["periodic"],
        n0=1e20, Te=100.0, const_dt=9e-13,
    )
    r = validate_electrostatic_plasma_spec(spec)
    assert r.ok  # warning only
    assert any(i.code == "es.plasma_frequency.accuracy" for i in r.issues)


def test_es_generate_inputs():
    """Generator produces a non-empty string containing key WarpX parameters."""
    from pywarpx.inputgen.electrostatic_plasma import generate_inputs_electrostatic_plasma

    spec = ElectrostaticPlasmaSpec()
    text = generate_inputs_electrostatic_plasma(spec)
    assert "warpx.do_electrostatic = labframe" in text
    assert "electrons.charge = -q_e" in text
    assert "ions.charge = q_e" in text
    assert "warpx.const_dt" in text


def test_es_generate_inputs_no_ions():
    """Generator omits ion section when include_ions=False."""
    from pywarpx.inputgen.electrostatic_plasma import generate_inputs_electrostatic_plasma

    spec = _spec(include_ions=False)
    text = generate_inputs_electrostatic_plasma(spec)
    assert "electrons.charge = -q_e" in text
    assert "ions.charge = q_e" not in text


def test_es_validate_eb_ok():
    """EB spec with sphere implicit function passes validation."""
    spec = _spec(eb_implicit_function="-(x**2 + y**2 + z**2 - 0.001**2)")
    r = validate_electrostatic_plasma_spec(spec)
    assert r.ok, r.issues


def test_es_validate_eb_overspecified():
    """Setting both eb_implicit_function and stl_file is an error."""
    spec = _spec(
        eb_implicit_function="-(x**2 - 1e-4)",
        stl_file="sphere.stl",
    )
    r = validate_electrostatic_plasma_spec(spec)
    assert not r.ok
    assert any(i.code == "eb.overspecified" for i in r.issues)


def test_es_generate_inputs_with_eb():
    """Generator emits EB section when eb_implicit_function is set."""
    from pywarpx.inputgen.electrostatic_plasma import generate_inputs_electrostatic_plasma

    spec = _spec(eb_implicit_function="-(x**2 + y**2 - 0.001**2)",
                 eb_potential="100.0")
    text = generate_inputs_electrostatic_plasma(spec)
    assert "warpx.eb_implicit_function" in text
    assert "warpx.eb_potential(x,y,z,t) = 100.0" in text


def test_es_generate_inputs_no_eb():
    """Generator omits EB section when no EB is configured."""
    from pywarpx.inputgen.electrostatic_plasma import generate_inputs_electrostatic_plasma

    spec = ElectrostaticPlasmaSpec()
    text = generate_inputs_electrostatic_plasma(spec)
    assert "warpx.eb_implicit_function" not in text
