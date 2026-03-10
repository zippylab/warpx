"""Tests for ElectromagneticPICSpec validation and input generation."""

from pywarpx.inputgen.electromagnetic_pic import (
    ElectromagneticPICSpec,
    generate_inputs_electromagnetic_pic,
)
from pywarpx.inputgen.electromagnetic_pic_validate import validate_electromagnetic_pic_spec
from pywarpx.inputgen.spec import Severity


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TWO_SPECIES = [
    {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e24, "ppc": 4},
    {"name": "protons", "charge": 1, "mass_amu": 1.00728, "density": 1e24, "ppc": 4},
]


def _spec(**overrides) -> ElectromagneticPICSpec:
    d = {"species": _TWO_SPECIES}
    d.update(overrides)
    return ElectromagneticPICSpec.from_dict(d)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_em_pic_validate_ok():
    """Default two-species EM-PIC spec passes validation."""
    spec = _spec()
    r = validate_electromagnetic_pic_spec(spec)
    assert r.ok, r.issues


def test_em_pic_bad_dim():
    """dim=5 is not supported → ERROR."""
    spec = _spec(dim=5, number_of_cells=[128, 128, 128, 128, 128],
                 lower_bound=[0]*5, upper_bound=[1]*5, field_bc=["periodic"]*5)
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and i.code == "domain.dim" for i in r.issues)


def test_em_pic_bad_maxwell_solver():
    """Invalid maxwell_solver → ERROR."""
    spec = _spec(maxwell_solver="fdtd")
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "em.maxwell_solver" for i in r.issues)


def test_em_pic_bad_particle_pusher():
    """Invalid particle_pusher → ERROR."""
    spec = _spec(particle_pusher="leapfrog")
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "em.particle_pusher" for i in r.issues)


def test_em_pic_higuera_explicit_warning():
    """higuera pusher with explicit FDTD → WARNING (not ERROR)."""
    spec = _spec(particle_pusher="higuera")
    r = validate_electromagnetic_pic_spec(spec)
    assert any(i.severity == Severity.WARNING and "higuera" in i.code for i in r.issues)
    # Should still be overall OK (warning not error)
    errors = [i for i in r.issues if i.severity == Severity.ERROR]
    assert not errors


def test_em_pic_ionization_product_missing():
    """Ionization product species not in species list → ERROR."""
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e20, "ppc": 4},
        {"name": "nitrogen", "charge": 7, "mass_amu": 14.003, "density": 1e20, "ppc": 4,
         "do_field_ionization": True, "physical_element": "N",
         "ionization_product_species": "ionized_electrons"},  # not in list
    ]
    spec = ElectromagneticPICSpec.from_dict({"species": species})
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any(i.code == "species.ionization.product" for i in r.issues)


def test_em_pic_qed_bw_product_missing():
    """QED Breit-Wheeler product species not in species list → ERROR."""
    species = [
        {"name": "photons", "charge": 0, "mass_amu": 1e-30,
         "do_qed_breit_wheeler": True,
         "qed_bw_ele_product": "electrons",   # not in list
         "qed_bw_pos_product": "positrons"},  # not in list
    ]
    spec = ElectromagneticPICSpec.from_dict({"species": species})
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any("qed_bw" in i.code for i in r.issues)


def test_em_pic_collision_unknown_species():
    """Collision references a species not in the species list → ERROR."""
    spec = _spec(collisions=[{
        "name": "coul_ei",
        "type": "coulomb",
        "species": ["electrons", "neutrons"],  # neutrons not in list
    }])
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any("collision.species.unknown" in i.code for i in r.issues)


def test_em_pic_nuclear_fusion_multiplier_warning():
    """Nuclear fusion event_multiplier=1.0 → WARNING."""
    species = [
        {"name": "deuterium", "charge": 1, "mass_amu": 2.014, "density": 1e26, "ppc": 4},
        {"name": "tritium", "charge": 1, "mass_amu": 3.016, "density": 1e26, "ppc": 4},
        {"name": "helium4", "charge": 2, "mass_amu": 4.003, "injection_style": "none"},
        {"name": "neutron", "charge": 0, "mass_amu": 1.009, "injection_style": "none"},
    ]
    collisions = [{
        "name": "dt_fusion", "type": "nuclearfusion",
        "species": ["deuterium", "tritium"],
        "product_species": ["helium4", "neutron"],
        "event_multiplier": 1.0,
    }]
    spec = ElectromagneticPICSpec.from_dict({"species": species, "collisions": collisions})
    r = validate_electromagnetic_pic_spec(spec)
    assert any(
        i.severity == Severity.WARNING and "multiplier" in i.code
        for i in r.issues
    )


def test_em_pic_cfl_error():
    """cfl > 1.0 with Yee solver → ERROR (von Neumann instability)."""
    spec = _spec(cfl=1.5)
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and "cfl" in i.code for i in r.issues)


def test_em_pic_psatd_implicit_incompatible():
    """PSATD + implicit → ERROR."""
    spec = _spec(maxwell_solver="psatd",
                 implicit_enabled=True, implicit_const_dt=1e-13)
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any("psatd_implicit" in i.code for i in r.issues)


def test_em_pic_field_bc_lo_wrong_length():
    """field_bc_lo with wrong length for dim → ERROR."""
    spec = _spec(field_bc_lo=["periodic"])  # dim=2, so needs 2 entries
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    assert any("field_bc_lo.len" in i.code for i in r.issues)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

def test_em_pic_generate_basic():
    """Basic two-species EM-PIC generate: key ParmParse lines present."""
    spec = _spec()
    text = generate_inputs_electromagnetic_pic(spec)
    assert "algo.maxwell_solver = yee" in text
    assert "particles.species_names = electrons protons" in text
    assert "electrons.charge = -q_e" in text
    assert "protons.charge = q_e" in text
    assert "warpx.do_electrostatic" not in text


def test_em_pic_generate_ionization():
    """Generator emits field ionization block correctly."""
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e20, "ppc": 4},
        {"name": "nitrogen", "charge": 7, "mass_amu": 14.003, "density": 1e20, "ppc": 4,
         "do_field_ionization": True, "physical_element": "N",
         "ionization_initial_level": 0, "ionization_product_species": "electrons"},
    ]
    spec = ElectromagneticPICSpec.from_dict({"species": species})
    text = generate_inputs_electromagnetic_pic(spec)
    assert "nitrogen.do_field_ionization = 1" in text
    assert "nitrogen.physical_element = N" in text
    assert "nitrogen.ionization_product_species = electrons" in text


def test_em_pic_generate_coulomb_collision():
    """Generator emits Coulomb collision block."""
    collisions = [{"name": "coul_ei", "type": "coulomb",
                   "species": ["electrons", "protons"], "CoulombLog": 10.0}]
    spec = _spec(collisions=collisions)
    text = generate_inputs_electromagnetic_pic(spec)
    assert "collisions.collision_names = coul_ei" in text
    assert "coul_ei.type = coulomb" in text
    assert "coul_ei.CoulombLog = 10" in text


def test_em_pic_generate_nuclear_fusion():
    """Generator emits nuclear fusion block with event_multiplier."""
    species = [
        {"name": "deuterium", "charge": 1, "mass_amu": 2.014, "density": 1e26, "ppc": 4},
        {"name": "tritium", "charge": 1, "mass_amu": 3.016, "density": 1e26, "ppc": 4},
        {"name": "helium4", "charge": 2, "mass_amu": 4.003, "injection_style": "none"},
        {"name": "neutron", "charge": 0, "mass_amu": 1.009, "injection_style": "none"},
    ]
    collisions = [{
        "name": "dt_fusion", "type": "nuclearfusion",
        "species": ["deuterium", "tritium"],
        "product_species": ["helium4", "neutron"],
        "event_multiplier": 1e13,
    }]
    spec = ElectromagneticPICSpec.from_dict({"species": species, "collisions": collisions})
    text = generate_inputs_electromagnetic_pic(spec)
    assert "dt_fusion.type = nuclearfusion" in text
    assert "dt_fusion.event_multiplier" in text
    assert "dt_fusion.product_species = helium4 neutron" in text
    assert "neutron.charge = 0.0" in text


def test_em_pic_generate_qed():
    """Generator emits QED Breit-Wheeler block."""
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e28, "ppc": 4},
        {"name": "positrons", "charge": 1, "mass_amu": 5.486e-4, "injection_style": "none"},
        {"name": "photons", "charge": 0, "mass_amu": 1e-30, "density": 1e28, "ppc": 4,
         "do_qed_breit_wheeler": True,
         "qed_bw_ele_product": "electrons", "qed_bw_pos_product": "positrons"},
    ]
    spec = ElectromagneticPICSpec.from_dict({"species": species})
    text = generate_inputs_electromagnetic_pic(spec)
    assert "photons.do_qed_breit_wheeler = 1" in text
    assert "photons.qed_breit_wheeler_ele_product_species = electrons" in text
    assert "photons.qed_breit_wheeler_pos_product_species = positrons" in text


def test_em_pic_generate_gaussian_beam():
    """Generator emits gaussian_beam injection block."""
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e20, "ppc": 4},
        {"name": "beam", "charge": -1, "mass_amu": 5.486e-4,
         "injection_style": "gaussian_beam",
         "x_rms": 2e-6, "y_rms": 2e-6, "z_rms": 4e-6,
         "z_cut": 3.0, "n_macro": 5000, "q_tot": -1e-9, "z_mean": -50e-6,
         "uz_m": 2000.0},
    ]
    spec = ElectromagneticPICSpec.from_dict({"species": species})
    text = generate_inputs_electromagnetic_pic(spec)
    assert "beam.injection_style = gaussian_beam" in text
    assert "beam.npart = 5000" in text
    assert "beam.q_tot" in text


def test_em_pic_generate_asymmetric_bc():
    """Asymmetric field_bc_lo/hi produce distinct boundary lines."""
    spec = _spec(
        field_bc_lo=["pml", "periodic"],
        field_bc_hi=["pml", "pml"],
    )
    text = generate_inputs_electromagnetic_pic(spec)
    lo_line = [l for l in text.splitlines() if "boundary.field_lo" in l][0]
    hi_line = [l for l in text.splitlines() if "boundary.field_hi" in l][0]
    assert lo_line != hi_line
    assert "pml periodic" in lo_line
    assert "pml pml" in hi_line


# ---------------------------------------------------------------------------
# AMR tests
# ---------------------------------------------------------------------------

def test_em_pic_amr_box_ok():
    """max_level=1 with valid blocking_factor and fine_tag → no AMR errors."""
    spec = _spec(
        amr_max_level=1, amr_blocking_factor=8,
        fine_tag_lo=[-25e-6, 50e-6], fine_tag_hi=[25e-6, 150e-6],
    )
    r = validate_electromagnetic_pic_spec(spec)
    amr_errors = [i for i in r.issues if i.code.startswith("amr.") and i.severity == Severity.ERROR]
    assert not amr_errors, amr_errors


def test_em_pic_amr_missing_tag():
    """max_level=1 with tag_by='box' but no fine_tag → ERROR amr.fine_tag.missing."""
    spec = _spec(amr_max_level=1, amr_blocking_factor=8)
    r = validate_electromagnetic_pic_spec(spec)
    assert any(i.code == "amr.fine_tag.missing" for i in r.issues)


def test_em_pic_amr_bad_blocking():
    """n_cell=[100, 256] not divisible by blocking_factor=8 → ERROR."""
    spec = _spec(
        amr_max_level=1, amr_blocking_factor=8, amr_tag_by="plasma",
        number_of_cells=[100, 256],
    )
    r = validate_electromagnetic_pic_spec(spec)
    assert any("amr.blocking_factor.divisibility" in i.code for i in r.issues)


def test_em_pic_generate_amr():
    """Generator emits amr.max_level and warpx.fine_tag_lo when AMR enabled."""
    spec = _spec(
        amr_max_level=1, amr_blocking_factor=8,
        fine_tag_lo=[-25e-6, 50e-6], fine_tag_hi=[25e-6, 150e-6],
    )
    text = generate_inputs_electromagnetic_pic(spec)
    assert "amr.max_level = 1" in text
    assert "warpx.fine_tag_lo" in text
    assert "warpx.fine_tag_hi" in text
    assert "amr.blocking_factor = 8" in text


def test_em_pic_dx_target():
    """dx_target in from_dict auto-computes number_of_cells rounded to blocking_factor."""
    spec = _spec(
        lower_bound=[-50e-6, 0.0], upper_bound=[50e-6, 200e-6],
        dx_target=1e-6, amr_blocking_factor=8,
    )
    assert len(spec.domain.number_of_cells) == 2
    for n in spec.domain.number_of_cells:
        assert n % 8 == 0
        assert n > 0


# ---------------------------------------------------------------------------
# New physics check tests
# ---------------------------------------------------------------------------

def test_em_pic_laser_resolution_error():
    """< 5 cells per laser wavelength → ERROR."""
    # wavelength=0.8μm, 10 cells over 1mm → dx=100μm → 0.008 cells/λ → ERROR
    spec = _spec(
        dim=1,
        number_of_cells=[10],
        lower_bound=[0.0],
        upper_bound=[1e-3],
        field_bc=["periodic"],
        wavelength=0.8e-6, a0=1.0, waist=100e-6, duration=30e-15,
        focal_position_z=0.5e-3, centroid_position_z=0.0,
    )
    r = validate_electromagnetic_pic_spec(spec)
    assert not r.ok
    issues = [i for i in r.issues if i.code == "em.laser_resolution"]
    assert issues and issues[0].severity == Severity.ERROR


def test_em_pic_laser_resolution_warns():
    """5–10 cells per laser wavelength → WARNING (not ERROR)."""
    # wavelength=0.8μm, 8000 cells over 1mm → dx=0.125μm → 6.4 cells/λ → WARNING
    spec = _spec(
        dim=1,
        number_of_cells=[8000],
        lower_bound=[0.0],
        upper_bound=[1e-3],
        field_bc=["periodic"],
        wavelength=0.8e-6, a0=1.0, waist=100e-6, duration=30e-15,
        focal_position_z=0.5e-3, centroid_position_z=0.0,
    )
    r = validate_electromagnetic_pic_spec(spec)
    issues = [i for i in r.issues if i.code == "em.laser_resolution"]
    assert issues and issues[0].severity == Severity.WARNING
    assert r.ok  # warning only


def test_em_pic_laser_resolution_ok():
    """≥ 10 cells per laser wavelength → no laser resolution issue."""
    # wavelength=0.8μm, 20000 cells over 1mm → dx=0.05μm → 16 cells/λ → OK
    spec = _spec(
        dim=1,
        number_of_cells=[20000],
        lower_bound=[0.0],
        upper_bound=[1e-3],
        field_bc=["periodic"],
        wavelength=0.8e-6, a0=1.0, waist=100e-6, duration=30e-15,
        focal_position_z=0.5e-3, centroid_position_z=0.0,
    )
    r = validate_electromagnetic_pic_spec(spec)
    assert not any(i.code == "em.laser_resolution" for i in r.issues)


def test_em_pic_debye_resolution_warning_only():
    """dx > λ_De for species with temperature → WARNING only (not ERROR) for EM-PIC."""
    # electrons: n=1e24, Te=1eV → λ_De≈7.4nm; dx=78nm (16 cells over 1.25μm) → ratio≈10.5 → WARNING
    species = [
        {"name": "electrons", "charge": -1, "mass_amu": 5.486e-4, "density": 1e24,
         "ppc": 4, "temperature_eV": 1.0},
        {"name": "protons", "charge": 1, "mass_amu": 1.00728, "density": 1e24, "ppc": 4},
    ]
    spec = ElectromagneticPICSpec.from_dict({
        "species": species,
        "dim": 1,
        "number_of_cells": [16],
        "lower_bound": [0.0],
        "upper_bound": [1.25e-6],
        "field_bc": ["periodic"],
    })
    r = validate_electromagnetic_pic_spec(spec)
    issues = [i for i in r.issues if i.code == "em.debye_resolution"]
    assert issues and issues[0].severity == Severity.WARNING
    # Must still be OK overall (only a warning)
    errors = [i for i in r.issues if i.severity == Severity.ERROR]
    assert not errors


# ---------------------------------------------------------------------------
# PICMI generator smoke tests
# ---------------------------------------------------------------------------

def test_em_pic_picmi_generates_script():
    from pywarpx.inputgen.electromagnetic_pic import generate_picmi_electromagnetic_pic
    spec = _spec()
    script = generate_picmi_electromagnetic_pic(spec)
    assert isinstance(script, str) and script
    assert "from pywarpx import picmi" in script


def test_em_pic_picmi_correct_solver():
    from pywarpx.inputgen.electromagnetic_pic import generate_picmi_electromagnetic_pic
    spec = _spec()
    script = generate_picmi_electromagnetic_pic(spec)
    assert "ElectromagneticSolver" in script


def test_em_pic_picmi_has_diagnostics():
    from pywarpx.inputgen.electromagnetic_pic import generate_picmi_electromagnetic_pic
    spec = _spec()
    script = generate_picmi_electromagnetic_pic(spec)
    assert "sim.add_diagnostic" in script
    assert "sim.initialize_inputs" in script
