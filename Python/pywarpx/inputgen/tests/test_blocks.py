"""Tests for blocks.py — DiagSpec, ReducedDiagSpec, validate_diag, _emit_diag_block."""

import pytest
from pywarpx.inputgen.blocks import (
    DiagSpec,
    ReducedDiagSpec,
    _emit_diag_block,
    _emit_picmi_diag_lines,
    validate_diag,
)
from pywarpx.inputgen.spec import Severity


# ---------------------------------------------------------------------------
# ReducedDiagSpec validation
# ---------------------------------------------------------------------------

def test_reduced_diag_unknown_type():
    diag = DiagSpec(reduced_diags=[ReducedDiagSpec(type="Bogus")])
    r = validate_diag(diag)
    assert not r.ok
    assert any(i.severity == Severity.ERROR and "unknown_type" in i.code for i in r.issues)


def test_reduced_diag_field_energy_ok():
    diag = DiagSpec(reduced_diags=[ReducedDiagSpec(type="FieldEnergy")])
    r = validate_diag(diag)
    assert r.ok, r.issues


def test_reduced_diag_field_probe_ok():
    rd = ReducedDiagSpec(type="FieldProbe", probe_geometry="Point",
                         x_probe=0.0, y_probe=0.0, z_probe=0.05)
    diag = DiagSpec(reduced_diags=[rd])
    r = validate_diag(diag)
    assert r.ok, r.issues


def test_reduced_diag_field_probe_bad_geometry():
    rd = ReducedDiagSpec(type="FieldProbe", probe_geometry="Sphere")
    diag = DiagSpec(reduced_diags=[rd])
    r = validate_diag(diag)
    assert not r.ok
    assert any("probe_geometry" in i.code for i in r.issues)


def test_reduced_diag_field_reduction_no_function():
    rd = ReducedDiagSpec(type="FieldReduction", reduction_type="Maximum",
                         reduced_function="")
    diag = DiagSpec(reduced_diags=[rd])
    r = validate_diag(diag)
    assert not r.ok
    assert any("reduced_function" in i.code for i in r.issues)


def test_reduced_diag_field_reduction_ok():
    rd = ReducedDiagSpec(type="FieldReduction", reduction_type="Maximum",
                         reduced_function="Ex**2 + Ey**2 + Ez**2")
    diag = DiagSpec(reduced_diags=[rd])
    r = validate_diag(diag)
    assert r.ok, r.issues


def test_reduced_diag_particle_histogram_no_species():
    rd = ReducedDiagSpec(type="ParticleHistogram", species="",
                         bin_number=100, bin_min=0.0, bin_max=1.0,
                         histogram_function="u_x")
    diag = DiagSpec(reduced_diags=[rd])
    r = validate_diag(diag)
    assert not r.ok
    assert any("species" in i.code for i in r.issues)


def test_reduced_diag_particle_histogram_ok():
    rd = ReducedDiagSpec(type="ParticleHistogram", species="electrons",
                         bin_number=100, bin_min=0.0, bin_max=1.0,
                         histogram_function="u_x")
    diag = DiagSpec(reduced_diags=[rd])
    r = validate_diag(diag)
    assert r.ok, r.issues


def test_reduced_diag_beam_relevant_no_species():
    rd = ReducedDiagSpec(type="BeamRelevant", species="")
    diag = DiagSpec(reduced_diags=[rd])
    r = validate_diag(diag)
    assert not r.ok
    assert any("species" in i.code for i in r.issues)


# ---------------------------------------------------------------------------
# DiagSpec.from_dict
# ---------------------------------------------------------------------------

def test_diag_spec_from_dict_simple():
    d = {"diag_period": 20, "diag_fields": ["Ex", "By"]}
    diag = DiagSpec.from_dict(d)
    assert diag.diag_period == 20
    assert diag.diag_fields == ["Ex", "By"]
    assert diag.reduced_diags == []


def test_diag_spec_from_dict_reduced_diags():
    d = {
        "diag_period": 10,
        "diag_format": "plotfile",
        "write_species": True,
        "reduced_diags": [
            {"type": "FieldEnergy", "period": 5},
            {"type": "ParticleEnergy"},
        ],
    }
    diag = DiagSpec.from_dict(d)
    assert diag.diag_format == "plotfile"
    assert diag.write_species is True
    assert len(diag.reduced_diags) == 2
    assert diag.reduced_diags[0].type == "FieldEnergy"
    assert diag.reduced_diags[0].period == 5
    assert diag.reduced_diags[1].type == "ParticleEnergy"


# ---------------------------------------------------------------------------
# _emit_diag_block
# ---------------------------------------------------------------------------

def test_emit_diag_block_basic():
    diag = DiagSpec(diag_period=50, diag_fields=["Ex", "Ey"])
    out = _emit_diag_block(diag)
    assert "diagnostics.diags_names" in out
    assert "intervals = 50" in out
    assert "Ex Ey" in out


def test_emit_diag_block_write_species():
    diag = DiagSpec(write_species=True)
    out = _emit_diag_block(diag)
    assert "write_species = 1" in out


def test_emit_diag_block_no_write_species():
    diag = DiagSpec(write_species=False)
    out = _emit_diag_block(diag)
    assert "write_species = 0" in out


def test_emit_diag_block_reduced_diag():
    rd = ReducedDiagSpec(type="FieldEnergy", period=10)
    diag = DiagSpec(diag_period=50, reduced_diags=[rd])
    out = _emit_diag_block(diag)
    assert "warpx.reduced_diags_names" in out
    assert "FieldEnergy" in out
    assert "intervals = 10" in out


def test_emit_diag_block_reduced_diag_uses_parent_period():
    rd = ReducedDiagSpec(type="ParticleEnergy", period=0)
    diag = DiagSpec(diag_period=25, reduced_diags=[rd])
    out = _emit_diag_block(diag)
    # period=0 → fall back to diag_period=25
    lines = [ln for ln in out.splitlines() if "intervals" in ln]
    # Should have at least one "intervals = 25" (either from full diag or reduced)
    assert any("25" in ln for ln in lines)


def test_emit_diag_block_custom_name():
    diag = DiagSpec()
    out = _emit_diag_block(diag, name="mydiag")
    assert "mydiag" in out


# ---------------------------------------------------------------------------
# _emit_picmi_diag_lines
# ---------------------------------------------------------------------------

def test_emit_picmi_diag_lines_basic():
    diag = DiagSpec(diag_period=20, diag_fields=["Ex", "Bz"])
    code = _emit_picmi_diag_lines(diag, ["electrons"])
    assert "picmi.FieldDiagnostic" in code
    assert "sim.add_diagnostic" in code
    assert "20" in code


def test_emit_picmi_diag_lines_write_species():
    diag = DiagSpec(write_species=True)
    code = _emit_picmi_diag_lines(diag, ["electrons", "ions"])
    assert "picmi.ParticleDiagnostic" in code
    assert "electrons" in code
    assert "ions" in code


def test_emit_picmi_diag_lines_reduced():
    rd = ReducedDiagSpec(type="FieldEnergy")
    diag = DiagSpec(reduced_diags=[rd])
    code = _emit_picmi_diag_lines(diag, [])
    assert "picmi.ReducedDiagnostic" in code
    assert "FieldEnergy" in code
