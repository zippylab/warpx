from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .electromagnetic_pic import ElectromagneticPICSpec, generate_inputs_electromagnetic_pic
from .electromagnetic_pic_validate import validate_electromagnetic_pic_spec
from .electrostatic_pic import ElectrostaticPICSpec, generate_inputs_electrostatic_pic
from .electrostatic_pic_validate import validate_electrostatic_pic_spec
from .electrostatic_plasma import ElectrostaticPlasmaSpec, generate_inputs_electrostatic_plasma
from .electrostatic_plasma_validate import validate_electrostatic_plasma_spec
from .generate import generate_picmi_uniform_plasma
from .hybrid_plasma import HybridPlasmaSpec, generate_inputs_hybrid_plasma
from .hybrid_plasma_validate import validate_hybrid_plasma_spec
from .ion_beam_instability import IonBeamInstabilitySpec, generate_inputs_ion_beam_instability
from .ion_beam_instability_validate import validate_ion_beam_instability_spec
from .laser_acceleration import LaserAccelerationSpec, generate_picmi_laser_acceleration
from .laser_acceleration_native import generate_inputs_laser_acceleration
from .laser_acceleration_validate import validate_laser_acceleration_spec
from .magnetic_reconnection import MagneticReconnectionSpec, generate_inputs_magnetic_reconnection
from .magnetic_reconnection_validate import validate_magnetic_reconnection_spec
from .native import generate_inputs_uniform_plasma
from .pwfa import PWFASpec, generate_inputs_pwfa
from .pwfa_validate import validate_pwfa_spec
from .blocks import suggest_cells
from .spec import UniformPlasmaSpec
from .validate import validate_picmi_syntax, validate_uniform_plasma_spec


def _load_electrostatic_plasma_spec(path: str) -> ElectrostaticPlasmaSpec:
    data = json.loads(Path(path).read_text())
    return ElectrostaticPlasmaSpec.from_dict(data)


def _load_uniform_plasma_spec(path: str) -> UniformPlasmaSpec:
    data = json.loads(Path(path).read_text())
    return UniformPlasmaSpec(**data)


def _load_laser_acceleration_spec(path: str) -> LaserAccelerationSpec:
    data = json.loads(Path(path).read_text())
    return LaserAccelerationSpec.from_dict(data)


def _load_hybrid_plasma_spec(path: str) -> HybridPlasmaSpec:
    data = json.loads(Path(path).read_text())
    return HybridPlasmaSpec.from_dict(data)


def _load_ion_beam_instability_spec(path: str) -> IonBeamInstabilitySpec:
    data = json.loads(Path(path).read_text())
    return IonBeamInstabilitySpec.from_dict(data)


def _load_magnetic_reconnection_spec(path: str) -> MagneticReconnectionSpec:
    data = json.loads(Path(path).read_text())
    return MagneticReconnectionSpec.from_dict(data)


def _load_pwfa_spec(path: str) -> PWFASpec:
    data = json.loads(Path(path).read_text())
    return PWFASpec.from_dict(data)


def _load_electromagnetic_pic_spec(path: str) -> ElectromagneticPICSpec:
    data = json.loads(Path(path).read_text())
    return ElectromagneticPICSpec.from_dict(data)


def _load_electrostatic_pic_spec(path: str) -> ElectrostaticPICSpec:
    data = json.loads(Path(path).read_text())
    return ElectrostaticPICSpec.from_dict(data)


def _apply_picmi_dry_run(script: str) -> str:
    """Patch a PICMI script to only call initialize_inputs() and then exit."""

    marker = "sim.initialize_inputs()"
    if marker not in script:
        raise RuntimeError("initialize_inputs marker not found")

    # Keep everything up to the initialize_inputs marker, then print a sentinel.
    prefix = script.split(marker, 1)[0]
    return prefix + marker + "\nprint(\"DRYRUN_OK\")\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="warpx-inputgen")
    sub = p.add_subparsers(dest="cmd", required=True)

    gen_es = sub.add_parser(
        "gen-electrostatic-plasma-native",
        help="Generate a native WarpX inputs file from an ElectrostaticPlasmaSpec JSON",
    )
    gen_es.add_argument("spec_json", help="Path to JSON spec")
    gen_es.add_argument("--out", required=True, help="Output inputs file path")

    val_es = sub.add_parser(
        "validate-electrostatic-plasma",
        help="Validate an ElectrostaticPlasmaSpec JSON",
    )
    val_es.add_argument("spec_json", help="Path to JSON spec")

    gen = sub.add_parser("gen-uniform-plasma", help="Generate a PICMI script from a UniformPlasmaSpec JSON")
    gen.add_argument("spec_json", help="Path to JSON spec")
    gen.add_argument("--out", required=True, help="Output PICMI script path")

    gen_native = sub.add_parser(
        "gen-uniform-plasma-native",
        help="Generate a native WarpX inputs file from a UniformPlasmaSpec JSON",
    )
    gen_native.add_argument("spec_json", help="Path to JSON spec")
    gen_native.add_argument("--out", required=True, help="Output inputs file path")

    val = sub.add_parser("validate-uniform-plasma", help="Validate a UniformPlasmaSpec JSON")
    val.add_argument("spec_json", help="Path to JSON spec")

    gen_laser = sub.add_parser("gen-laser-acceleration", help="Generate a PICMI script from a LaserAccelerationSpec JSON")
    gen_laser.add_argument("spec_json", help="Path to JSON spec")
    gen_laser.add_argument("--out", required=True, help="Output PICMI script path")
    gen_laser.add_argument(
        "--dry-run",
        action="store_true",
        help="Patch the generated PICMI script to only run initialize_inputs() and exit",
    )
    gen_laser.add_argument(
        "--emit-native-inputs",
        default=None,
        help="Also emit a native inputs file by patching the generated script to call sim.write_input_file(file_name=...).",
    )

    gen_laser_native = sub.add_parser(
        "gen-laser-acceleration-native",
        help="Generate a native WarpX inputs file from a LaserAccelerationSpec JSON",
    )
    gen_laser_native.add_argument("spec_json", help="Path to JSON spec")
    gen_laser_native.add_argument("--out", required=True, help="Output inputs file path")

    val_laser = sub.add_parser("validate-laser-acceleration", help="Validate a LaserAccelerationSpec JSON")
    val_laser.add_argument("spec_json", help="Path to JSON spec")

    gen_hybrid = sub.add_parser(
        "gen-hybrid-plasma-native",
        help="Generate a native WarpX inputs file from a HybridPlasmaSpec JSON",
    )
    gen_hybrid.add_argument("spec_json", help="Path to JSON spec")
    gen_hybrid.add_argument("--out", required=True, help="Output inputs file path")

    val_hybrid = sub.add_parser(
        "validate-hybrid-plasma",
        help="Validate a HybridPlasmaSpec JSON",
    )
    val_hybrid.add_argument("spec_json", help="Path to JSON spec")

    gen_beam = sub.add_parser(
        "gen-ion-beam-instability-native",
        help="Generate a native WarpX inputs file from an IonBeamInstabilitySpec JSON",
    )
    gen_beam.add_argument("spec_json", help="Path to JSON spec")
    gen_beam.add_argument("--out", required=True, help="Output inputs file path")

    val_beam = sub.add_parser(
        "validate-ion-beam-instability",
        help="Validate an IonBeamInstabilitySpec JSON",
    )
    val_beam.add_argument("spec_json", help="Path to JSON spec")

    gen_recon = sub.add_parser(
        "gen-magnetic-reconnection-native",
        help="Generate a native WarpX inputs file from a MagneticReconnectionSpec JSON",
    )
    gen_recon.add_argument("spec_json", help="Path to JSON spec")
    gen_recon.add_argument("--out", required=True, help="Output inputs file path")

    val_recon = sub.add_parser(
        "validate-magnetic-reconnection",
        help="Validate a MagneticReconnectionSpec JSON",
    )
    val_recon.add_argument("spec_json", help="Path to JSON spec")

    gen_pwfa = sub.add_parser(
        "gen-pwfa-native",
        help="Generate a native WarpX inputs file from a PWFASpec JSON",
    )
    gen_pwfa.add_argument("spec_json", help="Path to JSON spec")
    gen_pwfa.add_argument("--out", required=True, help="Output inputs file path")

    val_pwfa = sub.add_parser(
        "validate-pwfa",
        help="Validate a PWFASpec JSON",
    )
    val_pwfa.add_argument("spec_json", help="Path to JSON spec")

    gen_em_pic = sub.add_parser(
        "gen-electromagnetic-pic-native",
        help="Generate a native WarpX inputs file from an ElectromagneticPICSpec JSON",
    )
    gen_em_pic.add_argument("spec_json", help="Path to JSON spec")
    gen_em_pic.add_argument("--out", required=True, help="Output inputs file path")

    val_em_pic = sub.add_parser(
        "validate-electromagnetic-pic",
        help="Validate an ElectromagneticPICSpec JSON",
    )
    val_em_pic.add_argument("spec_json", help="Path to JSON spec")

    gen_es_pic = sub.add_parser(
        "gen-electrostatic-pic-native",
        help="Generate a native WarpX inputs file from an ElectrostaticPICSpec JSON",
    )
    gen_es_pic.add_argument("spec_json", help="Path to JSON spec")
    gen_es_pic.add_argument("--out", required=True, help="Output inputs file path")

    val_es_pic = sub.add_parser(
        "validate-electrostatic-pic",
        help="Validate an ElectrostaticPICSpec JSON",
    )
    val_es_pic.add_argument("spec_json", help="Path to JSON spec")

    sc = sub.add_parser(
        "suggest-cells",
        help="Given domain bounds and target cell size, compute number_of_cells",
    )
    sc.add_argument(
        "spec_json",
        nargs="?",
        default=None,
        help="Optional JSON file with lower_bound, upper_bound, dx_target, [amr_blocking_factor]",
    )
    sc.add_argument("--lower-bound", help="Space-separated lower bound coords (e.g. '0.0 0.0')")
    sc.add_argument("--upper-bound", help="Space-separated upper bound coords (e.g. '1e-3 1e-3')")
    sc.add_argument("--dx-target", type=float, help="Target cell size in metres")
    sc.add_argument("--blocking-factor", type=int, default=8, help="AMReX blocking factor (default: 8)")

    args = p.parse_args(argv)

    if args.cmd == "gen-electrostatic-plasma-native":
        spec = _load_electrostatic_plasma_spec(args.spec_json)
        report = validate_electrostatic_plasma_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2
        text = generate_inputs_electrostatic_plasma(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-electrostatic-plasma":
        spec = _load_electrostatic_plasma_spec(args.spec_json)
        report = validate_electrostatic_plasma_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "validate-uniform-plasma":
        spec = _load_uniform_plasma_spec(args.spec_json)
        report = validate_uniform_plasma_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "gen-uniform-plasma-native":
        spec = _load_uniform_plasma_spec(args.spec_json)
        report = validate_uniform_plasma_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2

        text = generate_inputs_uniform_plasma(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "gen-uniform-plasma":
        spec = _load_uniform_plasma_spec(args.spec_json)
        report = validate_uniform_plasma_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2

        script = generate_picmi_uniform_plasma(spec)
        syn_report, _ = validate_picmi_syntax(script)
        if not syn_report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in syn_report.issues]}, indent=2, default=str))
            return 2

        Path(args.out).write_text(script)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "gen-laser-acceleration-native":
        spec = _load_laser_acceleration_spec(args.spec_json)
        report = validate_laser_acceleration_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2

        text = generate_inputs_laser_acceleration(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-laser-acceleration":
        spec = _load_laser_acceleration_spec(args.spec_json)
        report = validate_laser_acceleration_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "gen-laser-acceleration":
        spec = _load_laser_acceleration_spec(args.spec_json)
        report = validate_laser_acceleration_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2

        script = generate_picmi_laser_acceleration(spec)

        # Mutually exclusive modes
        if args.dry_run and args.emit_native_inputs:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "--dry-run and --emit-native-inputs are mutually exclusive",
                    },
                    indent=2,
                )
            )
            return 2

        if args.emit_native_inputs:
            # Patch the script to only write a native inputs file, then exit.
            # This requires the PICMI Python dependencies, but avoids `initialize_warpx()`.
            # Note: `write_input_file()` calls `initialize_inputs()`, so it still requires
            # that the Python interface can set up inputs.
            marker = "sim.initialize_inputs()"
            if marker not in script:
                print(json.dumps({"ok": False, "error": "initialize_inputs marker not found"}, indent=2))
                return 2

            prefix = script.split(marker, 1)[0]
            native_name = args.emit_native_inputs
            script = (
                prefix
                + f"sim.write_input_file(file_name={native_name!r})\n"
                + "print(\"WROTE_NATIVE_INPUTS\")\n"
            )

        elif args.dry_run:
            script = _apply_picmi_dry_run(script)

        syn_report, _ = validate_picmi_syntax(script)
        if not syn_report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in syn_report.issues]}, indent=2, default=str))
            return 2

        Path(args.out).write_text(script)
        print(
            json.dumps(
                {
                    "ok": True,
                    "out": args.out,
                    "dry_run": bool(args.dry_run),
                    "emit_native_inputs": args.emit_native_inputs,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "gen-hybrid-plasma-native":
        spec = _load_hybrid_plasma_spec(args.spec_json)
        report = validate_hybrid_plasma_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2
        text = generate_inputs_hybrid_plasma(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-hybrid-plasma":
        spec = _load_hybrid_plasma_spec(args.spec_json)
        report = validate_hybrid_plasma_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "gen-ion-beam-instability-native":
        spec = _load_ion_beam_instability_spec(args.spec_json)
        report = validate_ion_beam_instability_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2
        text = generate_inputs_ion_beam_instability(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-ion-beam-instability":
        spec = _load_ion_beam_instability_spec(args.spec_json)
        report = validate_ion_beam_instability_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "gen-magnetic-reconnection-native":
        spec = _load_magnetic_reconnection_spec(args.spec_json)
        report = validate_magnetic_reconnection_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2
        text = generate_inputs_magnetic_reconnection(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-magnetic-reconnection":
        spec = _load_magnetic_reconnection_spec(args.spec_json)
        report = validate_magnetic_reconnection_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "gen-pwfa-native":
        spec = _load_pwfa_spec(args.spec_json)
        report = validate_pwfa_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2
        text = generate_inputs_pwfa(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-pwfa":
        spec = _load_pwfa_spec(args.spec_json)
        report = validate_pwfa_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "gen-electromagnetic-pic-native":
        spec = _load_electromagnetic_pic_spec(args.spec_json)
        report = validate_electromagnetic_pic_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2
        text = generate_inputs_electromagnetic_pic(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-electromagnetic-pic":
        spec = _load_electromagnetic_pic_spec(args.spec_json)
        report = validate_electromagnetic_pic_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "gen-electrostatic-pic-native":
        spec = _load_electrostatic_pic_spec(args.spec_json)
        report = validate_electrostatic_pic_spec(spec)
        if not report.ok:
            print(json.dumps({"ok": False, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
            return 2
        text = generate_inputs_electrostatic_pic(spec)
        Path(args.out).write_text(text)
        print(json.dumps({"ok": True, "out": args.out}, indent=2))
        return 0

    if args.cmd == "validate-electrostatic-pic":
        spec = _load_electrostatic_pic_spec(args.spec_json)
        report = validate_electrostatic_pic_spec(spec)
        print(json.dumps({"ok": report.ok, "issues": [i.__dict__ for i in report.issues]}, indent=2, default=str))
        return 0 if report.ok else 2

    if args.cmd == "suggest-cells":
        if args.spec_json is not None:
            data = json.loads(Path(args.spec_json).read_text())
            lo = data["lower_bound"]
            hi = data["upper_bound"]
            dx = data["dx_target"]
            bf = data.get("amr_blocking_factor", args.blocking_factor)
        else:
            if args.lower_bound is None or args.upper_bound is None or args.dx_target is None:
                p.error("suggest-cells requires either spec_json or --lower-bound/--upper-bound/--dx-target")
            lo = [float(x) for x in args.lower_bound.split()]
            hi = [float(x) for x in args.upper_bound.split()]
            dx = args.dx_target
            bf = args.blocking_factor
        cells = suggest_cells(lo, hi, dx, bf)
        dx_actual = [(hi[i] - lo[i]) / cells[i] for i in range(len(cells))]
        print(json.dumps({"number_of_cells": cells, "blocking_factor": bf, "dx_actual": dx_actual}, indent=2))
        return 0

    raise RuntimeError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
