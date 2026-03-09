from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .electrostatic_plasma import ElectrostaticPlasmaSpec, generate_inputs_electrostatic_plasma
from .electrostatic_plasma_validate import validate_electrostatic_plasma_spec
from .generate import generate_picmi_uniform_plasma
from .hybrid_plasma import HybridPlasmaSpec, generate_inputs_hybrid_plasma
from .hybrid_plasma_validate import validate_hybrid_plasma_spec
from .laser_acceleration import LaserAccelerationSpec, generate_picmi_laser_acceleration
from .laser_acceleration_native import generate_inputs_laser_acceleration
from .laser_acceleration_validate import validate_laser_acceleration_spec
from .native import generate_inputs_uniform_plasma
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

    raise RuntimeError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
