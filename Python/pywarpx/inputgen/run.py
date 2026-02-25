from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .generate import generate_picmi_uniform_plasma
from .native import generate_inputs_uniform_plasma
from .spec import Severity, UniformPlasmaSpec, ValidationReport


def dry_run_picmi_initialize_inputs(
    spec: UniformPlasmaSpec,
    python_exe: str = sys.executable,
    extra_env: Optional[dict[str, str]] = None,
) -> ValidationReport:
    """Run a generated PICMI script in a dry-run mode (initialize_inputs only).

    This is meant for environments where pywarpx + PICMI deps are installed.
    It will not call `initialize_warpx()` or `step()`.
    """

    report = ValidationReport()

    script = generate_picmi_uniform_plasma(spec)

    # Patch script to stop after initialize_inputs
    marker = "sim.initialize_inputs()"
    if marker not in script:
        report.add(Severity.ERROR, "picmi.dryrun.patch_failed", "Could not find initialize_inputs marker")
        return report

    patched = script.split(marker)[0] + marker + "\nprint('DRYRUN_OK')\n"

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "inputs_picmi_dryrun.py"
        p.write_text(patched)

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        try:
            out = subprocess.check_output([python_exe, str(p)], stderr=subprocess.STDOUT, env=env, text=True)
            if "DRYRUN_OK" not in out:
                report.add(Severity.WARNING, "picmi.dryrun.no_marker", "Dry-run did not emit expected marker", output=out)
        except subprocess.CalledProcessError as e:
            report.add(
                Severity.ERROR,
                "picmi.dryrun.failed",
                "PICMI dry-run failed during initialize_inputs()",
                returncode=e.returncode,
                output=e.output,
            )

    return report


def dry_run_native_max_step_zero(
    spec: UniformPlasmaSpec,
    warpx_exe: str,
    extra_args: Optional[list[str]] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> ValidationReport:
    """Run a native inputs file with `max_step=0` to validate initialization.

    Intended for environments where a `warpx.*` binary is available.
    """

    report = ValidationReport()

    inputs_text = generate_inputs_uniform_plasma(spec)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        inputs_path = td_path / "inputs"
        inputs_path.write_text(inputs_text)

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        cmd = [warpx_exe, str(inputs_path), "max_step=0"]
        if extra_args:
            cmd.extend(extra_args)

        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env, text=True)

            # Surface unused inputs as warnings (useful when template knobs are ignored).
            if "Unused ParmParse Variables:" in out:
                tail = out.split("Unused ParmParse Variables:", 1)[1]
                block = tail.split("\n\n", 1)[0].strip()
                report.add(
                    Severity.WARNING,
                    "native.dryrun.unused_inputs",
                    "WarpX reported unused ParmParse variables",
                    unused_block=block,
                )

            if "Abort" in out or "### ERROR" in out:
                report.add(
                    Severity.WARNING,
                    "native.dryrun.suspicious_output",
                    "WarpX output contains ERROR/Abort",
                    output=out,
                )

        except FileNotFoundError:
            report.add(Severity.ERROR, "native.dryrun.no_exe", "WarpX executable not found", warpx_exe=warpx_exe)
        except subprocess.CalledProcessError as e:
            report.add(
                Severity.ERROR,
                "native.dryrun.failed",
                "Native dry-run failed (max_step=0)",
                returncode=e.returncode,
                output=e.output,
                cmd=cmd,
                spec=asdict(spec),
            )

    return report
