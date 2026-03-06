# warpx-inputgen Integration Reference

This document describes the `warpx-inputgen` CLI contract for external callers
(MCP servers, CI scripts, etc.).  For the full borealis-mcp integration design
see `borealis-mcp/src/borealis_mcp/applications/warpx/INTEGRATION.md`.

## CLI conventions

- All subcommands take a JSON spec file as primary positional argument.
- All subcommands write a JSON object to **stdout**.
- Exit code **0** = success.  Exit code **2** = validation error or bad input.
- Callers should capture stdout and parse JSON before inspecting exit code.

## Subcommands

### validate-laser-acceleration

```
warpx-inputgen validate-laser-acceleration <spec.json>
```

Schema + physics sanity checks on a `LaserAccelerationSpec`.  No files written.

### gen-laser-acceleration

```
warpx-inputgen gen-laser-acceleration <spec.json> --out <script.py>
             [--dry-run | --emit-native-inputs <inputs_path>]
```

Generates a PICMI Python script.

| Flag | Behaviour | Sentinel on stdout |
|---|---|---|
| (none) | Full PICMI script; runs `initialize_warpx()` + `step()` | — |
| `--dry-run` | Stops after `initialize_inputs()`; safe on login nodes | `DRYRUN_OK` |
| `--emit-native-inputs <path>` | Calls `write_input_file(file_name=<path>)`; produces native inputs | `WROTE_NATIVE_INPUTS` |

`--dry-run` and `--emit-native-inputs` are mutually exclusive.

### validate-uniform-plasma

```
warpx-inputgen validate-uniform-plasma <spec.json>
```

### gen-uniform-plasma

```
warpx-inputgen gen-uniform-plasma <spec.json> --out <script.py>
```

### gen-uniform-plasma-native

```
warpx-inputgen gen-uniform-plasma-native <spec.json> --out <inputs_file>
```

Writes an AMReX ParmParse native inputs file directly (no PICMI).

## stdout JSON schemas

**Success (generation):**
```json
{"ok": true, "out": "/path/to/output", "dry_run": false, "emit_native_inputs": null}
```

**Success (validation only):**
```json
{"ok": true, "issues": []}
```

**Failure:**
```json
{
  "ok": false,
  "issues": [
    {"severity": "error", "code": "CODE", "message": "...", "details": {}}
  ]
}
```

Severity values: `"error"` (blocks generation), `"warning"` (advisory).

## Environment dependencies

| Variable | Required by | Purpose |
|---|---|---|
| `WARPX_PYBIND_PATH` | Generated PICMI scripts | Dir containing `warpx_pybind_*.so` |
| `LD_LIBRARY_PATH` | `import pywarpx` | WarpX + openPMD shared libs |

`warpx-inputgen` itself (validation and generation) does not need `WARPX_PYBIND_PATH`.
Only the *generated* PICMI scripts need it, at run time.
