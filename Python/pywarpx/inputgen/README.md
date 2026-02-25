# pywarpx.inputgen

A small, intentionally conservative input generation and validation toolkit.

## Goals

- Convert a narrow, structured *spec* into runnable WarpX inputs
- Validate specs/outputs to reduce garbage-in runs
- Be callable from external workflow tools (e.g. MCP servers)

## Current scope (vertical slice)

- Uniform plasma (electron-only) PICMI script generation
- Validation:
  - schema checks
  - basic physics sanity checks (Debye length heuristic, CFL estimate)
  - PICMI script syntax check

## CLI

Installed as `warpx-inputgen` (via `console_scripts`).

Validate a spec:

```bash
warpx-inputgen validate-uniform-plasma spec.json
```

Generate a PICMI script:

```bash
warpx-inputgen gen-uniform-plasma spec.json --out inputs_picmi.py
```
