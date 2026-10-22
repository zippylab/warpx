# Untracked validation scripts

Scripts that validate work on this branch but are deliberately **not** tracked
by git. They live under `test_dir/`, which `.gitignore` excludes, so that
machine-specific scaffolding cannot reach a pull request by accident.

This file is the record of what exists and where, since git will not show it.

## `test_dir/sycl_slm_deposition/`

Validates the SYCL shared-memory (SLM) deposition port in `2b4cc0be5`. All
three scripts are Sunspot-specific: they hardcode the project allocation, the
science venv, and `build_Release/`.

| File | What it does |
|---|---|
| `run_slm_tests.pbs` | The full validation, in three steps: the pre-existing deposition unit tests as a regression check, a 40-step Langmuir run comparing shared against direct deposition, and the shared-memory unit tests added in `c06bf8763`. |
| `compare_slm_fields.py` | Diffs two WarpX plotfiles cell by cell over Ex/Ey/Ez, Bx/By/Bz, jx/jy/jz, rho and divE. Used by step 2 above. Not machine-specific, and the piece most likely to be worth tracking later if a comparison of this kind is ever wanted in-tree. |
| `verify_clean.pbs` | Runs only the four shared-memory unit tests. Started life as the negative control: the same script with a deliberate one-cell perturbation added to the test, to confirm the comparison can actually fail. |

Alongside them are the PBS logs and output directories of the runs that
produced the results quoted in `reports/sycl_shared_deposition_report.md` and
in the commit message of `c06bf8763`.

### Results, on Sunspot (Intel PVC, 2 ranks)

- The 7 pre-existing deposition tests pass, so the port is not a regression.
- All 11 fields of the Langmuir run agree between shared and direct
  deposition, worst relative error 1.85e-12 against a 1e-9 tolerance.
- The 4 new shared-memory unit tests pass.
- Negative control: perturbing one cell of the shared result by 0.1% fails
  the test, 1 mismatch in 4624 cells. The comparison is load-bearing.

### Note for anyone re-running these

Aurora puts the frameworks `site-packages` on `PYTHONPATH` at login, ahead of
the science venv. pytest's plugin autoload then reads an entry point from one
platformdirs install and tries to import it from the other, which fails
before any test runs. Both `.pbs` scripts put the venv ahead of frameworks to
avoid this. The same precedence problem has bitten the CMake Python detection
and the dependency install, so it is worth suspecting first whenever a tool
resolves the wrong package here.
