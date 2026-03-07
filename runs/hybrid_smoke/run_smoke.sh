#!/bin/bash
# Smoke test: hybrid-PIC native inputs, max_step=0.
# Run on a sunspot/aurora compute node (1 GPU tile sufficient).
#
# Usage:
#   On login node:  copy this directory to sunspot, then:
#   qsub run_smoke.pbs
#   -- OR --
#   Interactive node:  bash run_smoke.sh

WARPX_INSTALL="${WARPX_INSTALL:-/lus/flare/projects/catalyst/world_shared/warpx}"

# 1D hybrid-PIC → look for a 1D binary first, fall back to generic name patterns
WARPX_BIN=$(ls \
    "${WARPX_INSTALL}"/bin/warpx.1d.MPI.OMP.DP.OPMD \
    "${WARPX_INSTALL}"/bin/warpx.1d* \
    "${WARPX_INSTALL}"/bin/warpx* \
    2>/dev/null | head -1)

if [ -z "$WARPX_BIN" ]; then
    echo "ERROR: no WarpX binary found under ${WARPX_INSTALL}/bin" >&2
    exit 1
fi

echo "Using binary: $WARPX_BIN"
echo "Inputs:       $(pwd)/inputs"

mpiexec -n 1 "$WARPX_BIN" inputs max_step=0 2>&1 | tee log_smoke.out
rc=${PIPESTATUS[0]}

echo ""
if [ $rc -eq 0 ]; then
    echo "SMOKE TEST PASSED (exit $rc)"
else
    echo "SMOKE TEST FAILED (exit $rc)"
fi
exit $rc
