#!/bin/bash
# Fast JIT build of gpu_rng_test — skips SYCL AOT offline compilation.
#
# Compared to the cmake AOT build:
#   AOT build:  ~5-20 min  (offline pvc device code compilation + fat link)
#   JIT build:  ~15-30 sec (SPIR-V only; native code compiled at first GPU launch)
#
# The binary produced here uses JIT — the first kernel launch on the GPU node
# will be ~1-2 sec slower while oneAPI compiles to native PVC code, but all
# subsequent launches are fast (cached).
#
# Run from: /home/zippy/src/warpx/build_installShared
# Output:   /tmp/gpu_rng_test_jit

set -euo pipefail

BUILD_DIR="$(cd "$(dirname "$0")/../../../build_installShared" && pwd)"
SRC="$(cd "$(dirname "$0")" && pwd)/main.cpp"
OUT="/home/zippy/src/warpx/Examples/Tests/gpu_rng_test/gpu_rng_test_jit"

AMREX_SRC="${BUILD_DIR}/_deps/fetchedamrex-src/Src"
AMREX_BUILD="${BUILD_DIR}/_deps/fetchedamrex-build"
MPI_PREFIX="/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/mpich-5.0.0.aurora_test.3c70a61-hlkigtk"
ICPX="/opt/aurora/26.26.0/oneapi/compiler/latest/bin/icpx"

echo "Building ${OUT} (JIT, no AOT)..."
"${ICPX}" \
    -DAMREX_SPACEDIM=3 \
    -I"${AMREX_SRC}/Base" \
    -I"${AMREX_SRC}/Base/Parser" \
    -I"${AMREX_SRC}/Boundary" \
    -I"${AMREX_SRC}/AmrCore" \
    -I"${AMREX_SRC}/EB" \
    -I"${AMREX_SRC}/LinearSolvers" \
    -I"${AMREX_SRC}/LinearSolvers/MLMG" \
    -I"${AMREX_SRC}/LinearSolvers/OpenBC" \
    -I"${AMREX_SRC}/Particle" \
    -I"${AMREX_BUILD}" \
    -isystem "${MPI_PREFIX}/include" \
    -O0 -g -std=c++17 -pthread \
    -mlong-double-64 -Xclang -mlong-double-64 \
    -fsycl \
    -Wl,-rpath,"${BUILD_DIR}/lib" \
    -L"${BUILD_DIR}/lib" -lamrex_3d \
    -Wl,-rpath,"${MPI_PREFIX}/lib" \
    "${MPI_PREFIX}/lib/libmpicxx.so" \
    "${MPI_PREFIX}/lib/libmpi.so" \
    "${SRC}" \
    -o "${OUT}"

echo "Done: ${OUT}"
echo "Run on GPU node with: mpiexec --np 1 ... ${OUT} amrex.verbose=0"
