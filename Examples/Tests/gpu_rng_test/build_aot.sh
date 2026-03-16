#!/bin/bash
# AOT build of gpu_rng_test using the same flags as the WarpX binary.
# If this fails while the JIT build passes, the AOT flags are the culprit.
#
# Key flags under test (all present in WarpX cmake build, absent in build_jit.sh):
#   -fsycl-device-code-split=off       (WarpX uses this via AMReX_CXX_FLAGS)
#   -fsycl-targets=spir64_gen          (AOT for Intel GPU)
#   -Xsycl-target-backend "-device pvc" (PVC-specific)
#   -flink-huge-device-code            (large device binary link)
#   -ftarget-register-alloc-mode=pvc:large
#   --offload-compress
#   -fsycl-max-parallel-link-jobs=20
#
# Output: Examples/Tests/gpu_rng_test/gpu_rng_test_aot

set -euo pipefail

# Use build_installShared for AMReX headers and shared library — this is what
# the original AOT binary used and is the only path that provides libamrex_3d.so
# (build_for_debugging has a static .a, which causes link-order issues with the
# AOT SYCL link step). AMReX source version is identical between the two builds.
BUILD_DIR="$(cd "$(dirname "$0")/../../../build_installShared" && pwd)"
SRC="$(cd "$(dirname "$0")" && pwd)/main.cpp"
OBJ="$(cd "$(dirname "$0")" && pwd)/main_aot.cpp.o"
OUT="$(cd "$(dirname "$0")" && pwd)/gpu_rng_test_aot"

AMREX_SRC="${BUILD_DIR}/_deps/fetchedamrex-src/Src"
AMREX_BUILD="${BUILD_DIR}/_deps/fetchedamrex-build"

# Allow AOT compilation of kernels with >2048-byte argument blocks.
# This is needed for N_PER_BLOCK=64 (4096-byte user lambda).
# The key question: does AOT compilation handle large blocks correctly
# at runtime (e.g. via argument spilling), unlike JIT which zeroes them?
export IGC_OverrideOCLMaxParamSize=8192
MPI_PREFIX="/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/mpich-5.0.0.aurora_test.3c70a61-hlkigtk"
ICPX="/opt/aurora/26.26.0/oneapi/compiler/latest/bin/icpx"

INCLUDES=(
    -DAMREX_SPACEDIM=3
    -I"${AMREX_SRC}/Base"
    -I"${AMREX_SRC}/Base/Parser"
    -I"${AMREX_SRC}/Boundary"
    -I"${AMREX_SRC}/AmrCore"
    -I"${AMREX_SRC}/EB"
    -I"${AMREX_SRC}/LinearSolvers"
    -I"${AMREX_SRC}/LinearSolvers/MLMG"
    -I"${AMREX_SRC}/LinearSolvers/OpenBC"
    -I"${AMREX_SRC}/Particle"
    -I"${AMREX_BUILD}"
    -isystem "${MPI_PREFIX}/include"
)

echo "=== Compile (AOT) ==="
"${ICPX}" \
    "${INCLUDES[@]}" \
    -O3 -g -fno-system-debug \
    -std=c++17 -pthread \
    -mlong-double-64 -Xclang -mlong-double-64 \
    -fsycl \
    -fsycl-device-code-split=off \
    -fsycl-targets=spir64_gen \
    -gline-tables-only -fdebug-info-for-profiling \
    --offload-compress \
    -c "${SRC}" -o "${OBJ}"

echo "=== Link (AOT, PVC-specific) ==="
"${ICPX}" \
    -O3 -g -fno-system-debug \
    -fsycl \
    -fsycl-device-code-split=off \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend "-device pvc" \
    -flink-huge-device-code \
    -ftarget-register-alloc-mode=pvc:large \
    --offload-compress \
    -fsycl-max-parallel-link-jobs=20 \
    -std=c++17 -pthread \
    -mlong-double-64 \
    -Wl,-rpath,"${BUILD_DIR}/lib" \
    -L"${BUILD_DIR}/lib" -lamrex_3d \
    -Wl,-rpath,"${MPI_PREFIX}/lib" \
    "${MPI_PREFIX}/lib/libmpicxx.so" \
    "${MPI_PREFIX}/lib/libmpi.so" \
    -Wl,-rpath,"/opt/aurora/26.26.0/oneapi/mkl/latest/lib" \
    -L"/opt/aurora/26.26.0/oneapi/mkl/latest/lib" \
    -lmkl_sycl_blas -lmkl_sycl_lapack -lmkl_sycl_sparse -lmkl_sycl_dft \
    -lmkl_sycl_vm -lmkl_sycl_rng -lmkl_sycl_stats -lmkl_sycl_data_fitting \
    -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
    "${OBJ}" \
    -o "${OUT}"

rm -f "${OBJ}"
echo "Done: ${OUT}"
