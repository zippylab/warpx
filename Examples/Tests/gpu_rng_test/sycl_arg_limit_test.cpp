// sycl_arg_limit_test.cpp
//
// Standalone SYCL reproducer for the Intel PVC 2048-byte kernel argument limit.
// No AMReX or MPI required — links only against the SYCL/oneAPI runtime.
//
// The Intel PVC hardware can only pass 2048 bytes of kernel arguments.
// Arguments whose byte offset from the start of the argument block exceeds 2048
// are silently read as garbage (zero in JIT mode; uninitialized in AOT mode).
// IGC_OverrideOCLMaxParamSize suppresses the compile-time error check but does
// NOT fix the hardware limitation.
//
// This program runs two tests:
//   BELOW: 8 blocks x  8 doubles (512 B user arrays) — all correct
//   ABOVE: 8 blocks x 64 doubles (4096 B user arrays) — blocks 4-7 corrupted
//
// ============================================================
// Build — JIT (fast, kernels compiled at first GPU launch):
//   source /home/zippy/sunspot_warpx_debugging.profile
//   export IGC_OverrideOCLMaxParamSize=8192
//   icpx -fsycl -O2 -std=c++17 sycl_arg_limit_test.cpp -o sycl_arg_limit_test_jit
//
// Build — AOT (slow, kernels compiled offline for PVC):
//   source /home/zippy/sunspot_warpx_debugging.profile
//   export IGC_OverrideOCLMaxParamSize=8192
//   icpx -fsycl -fsycl-targets=spir64_gen \
//        -Xsycl-target-backend "-device pvc" \
//        -flink-huge-device-code \
//        -ftarget-register-alloc-mode=pvc:large \
//        --offload-compress \
//        -O2 -std=c++17 sycl_arg_limit_test.cpp -o sycl_arg_limit_test_aot
//
// Run (must set override before launch for JIT; for AOT override is baked in):
//   ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
//   IGC_OverrideOCLMaxParamSize=8192 \
//   ZE_AFFINITY_MASK=0 \
//   ./sycl_arg_limit_test_jit
// ============================================================

#include <sycl/sycl.hpp>
#include <array>
#include <cstdio>
#include <cmath>

// A block of N doubles captured by value inside a SYCL kernel lambda.
template <int N>
using Block = std::array<double, N>;

// Launch a kernel that captures 8 Block<N_PER_BLOCK> arrays plus an output
// pointer, all by value.  Writes b_k[0] to out[k].  Sentinel: b_k[0] = k*10000.
// Returns true if all 8 values are read correctly.
template <int N_PER_BLOCK>
bool run_capture_test(sycl::queue& q, const char* label)
{
    constexpr int N_BLOCKS = 8;
    constexpr int user_array_bytes = N_BLOCKS * N_PER_BLOCK * sizeof(double);
    constexpr int user_capture_bytes = sizeof(void*) + user_array_bytes;

    printf("\n--- %s ---\n", label);
    printf("  User captures: 1 ptr (8 B) + %d blocks x %d doubles x 8 B = %d B\n",
           N_BLOCKS, N_PER_BLOCK, user_capture_bytes);

    // Initialise blocks: b_k[j] = k*10000 + j  (sentinel: b_k[0] = k*10000)
    Block<N_PER_BLOCK> b0, b1, b2, b3, b4, b5, b6, b7;
    auto init = [](auto& b, int k) {
        for (int j = 0; j < (int)b.size(); j++)
            b[j] = double(k * 10000 + j);
    };
    init(b0,0); init(b1,1); init(b2,2); init(b3,3);
    init(b4,4); init(b5,5); init(b6,6); init(b7,7);

    // Output buffer: USM shared memory, initialised to sentinel -999
    double* out = sycl::malloc_shared<double>(N_BLOCKS, q);
    for (int k = 0; k < N_BLOCKS; k++) out[k] = -999.0;

    // Kernel: all 8 blocks + out pointer captured by value via [=]
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) {
            out[0] = b0[0];  out[1] = b1[0];
            out[2] = b2[0];  out[3] = b3[0];
            out[4] = b4[0];  out[5] = b5[0];
            out[6] = b6[0];  out[7] = b7[0];
        });
    }).wait();

    bool all_ok = true;
    for (int k = 0; k < N_BLOCKS; k++) {
        double expected = k * 10000.0;
        // Approximate user-lambda offset of b_k[0] (ignoring runtime overhead):
        int approx_offset = sizeof(void*) + k * N_PER_BLOCK * (int)sizeof(double);
        bool ok = (std::abs(out[k] - expected) < 0.5);
        printf("  b%d[0]  user_offset~%4dB  expected=%6.0f  got=%6.0f  %s\n",
               k, approx_offset, expected, out[k],
               ok ? "OK" : "*** CORRUPTED ***");
        if (!ok) all_ok = false;
    }
    printf("  Result: %s\n", all_ok ? "PASS" : "FAIL");

    sycl::free(out, q);
    return all_ok;
}

int main()
{
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();
    printf("Device:              %s\n",
           dev.get_info<sycl::info::device::name>().c_str());
    printf("max_parameter_size:  %zu bytes  (hardware-reported argument block limit)\n",
           dev.get_info<sycl::info::device::max_parameter_size>());

    printf("\n=== Intel PVC kernel argument size limit reproducer ===\n");
    printf("Sentinel: b_k[0] = k*10000.  Hardware limit: 2048 bytes.\n");
    printf("Captures beyond that offset are silently read as garbage.\n");

    // BELOW threshold: 8 x 8 doubles = 512 B arrays + 8 B ptr = 520 B user captures
    bool below = run_capture_test<8>(q, "BELOW threshold (8 doubles/block, ~520 B user captures)");

    // ABOVE threshold: 8 x 64 doubles = 4096 B arrays + 8 B ptr = 4104 B user captures
    // b4[0] is at user_offset ~2056 B; including runtime overhead, total > 2048 B
    bool above = run_capture_test<64>(q, "ABOVE threshold (64 doubles/block, ~4104 B user captures)");

    printf("\n=== Summary ===\n");
    printf("BELOW threshold: %s\n", below ? "PASS" : "FAIL (unexpected)");
    printf("ABOVE threshold: %s\n", above ? "PASS (unexpected)" : "FAIL (expected — hardware limit confirmed)");

    if (below && !above)
        printf("Conclusion: Intel PVC 2048-byte hardware argument limit confirmed.\n"
               "Fix required: pass large captures via USM device pointer (8 B),\n"
               "not by value.\n");
    else if (below && above)
        printf("Conclusion: Both pass — either the runtime handles spilling on this\n"
               "hardware revision, or user captures fit within 2048 B.\n");
    else
        printf("Conclusion: Unexpected result — check device and build flags.\n");

    return (below && !above) ? 0 : 1;
}
