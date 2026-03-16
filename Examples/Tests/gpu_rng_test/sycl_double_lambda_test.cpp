// sycl_double_lambda_test.cpp
//
// Tests whether AMReX's double-lambda pattern (outer lambda capturing inner lambda)
// is what causes the 2048-byte argument limit to bite.
//
// AMReX ParallelFor(box, f) submits a kernel like:
//   q.submit([&](sycl::handler& h) {
//       h.parallel_for(range, [=, inner_f](sycl::item<3> it) {   // outer lambda
//           inner_f(ix, iy, iz);                                   // calls inner
//       });
//   });
//
// The outer lambda captures the inner lambda by value.  When inner_f captures
// large arrays, the outer lambda's argument block = sizeof(inner_f) + overhead.
// If sizeof(inner_f) > ~2000 bytes, the hardware 2048-byte limit is hit.
//
// This program repeats the ABOVE-threshold test (8 x 64 doubles) using the
// double-lambda pattern to see if Level Zero still handles it via indirection.
//
// Build:
//   source /home/zippy/sunspot_warpx_debugging.profile
//   export IGC_OverrideOCLMaxParamSize=8192
//   icpx -fsycl -O0 -g -std=c++17 sycl_double_lambda_test.cpp -o sycl_double_lambda_test
//   icpx -fsycl -O2    -std=c++17 sycl_double_lambda_test.cpp -o sycl_double_lambda_test_O2
//
// Run:
//   ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE IGC_OverrideOCLMaxParamSize=8192 \
//   ZE_AFFINITY_MASK=0 ./sycl_double_lambda_test

#include <sycl/sycl.hpp>
#include <array>
#include <cstdio>
#include <cmath>

template <int N>
using Block = std::array<double, N>;

// ============================================================
// Test A: Direct lambda (like sycl_arg_limit_test) — should PASS
// ============================================================
template <int N_PER_BLOCK>
bool test_direct_lambda(sycl::queue& q, const char* label)
{
    constexpr int N_BLOCKS = 8;
    constexpr int user_capture_bytes =
        sizeof(void*) + N_BLOCKS * N_PER_BLOCK * sizeof(double);

    printf("\n--- %s (direct lambda) ---\n", label);
    printf("  sizeof captures: 1 ptr (8B) + %d blocks x %d doubles x 8B = %d B\n",
           N_BLOCKS, N_PER_BLOCK, user_capture_bytes);

    Block<N_PER_BLOCK> b0, b1, b2, b3, b4, b5, b6, b7;
    auto init = [](auto& b, int k) {
        for (int j = 0; j < (int)b.size(); j++) b[j] = double(k * 10000 + j);
    };
    init(b0,0); init(b1,1); init(b2,2); init(b3,3);
    init(b4,4); init(b5,5); init(b6,6); init(b7,7);

    double* out = sycl::malloc_shared<double>(N_BLOCKS, q);
    for (int k = 0; k < N_BLOCKS; k++) out[k] = -999.0;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) {
            out[0]=b0[0]; out[1]=b1[0]; out[2]=b2[0]; out[3]=b3[0];
            out[4]=b4[0]; out[5]=b5[0]; out[6]=b6[0]; out[7]=b7[0];
        });
    }).wait();

    bool all_ok = true;
    for (int k = 0; k < N_BLOCKS; k++) {
        double expected = k * 10000.0;
        int approx_offset = sizeof(void*) + k * N_PER_BLOCK * (int)sizeof(double);
        bool ok = (std::abs(out[k] - expected) < 0.5);
        printf("  b%d[0]  user_offset~%4dB  expected=%6.0f  got=%6.0f  %s\n",
               k, approx_offset, expected, out[k], ok ? "OK" : "*** CORRUPTED ***");
        if (!ok) all_ok = false;
    }
    printf("  Result: %s\n", all_ok ? "PASS" : "FAIL");
    sycl::free(out, q);
    return all_ok;
}

// ============================================================
// Test B: Double-lambda pattern (AMReX ParallelFor style)
// Outer lambda captures inner lambda by value.
// ============================================================
template <int N_PER_BLOCK>
bool test_double_lambda(sycl::queue& q, const char* label)
{
    constexpr int N_BLOCKS = 8;
    constexpr int user_capture_bytes =
        sizeof(void*) + N_BLOCKS * N_PER_BLOCK * sizeof(double);

    printf("\n--- %s (double-lambda / AMReX ParallelFor pattern) ---\n", label);
    printf("  inner lambda captures: 1 ptr (8B) + %d blocks x %d doubles x 8B = %d B\n",
           N_BLOCKS, N_PER_BLOCK, user_capture_bytes);

    Block<N_PER_BLOCK> b0, b1, b2, b3, b4, b5, b6, b7;
    auto init = [](auto& b, int k) {
        for (int j = 0; j < (int)b.size(); j++) b[j] = double(k * 10000 + j);
    };
    init(b0,0); init(b1,1); init(b2,2); init(b3,3);
    init(b4,4); init(b5,5); init(b6,6); init(b7,7);

    double* out = sycl::malloc_shared<double>(N_BLOCKS, q);
    for (int k = 0; k < N_BLOCKS; k++) out[k] = -999.0;

    // AMReX-style: define the user function (inner lambda), then submit
    // via an outer lambda that captures and calls it.
    auto inner_f = [=](int /*unused_index*/) {
        out[0]=b0[0]; out[1]=b1[0]; out[2]=b2[0]; out[3]=b3[0];
        out[4]=b4[0]; out[5]=b5[0]; out[6]=b6[0]; out[7]=b7[0];
    };

    printf("  sizeof(inner_f) = %zu B\n", sizeof(inner_f));

    q.submit([&](sycl::handler& h) {
        // Outer lambda captures inner_f by value (the AMReX pattern).
        h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> id) {
            inner_f((int)id[0]);
        });
    }).wait();

    bool all_ok = true;
    for (int k = 0; k < N_BLOCKS; k++) {
        double expected = k * 10000.0;
        int approx_offset = sizeof(void*) + k * N_PER_BLOCK * (int)sizeof(double);
        bool ok = (std::abs(out[k] - expected) < 0.5);
        printf("  b%d[0]  user_offset~%4dB  expected=%6.0f  got=%6.0f  %s\n",
               k, approx_offset, expected, out[k], ok ? "OK" : "*** CORRUPTED ***");
        if (!ok) all_ok = false;
    }
    printf("  Result: %s\n", all_ok ? "PASS" : "FAIL");
    sycl::free(out, q);
    return all_ok;
}

// ============================================================
// Test C: AMReX-style with pinned host memory (PinnedVector pattern)
// Uses malloc_host instead of malloc_shared for the output buffer.
// ============================================================
template <int N_PER_BLOCK>
bool test_pinned_host_output(sycl::queue& q, const char* label)
{
    constexpr int N_BLOCKS = 8;
    constexpr int user_capture_bytes =
        sizeof(void*) + N_BLOCKS * N_PER_BLOCK * sizeof(double);

    printf("\n--- %s (pinned host memory output / double-lambda) ---\n", label);
    printf("  inner lambda captures: 1 ptr (8B) + %d blocks x %d doubles x 8B = %d B\n",
           N_BLOCKS, N_PER_BLOCK, user_capture_bytes);

    Block<N_PER_BLOCK> b0, b1, b2, b3, b4, b5, b6, b7;
    auto init = [](auto& b, int k) {
        for (int j = 0; j < (int)b.size(); j++) b[j] = double(k * 10000 + j);
    };
    init(b0,0); init(b1,1); init(b2,2); init(b3,3);
    init(b4,4); init(b5,5); init(b6,6); init(b7,7);

    // Use malloc_host (pinned, not shared USM) — closer to AMReX PinnedVector
    double* out = sycl::malloc_host<double>(N_BLOCKS, q);
    for (int k = 0; k < N_BLOCKS; k++) out[k] = -999.0;

    auto inner_f = [=](int) {
        out[0]=b0[0]; out[1]=b1[0]; out[2]=b2[0]; out[3]=b3[0];
        out[4]=b4[0]; out[5]=b5[0]; out[6]=b6[0]; out[7]=b7[0];
    };

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> id) {
            inner_f((int)id[0]);
        });
    }).wait();

    bool all_ok = true;
    for (int k = 0; k < N_BLOCKS; k++) {
        double expected = k * 10000.0;
        int approx_offset = sizeof(void*) + k * N_PER_BLOCK * (int)sizeof(double);
        bool ok = (std::abs(out[k] - expected) < 0.5);
        printf("  b%d[0]  user_offset~%4dB  expected=%6.0f  got=%6.0f  %s\n",
               k, approx_offset, expected, out[k], ok ? "OK" : "*** CORRUPTED ***");
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
    printf("Device:             %s\n",
           dev.get_info<sycl::info::device::name>().c_str());
    printf("max_parameter_size: %zu bytes\n",
           dev.get_info<sycl::info::device::max_parameter_size>());

    printf("\n=== Intel PVC double-lambda and pinned-memory reproducer ===\n");
    printf("ABOVE threshold: 8 blocks x 64 doubles = 4096 B arrays + 8 B ptr = 4104 B\n");
    printf("Expected: direct lambda PASSES; double-lambda may FAIL (AMReX-pattern).\n");

    // Baseline: direct lambda, same as sycl_arg_limit_test
    bool direct_below  = test_direct_lambda<8>(q,  "BELOW ~520B");
    bool direct_above  = test_direct_lambda<64>(q, "ABOVE ~4104B");

    // AMReX pattern: double-lambda
    bool double_below  = test_double_lambda<8>(q,  "BELOW ~520B");
    bool double_above  = test_double_lambda<64>(q, "ABOVE ~4104B");

    // Pinned host output + double-lambda
    bool pinned_below  = test_pinned_host_output<8>(q,  "BELOW ~520B");
    bool pinned_above  = test_pinned_host_output<64>(q, "ABOVE ~4104B");

    printf("\n=== Summary ===\n");
    printf("Direct  BELOW: %s\n", direct_below ? "PASS" : "FAIL");
    printf("Direct  ABOVE: %s\n", direct_above ? "PASS" : "FAIL");
    printf("Double  BELOW: %s\n", double_below ? "PASS" : "FAIL");
    printf("Double  ABOVE: %s\n", double_above ? "PASS" : "FAIL");
    printf("Pinned  BELOW: %s\n", pinned_below ? "PASS" : "FAIL");
    printf("Pinned  ABOVE: %s\n", pinned_above ? "PASS" : "FAIL");

    if (!double_above && direct_above)
        printf("\nConclusion: double-lambda pattern triggers 2048B limit.\n"
               "AMReX ParallelFor is the root cause.\n");
    else if (double_above && direct_above)
        printf("\nConclusion: both pass — Level Zero indirection works for both patterns.\n"
               "Look elsewhere for AMReX's failure (build flags, MKL, queue config).\n");
    else
        printf("\nConclusion: unexpected — check build and run environment.\n");

    return 0;
}
