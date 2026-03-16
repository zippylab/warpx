// sycl_usm_acc_ptr_bug.cpp
//
// Standalone SYCL reproducer for a DPC++ compiler bug on Intel PVC where
// device-USM pointer fields inside a nested functor are silently zeroed at
// kernel launch when the outer SYCL kernel lambda also captures a
// sycl::accessor.
//
// BACKGROUND
// ----------
// In WarpX, AddParticles.cpp uses amrex::ParallelForRNG(box, fill_kernel).
// Internally, AMReX submits a SYCL kernel of the form:
//
//   auto engine_acc = rng_buf.get_access<read_write>(h);   // SYCL accessor
//   h.parallel_for(range, [=](sycl::id<3> id, auto /*rng*/) {
//       fill_kernel(ix, iy, iz, engine);   // inner functor captured by [=]
//   });
//
// fill_kernel is a compiler-generated lambda struct whose capture list
// includes device-USM pointers (e.g. inj_pos*, inj_mom*, inj_rho*).
// When engine_acc (a SYCL accessor) is also present in the outer lambda's
// capture list, DPC++ silently zeros those device pointer fields at kernel
// launch time.  The pointers are valid on the host immediately before
// submission, but are 0x0 inside the kernel, causing a GPU segfault on
// the first dereference.
//
// This file isolates the trigger: outer lambda captures (a) a struct with
// a device-USM pointer field and (b) a sycl::accessor.
//
// OBSERVATION STRATEGY
// --------------------
// We store the raw pointer value (as uintptr_t) into a second USM allocation
// from inside the kernel, then read it back on the host WITHOUT dereferencing
// the possibly-null pointer.  This lets us see whether the pointer is zeroed
// without triggering a GPU segfault.
//
// BUILD
// -----
//   source /home/zippy/sunspot_warpx_debugging.profile
//   icpx -fsycl -O2 -std=c++17 sycl_usm_acc_ptr_bug.cpp -o sycl_usm_acc_ptr_bug_O2
//   icpx -fsycl -O0 -g  -std=c++17 sycl_usm_acc_ptr_bug.cpp -o sycl_usm_acc_ptr_bug_O0
//
// RUN
// ---
//   ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0 ./sycl_usm_acc_ptr_bug_O2
//
// EXPECTED (buggy compiler)
//   Test A (no accessor):   ptr seen in kernel = 0x<valid>   [OK]
//   Test B (with accessor): ptr seen in kernel = 0x0          [BUG]
//
// EXPECTED (fixed compiler or workaround applied)
//   Both tests: ptr seen in kernel = 0x<valid>               [OK]
//
// WORKAROUND (used in WarpX AddParticles.cpp)
// -------------------------------------------
//   Copy fill_kernel to device-arena USM via htod_memcpy_async, then
//   capture only an 8-byte device pointer in the outer lambda.  This
//   bypasses the SYCL argument-passing path entirely for fill_kernel.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// The functor that represents WarpX's fill_kernel.
// It holds one or more device-USM pointer fields (the key ingredient).
// In the real code these point to InjectorPosition, InjectorMomentum, etc.
// ---------------------------------------------------------------------------
struct FillKernel {
    double* device_data;   // device-USM pointer — this is what gets zeroed

    // In the real code the operator() dereferences device_data.
    // Here we just store the address into an observation buffer.
    void observe_ptr(uintptr_t* obs) const {
        *obs = reinterpret_cast<uintptr_t>(device_data);
    }
};

// ---------------------------------------------------------------------------
// Helper: returns true if the pointer seen inside the kernel matches expected.
// ---------------------------------------------------------------------------
static bool check(const char* label, uintptr_t seen, uintptr_t expected)
{
    bool ok = (seen == expected);
    printf("  %-36s  host addr = 0x%012lx  "
           "kernel addr = 0x%012lx  %s\n",
           label,
           (unsigned long)expected,
           (unsigned long)seen,
           ok ? "OK" : "*** BUG: pointer zeroed! ***");
    return ok;
}

int main()
{
    sycl::queue q{sycl::gpu_selector_v};
    printf("Device: %s\n\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());

    // Allocate device memory that FillKernel will point to.
    const int N = 64;
    double* d_data = sycl::malloc_device<double>(N, q);

    // Write a sentinel so we can later verify the memory is reachable.
    q.fill(d_data, 1.0, N).wait();

    // Where the kernel will write the captured pointer value.
    uintptr_t* d_obs = sycl::malloc_device<uintptr_t>(1, q);
    q.fill(d_obs, (uintptr_t)0, 1).wait();

    uintptr_t expected_addr = reinterpret_cast<uintptr_t>(d_data);
    printf("Expected device_data address: 0x%012lx\n\n", (unsigned long)expected_addr);

    // -----------------------------------------------------------------------
    // TEST A: outer kernel lambda captures FillKernel only — NO accessor.
    // Should work correctly even on buggy compilers.
    // -----------------------------------------------------------------------
    {
        FillKernel fk{d_data};
        q.fill(d_obs, (uintptr_t)0, 1).wait();

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) {
                fk.observe_ptr(d_obs);
            });
        }).wait();

        uintptr_t seen = 0;
        q.memcpy(&seen, d_obs, sizeof(uintptr_t)).wait();
        check("Test A: no accessor", seen, expected_addr);
    }

    // -----------------------------------------------------------------------
    // TEST B: outer kernel lambda captures FillKernel AND a sycl::accessor.
    // On buggy DPC++/PVC: fk.device_data is 0x0 inside the kernel.
    // This is the combination that appears in AMReX ParallelForRNG.
    // -----------------------------------------------------------------------
    {
        FillKernel fk{d_data};
        q.fill(d_obs, (uintptr_t)0, 1).wait();

        // The accessor simulates AMReX's per-thread RNG engine buffer.
        sycl::buffer<int, 1> rng_buf{sycl::range<1>{N}};

        q.submit([&](sycl::handler& h) {
            // Both fk (with device-USM ptr) and rng_acc are captured by [=].
            auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> idx) {
                (void)rng_acc[idx];       // ensure accessor is actually captured
                fk.observe_ptr(d_obs);    // read device_data ptr from fk
            });
        }).wait();

        uintptr_t seen = 0;
        q.memcpy(&seen, d_obs, sizeof(uintptr_t)).wait();
        check("Test B: with sycl::accessor", seen, expected_addr);
    }

    // -----------------------------------------------------------------------
    // TEST C: same as B but with MULTIPLE device-USM pointer fields.
    // Closer to the real WarpX fill_kernel (inj_pos, inj_mom, inj_rho).
    // -----------------------------------------------------------------------
    struct MultiPtrFunctor {
        double* ptr0;
        double* ptr1;
        double* ptr2;
        void observe(uintptr_t* obs) const {
            obs[0] = reinterpret_cast<uintptr_t>(ptr0);
            obs[1] = reinterpret_cast<uintptr_t>(ptr1);
            obs[2] = reinterpret_cast<uintptr_t>(ptr2);
        }
    };

    double* d_data1 = sycl::malloc_device<double>(N, q);
    double* d_data2 = sycl::malloc_device<double>(N, q);
    uintptr_t* d_obs3 = sycl::malloc_device<uintptr_t>(3, q);

    uintptr_t exp[3] = {
        reinterpret_cast<uintptr_t>(d_data),
        reinterpret_cast<uintptr_t>(d_data1),
        reinterpret_cast<uintptr_t>(d_data2),
    };
    q.fill(d_obs3, (uintptr_t)0, 3).wait();

    {
        MultiPtrFunctor mpf{d_data, d_data1, d_data2};
        sycl::buffer<int, 1> rng_buf{sycl::range<1>{N}};

        q.submit([&](sycl::handler& h) {
            auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> idx) {
                (void)rng_acc[idx];
                mpf.observe(d_obs3);
            });
        }).wait();

        uintptr_t seen[3] = {0, 0, 0};
        q.memcpy(seen, d_obs3, 3 * sizeof(uintptr_t)).wait();

        bool all_ok = true;
        for (int i = 0; i < 3; i++) {
            bool ok = check(
                (i == 0 ? "Test C: multi-ptr, ptr0" :
                 i == 1 ? "Test C: multi-ptr, ptr1" :
                          "Test C: multi-ptr, ptr2"),
                seen[i], exp[i]);
            all_ok = all_ok && ok;
        }
    }

    // -----------------------------------------------------------------------
    // WORKAROUND TEST: copy functor to device USM, pass only the device ptr.
    // This is the fix applied in WarpX AddParticles.cpp.
    // -----------------------------------------------------------------------
    {
        FillKernel fk{d_data};
        q.fill(d_obs, (uintptr_t)0, 1).wait();

        // Copy fk to device-arena USM via memcpy (bypasses SYCL arg passing).
        FillKernel* d_fk = sycl::malloc_device<FillKernel>(1, q);
        q.memcpy(d_fk, &fk, sizeof(FillKernel)).wait();

        sycl::buffer<int, 1> rng_buf{sycl::range<1>{N}};

        q.submit([&](sycl::handler& h) {
            auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(h);
            // Only the 8-byte d_fk pointer is in the outer lambda's capture.
            const FillKernel* d_fk_ptr = d_fk;
            h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> idx) {
                (void)rng_acc[idx];
                d_fk_ptr->observe_ptr(d_obs);  // dereference the device pointer
            });
        }).wait();

        uintptr_t seen = 0;
        q.memcpy(&seen, d_obs, sizeof(uintptr_t)).wait();
        check("Workaround: functor in device USM", seen, expected_addr);

        sycl::free(d_fk, q);
    }

    sycl::free(d_data,  q);
    sycl::free(d_data1, q);
    sycl::free(d_data2, q);
    sycl::free(d_obs,   q);
    sycl::free(d_obs3,  q);
    return 0;
}
