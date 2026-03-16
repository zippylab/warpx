/**
 * gpu_rng_test: Standalone diagnostic for AMReX ParallelForRNG on SYCL GPU.
 *
 * Tests whether the GPU random engine is functional by running two variants
 * of ParallelForRNG — the 1D-range version and the 3D-Box version — which
 * is the same call path used in WarpX AddPlasma (particle injection).
 *
 * Returns 0 (PASS) or 1 (FAIL) for CTest integration.
 *
 * Expected output if working:
 *   [Test 1] ... mean ≈ 0.5, n_zero = 0 ... PASS
 *   [Test 2] ... n_zero = 0            ... PASS
 *   [Test 3] ... n_written = N         ... PASS
 *
 * If the GPU kernel silently fails (as suspected on Aurora), all values will
 * be zero and every test will print FAIL, and the process exits with code 1.
 */

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_Random.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <cstdint>
#include <vector>

// Device printf for kernel debug — works inside SYCL device kernels.
// __SYCL_DEVICE_ONLY__ is set by the compiler when compiling device code;
// __SYCL_LANGUAGE_VERSION is defined by any SYCL-aware compiler (host or device).
#if defined(__SYCL_DEVICE_ONLY__) || defined(__SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#define DEVICE_PRINTF(...) sycl::ext::oneapi::experimental::printf(__VA_ARGS__)
#else
#define DEVICE_PRINTF(...) ((void)0)
#endif

using namespace amrex;

// ---------------------------------------------------------------------------
// Test 1: 1D range ParallelForRNG — same as AMReX upstream test
// ---------------------------------------------------------------------------
bool test_range_rng ()
{
    amrex::Print() << "\n[Test 1] ParallelForRNG over 1D range (N=10000)\n";

    const int N = 10000;
    Gpu::DeviceVector<Real> vals(N, Real(0.0));
    Real* p = vals.dataPtr();

    amrex::ParallelForRNG(N,
        [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
        {
            p[i] = amrex::Random(engine);
            if (i == 0) {
                DEVICE_PRINTF("[GPU T1] i=0 val=%.6f\n", (double)p[0]);
            }
        });
    Gpu::streamSynchronize();

    std::vector<Real> h(N);
    Gpu::copyAsync(Gpu::deviceToHost, vals.begin(), vals.end(), h.begin());
    Gpu::streamSynchronize();

    int n_zero = 0, n_bad = 0;
    Real sum = 0.0;
    for (int i = 0; i < N; ++i) {
        if (h[i] == Real(0.0)) ++n_zero;
        if (h[i] < Real(0.0) || h[i] > Real(1.0)) ++n_bad;
        sum += h[i];
    }
    Real mean = sum / N;

    amrex::Print() << "  mean=" << mean
                   << "  n_zero=" << n_zero
                   << "  n_bad(out-of-range)=" << n_bad << "\n";

    if (n_zero == N) {
        amrex::Print() << "  FAIL: all values are zero — kernel did not run "
                          "or RNG descriptor is broken\n";
        return false;
    }
    if (n_bad > 0) {
        amrex::Print() << "  FAIL: " << n_bad << " values outside [0,1]\n";
        return false;
    }
    if (std::abs(mean - 0.5) > 0.05) {
        amrex::Print() << "  FAIL: mean=" << mean << " too far from 0.5\n";
        return false;
    }
    amrex::Print() << "  PASS\n";
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: 3D Box ParallelForRNG — the variant called inside AddPlasma
// ---------------------------------------------------------------------------
bool test_box_rng ()
{
    amrex::Print() << "\n[Test 2] ParallelForRNG over 3D Box (8x8x8)\n";

    Box box(IntVect(0,0,0), IntVect(7,7,7));
    const int ncells = static_cast<int>(box.numPts());  // 512

    // Initialise all to zero; the GPU kernel should overwrite with 1
    Gpu::DeviceVector<int> marker(ncells, 0);
    int* pm = marker.dataPtr();

    const int nx = box.length(0);
    const int ny = box.length(1);
    const auto lo = amrex::lbound(box);

    amrex::ParallelForRNG(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine) noexcept
        {
            Real rx = amrex::Random(engine);
            Real ry = amrex::Random(engine);
            Real rz = amrex::Random(engine);

            int idx = (i - lo.x) + nx * ((j - lo.y) + ny * (k - lo.z));

            pm[idx] = (rx >= Real(0.0) && rx <= Real(1.0) &&
                       ry >= Real(0.0) && ry <= Real(1.0) &&
                       rz >= Real(0.0) && rz <= Real(1.0)) ? 1 : -1;

            if (i == lo.x && j == lo.y && k == lo.z) {
                DEVICE_PRINTF("[GPU T2] cell(0,0,0) rx=%.6f ry=%.6f rz=%.6f marker=%d\n",
                              (double)rx, (double)ry, (double)rz, pm[idx]);
            }
        });
    Gpu::streamSynchronize();

    std::vector<int> h(ncells);
    Gpu::copyAsync(Gpu::deviceToHost, marker.begin(), marker.end(), h.begin());
    Gpu::streamSynchronize();

    int n_zero = 0, n_pass = 0, n_fail = 0;
    for (int i = 0; i < ncells; ++i) {
        if      (h[i] == 0)  ++n_zero;
        else if (h[i] == 1)  ++n_pass;
        else                  ++n_fail;
    }

    amrex::Print() << "  Box=" << box << "  ncells=" << ncells << "\n";
    amrex::Print() << "  n_unwritten(zero)=" << n_zero
                   << "  n_pass=" << n_pass
                   << "  n_rng_bad=" << n_fail << "\n";

    if (n_zero == ncells) {
        amrex::Print() << "  FAIL: no cells written — Box ParallelForRNG kernel "
                          "did not execute on GPU\n";
        return false;
    }
    if (n_zero > 0) {
        amrex::Print() << "  FAIL: " << n_zero << " of " << ncells
                       << " cells not written\n";
        return false;
    }
    amrex::Print() << "  PASS\n";
    return true;
}

// ---------------------------------------------------------------------------
// Test 5: Union + enum-switch dispatch through a raw device pointer.
//
// WarpX InjectorDensity is a tagged union:
//
//   struct InjectorDensity {
//       enum struct Type { constant, predefined, parser, fromfile } type;
//       union Object {
//           InjectorDensityConstant   constant;  // just a Real
//           InjectorDensityParser     parser;    // contains ParserExecutor<3>
//           InjectorDensityPredefined predefined;
//           InjectorDensityFromFile   fromfile;
//       } object;
//       AMREX_GPU_HOST_DEVICE Real getDensity(x,y,z) {
//           switch (type) {
//               case Type::constant: return object.constant.getDensity(...);
//               ...
//           }
//       }
//   };
//
// The copy ctor is DELETED, so WarpX uses raw htod_memcpy_async() to put it
// on device.  This test replicates that exact pattern.
// ---------------------------------------------------------------------------

// Sub-structs matching WarpX's InjectorDensityConstant and a stand-in for the
// larger union members (InjectorDensityPredefined has 6 Reals + an enum).
struct FakeConstantInj {
    amrex::Real val;
    AMREX_GPU_HOST_DEVICE amrex::Real
    getDensity (amrex::Real, amrex::Real, amrex::Real) const noexcept
    { return val; }
};

struct FakePredefinedInj {
    // Deliberately larger than FakeConstantInj (mirrors union size mismatch)
    amrex::Real a, b, c, d, e, f;
    int         profile_tag;
    AMREX_GPU_HOST_DEVICE amrex::Real
    getDensity (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    { return a + b*x + c*y + d*z; }
};

// Tagged-union replica of InjectorDensity
struct FakeInjectorDensity {
    enum struct Type { constant, predefined } type;

    union Object {
        FakeConstantInj   constant;
        FakePredefinedInj predefined;
    } object;

    // Note: WarpX InjectorDensity has copy ctor deleted (union has non-trivial members),
    // forcing htod_memcpy_async for device copies. We replicate that here via
    // The_Arena()->alloc + htod_memcpy_async below, regardless of copy ctor.
    static FakeInjectorDensity make_constant (amrex::Real rho) {
        FakeInjectorDensity d;
        d.type = Type::constant;
        d.object.constant.val = rho;
        return d;
    }

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE amrex::Real
    getDensity (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
        switch (type)
        {
        case Type::constant:
            return object.constant.getDensity(x, y, z);
        case Type::predefined:
            return object.predefined.getDensity(x, y, z);
        default:
            return amrex::Real(0);
        }
    }

private:
    FakeInjectorDensity () = default;
};

bool test_union_switch_device_ptr ()
{
    amrex::Print() << "\n[Test 5] Union+switch dispatch through raw device pointer (InjectorDensity replica)\n";

    // Build on host
    FakeInjectorDensity h_inj = FakeInjectorDensity::make_constant(Real(1e16));

    // Allocate device memory and copy via raw bytes — same as WarpX htod_memcpy_async
    auto* d_raw = static_cast<FakeInjectorDensity*>(
        amrex::The_Arena()->alloc(sizeof(FakeInjectorDensity)));
    Gpu::htod_memcpy_async(d_raw, &h_inj, sizeof(FakeInjectorDensity));
    Gpu::streamSynchronize();

    const FakeInjectorDensity* d_inj = d_raw;  // const ptr captured in lambda

    Box box(IntVect(0,0,0), IntVect(7,7,7));
    const int ncells = static_cast<int>(box.numPts());
    const int nx = box.length(0);
    const int ny = box.length(1);
    const auto lo = amrex::lbound(box);
    const int num_ppc = 4;

    Gpu::DeviceVector<Long> counts(ncells, Long(0));
    Long* pcounts = counts.dataPtr();

    amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = Real(i) + Real(0.5);
            Real y = Real(j) + Real(0.5);
            Real z = Real(k) + Real(0.5);
            Real dens = d_inj->getDensity(x, y, z);

            int idx = (i - lo.x) + nx * ((j - lo.y) + ny * (k - lo.z));
            pcounts[idx] = (dens > Real(0.0)) ? Long(num_ppc) : Long(0);

            if (i == lo.x && j == lo.y && k == lo.z) {
                DEVICE_PRINTF("[GPU T5] cell(0,0,0) dens=%.3e type_tag=%d count=%lld\n",
                              (double)dens, (int)d_inj->type, (long long)pcounts[idx]);
            }
        });
    Gpu::streamSynchronize();

    std::vector<Long> h(ncells);
    Gpu::copyAsync(Gpu::deviceToHost, counts.begin(), counts.end(), h.begin());
    Gpu::streamSynchronize();

    amrex::The_Arena()->free(d_raw);

    int n_zero = 0, n_correct = 0, n_wrong = 0;
    for (int i = 0; i < ncells; ++i) {
        if      (h[i] == 0)        ++n_zero;
        else if (h[i] == num_ppc)  ++n_correct;
        else                        ++n_wrong;
    }

    amrex::Print() << "  sizeof(FakeInjectorDensity)=" << sizeof(FakeInjectorDensity)
                   << "  sizeof(FakeConstantInj)=" << sizeof(FakeConstantInj)
                   << "  sizeof(FakePredefinedInj)=" << sizeof(FakePredefinedInj) << "\n";
    amrex::Print() << "  n_zero=" << n_zero
                   << "  n_correct=" << n_correct
                   << "  n_wrong=" << n_wrong << "\n";

    if (n_zero == ncells) {
        amrex::Print() << "  FAIL: all counts zero — union/switch dispatch failed on GPU\n";
        amrex::Print() << "  This would explain 0 particles in WarpX AddPlasma counting kernel\n";
        return false;
    }
    if (n_zero > 0 || n_wrong > 0) {
        amrex::Print() << "  FAIL: unexpected count values\n";
        return false;
    }
    amrex::Print() << "  PASS\n";
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Simulate AddPlasma's *counting* kernel (amrex::ParallelFor, no RNG).
//
// AddPlasma runs two kernels:
//   1. ParallelFor  — counts particles per cell via inj_pos->overlapsWith()
//                     and inj_rho->getDensity() called through device pointers
//   2. ParallelForRNG — fills particle data (tested in Tests 1-3)
//
// If the counting kernel returns all-zero counts, max_new_particles=0 and
// the filling kernel never allocates any particles — silently producing 0
// particles even though ParallelForRNG itself works fine.
//
// This test replicates that pattern: a plain ParallelFor over a Box, calling
// getDensity() through a device-side struct pointer, writing to a counts array.
// ---------------------------------------------------------------------------

// Minimal replica of InjectorDensity (constant profile) stored in device memory
struct SimpleInjectorDensity {
    amrex::Real density_val;

    AMREX_GPU_HOST_DEVICE amrex::Real
    getDensity (amrex::Real /*x*/, amrex::Real /*y*/, amrex::Real /*z*/) const noexcept
    {
        return density_val;
    }
};

bool test_counting_kernel ()
{
    amrex::Print() << "\n[Test 4] AddPlasma counting kernel (ParallelFor + device struct ptr)\n";

    // Allocate the injector on device — mirrors PlasmaInjector::getInjectorDensity()
    // which returns a device pointer set up with htod_memcpy.
    Gpu::DeviceVector<SimpleInjectorDensity> d_inj_vec(1);
    SimpleInjectorDensity h_inj{Real(1e16)};
    Gpu::copyAsync(Gpu::hostToDevice, &h_inj, &h_inj + 1, d_inj_vec.begin());
    Gpu::streamSynchronize();
    const SimpleInjectorDensity* inj_rho = d_inj_vec.dataPtr();

    Box box(IntVect(0,0,0), IntVect(7,7,7));
    const int ncells = static_cast<int>(box.numPts());
    const int nx = box.length(0);
    const int ny = box.length(1);
    const auto lo = amrex::lbound(box);
    const int num_ppc = 4;

    Gpu::DeviceVector<Long> counts(ncells, Long(0));
    Long* pcounts = counts.dataPtr();

    amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Evaluate density at cell centre via device struct pointer
            // (mirrors: inj_rho->getDensity(x,y,z) in AddPlasma counting kernel)
            Real x = Real(i) + Real(0.5);
            Real y = Real(j) + Real(0.5);
            Real z = Real(k) + Real(0.5);
            Real dens = inj_rho->getDensity(x, y, z);

            int idx = (i - lo.x) + nx * ((j - lo.y) + ny * (k - lo.z));
            pcounts[idx] = (dens > Real(0.0)) ? Long(num_ppc) : Long(0);

            if (i == lo.x && j == lo.y && k == lo.z) {
                DEVICE_PRINTF("[GPU T4] cell(0,0,0) dens=%.3e count=%lld\n",
                              (double)dens, (long long)pcounts[idx]);
            }
        });
    Gpu::streamSynchronize();

    std::vector<Long> h(ncells);
    Gpu::copyAsync(Gpu::deviceToHost, counts.begin(), counts.end(), h.begin());
    Gpu::streamSynchronize();

    int n_zero = 0, n_correct = 0, n_wrong = 0;
    for (int i = 0; i < ncells; ++i) {
        if      (h[i] == 0)        ++n_zero;
        else if (h[i] == num_ppc)  ++n_correct;
        else                        ++n_wrong;
    }

    amrex::Print() << "  Box=" << box << "  ncells=" << ncells
                   << "  num_ppc=" << num_ppc << "\n";
    amrex::Print() << "  n_zero(no particles)=" << n_zero
                   << "  n_correct=" << n_correct
                   << "  n_wrong=" << n_wrong << "\n";

    if (n_zero == ncells) {
        amrex::Print() << "  FAIL: all counts are zero — ParallelFor counting kernel "
                          "did not write, or getDensity() returned <= 0 on GPU\n";
        amrex::Print() << "  This would explain 0 particles: max_new_particles=0 "
                          "before ParallelForRNG even runs\n";
        return false;
    }
    if (n_zero > 0 || n_wrong > 0) {
        amrex::Print() << "  FAIL: unexpected count values\n";
        return false;
    }
    amrex::Print() << "  PASS\n";
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: Simulate AddPlasma's particle-validity write pattern.
//         Mimics: pa_idcpu[ip] = SetParticleIDandCPU(pid+ip, cpuid)
//         We check whether the GPU actually writes non-zero idcpu values.
// ---------------------------------------------------------------------------
bool test_idcpu_write ()
{
    amrex::Print() << "\n[Test 3] Simulate AddPlasma idcpu write via ParallelForRNG\n";

    Box box(IntVect(0,0,0), IntVect(3,3,3));  // 4x4x4 = 64 cells
    const int ncells = static_cast<int>(box.numPts());

    Gpu::DeviceVector<uint64_t> idcpu(ncells, uint64_t(0));
    uint64_t* pa_idcpu = idcpu.dataPtr();

    const int nx = box.length(0);
    const int ny = box.length(1);
    const auto lo = amrex::lbound(box);

    // Bit 63 set = valid particle (SetParticleIDandCPU sets this bit)
    constexpr uint64_t VALID_MASK = (uint64_t(1) << 63);

    amrex::ParallelForRNG(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine) noexcept
        {
            int ip = (i - lo.x) + nx * ((j - lo.y) + ny * (k - lo.z));
            pa_idcpu[ip] = VALID_MASK | static_cast<uint64_t>(ip + 1);

            // Draw position within cell [0,1)^3 (mirrors AddPlasma)
            Real rx = amrex::Random(engine);
            Real ry = amrex::Random(engine);
            Real rz = amrex::Random(engine);

            // Mirror ZeroInitializeAndSetNegativeID: invalidate on triple-zero
            if (rx == Real(0.0) && ry == Real(0.0) && rz == Real(0.0)) {
                pa_idcpu[ip] = uint64_t(0);
            }

            if (ip == 0) {
                DEVICE_PRINTF("[GPU T3] ip=0 rx=%.6f ry=%.6f rz=%.6f idcpu_valid=%d\n",
                              (double)rx, (double)ry, (double)rz,
                              (int)((pa_idcpu[0] & VALID_MASK) != 0));
            }
        });
    Gpu::streamSynchronize();

    std::vector<uint64_t> h(ncells);
    Gpu::copyAsync(Gpu::deviceToHost, idcpu.begin(), idcpu.end(), h.begin());
    Gpu::streamSynchronize();

    int n_valid = 0, n_invalid = 0;
    for (int i = 0; i < ncells; ++i) {
        if (h[i] & VALID_MASK) ++n_valid;
        else                    ++n_invalid;
    }

    amrex::Print() << "  ncells=" << ncells
                   << "  n_valid=" << n_valid
                   << "  n_invalid(zero-idcpu)=" << n_invalid << "\n";

    if (n_valid == 0) {
        amrex::Print() << "  FAIL: all idcpu values are zero — GPU kernel did "
                          "not write valid particle IDs\n";
        amrex::Print() << "  This reproduces the WarpX 'Initial rhs = 0' bug.\n";
        return false;
    }
    amrex::Print() << "  PASS\n";
    return true;
}

// ---------------------------------------------------------------------------
// Test 6: GPU kernel argument block size limit probe.
//
// Intel PVC GPU has a hardware limit of 2048 bytes for the kernel argument
// block.  For a SYCL ParallelForRNG kernel the argument block contains:
//   (a) the MKL engine_acc (oneAPI RNG engine accessor — size unknown to user)
//   (b) an AMReX BoxIndexer
//   (c) the user lambda capture block (all captures by value)
//
// If (a)+(b)+(c) > 2048 bytes, captures at offsets >= 2048 are read as
// garbage on the hardware, even though the compiler accepts the kernel
// when IGC_OverrideOCLMaxParamSize=4096 is set.
//
// This test probes the limit by capturing N_BLOCKS arrays of N_PER_BLOCK
// doubles each (total = N_BLOCKS * N_PER_BLOCK * 8 bytes of user lambda
// data).  Each array[0] is initialised to a unique sentinel.  The GPU
// reads each array[0] and we compare to the sentinel.
//
// We run the SAME capture block under both ParallelFor (no engine_acc)
// and ParallelForRNG (with engine_acc).  The difference in failure point
// reveals the engine_acc + indexer contribution to the argument block.
// ---------------------------------------------------------------------------

// One probe pass: captures a0..a7 by value, reads a_k[0] into pr[k].
// template parameter USE_RNG selects ParallelForRNG vs ParallelFor.
// Returns true if all sentinels are intact.
template <int N_PER_BLOCK>
bool run_capture_probe (bool use_rng,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a0,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a1,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a2,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a3,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a4,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a5,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a6,
                        amrex::GpuArray<amrex::Real,N_PER_BLOCK> a7)
{
    constexpr int N_BLOCKS = 8;
    amrex::Gpu::PinnedVector<amrex::Real> r(N_BLOCKS, amrex::Real(-999.0));
    auto* pr = r.data();

    amrex::Box box(amrex::IntVect(0,0,0), amrex::IntVect(0,0,0));

    if (use_rng) {
        amrex::ParallelForRNG(box,
            [=] AMREX_GPU_DEVICE (int, int, int, amrex::RandomEngine const&) noexcept
            {
                pr[0] = a0[0];  pr[1] = a1[0];  pr[2] = a2[0];  pr[3] = a3[0];
                pr[4] = a4[0];  pr[5] = a5[0];  pr[6] = a6[0];  pr[7] = a7[0];
            });
    } else {
        amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int, int, int) noexcept
            {
                pr[0] = a0[0];  pr[1] = a1[0];  pr[2] = a2[0];  pr[3] = a3[0];
                pr[4] = a4[0];  pr[5] = a5[0];  pr[6] = a6[0];  pr[7] = a7[0];
            });
    }
    amrex::Gpu::streamSynchronize();

    const int stride = N_PER_BLOCK * (int)sizeof(amrex::Real);
    bool ok = true;
    for (int k = 0; k < N_BLOCKS; ++k) {
        amrex::Real expected = amrex::Real(k * 10000);
        bool match = (r[k] == expected);
        amrex::Print() << "    a" << k << "[0]"
                       << "  user_offset≈" << (k * stride) << "B"
                       << "  expected=" << expected
                       << "  got=" << r[k]
                       << (match ? "  OK" : "  *** CORRUPTED ***") << "\n";
        if (!match) ok = false;
    }
    return ok;
}

bool test_capture_size_limit ()
{
    // N_PER_BLOCK doubles per array, 8 arrays.
    // User-lambda size = 8 * N_PER_BLOCK * sizeof(Real) + 8 bytes (pr ptr).
    constexpr int N_PER_BLOCK = 64;  // 64 * 8 = 512 bytes per block; 8 blocks = 4096 bytes total

    amrex::Print() << "\n[Test 6] Kernel argument block size limit probe\n";
    amrex::Print() << "  8 blocks x " << N_PER_BLOCK << " doubles = "
                   << 8 * N_PER_BLOCK * (int)sizeof(amrex::Real)
                   << " bytes of user-lambda captures\n";
    amrex::Print() << "  (plus 8-byte pr pointer; plus engine_acc for RNG variant)\n";

    // Initialise sentinels: a_k[0] = k * 10000
    amrex::GpuArray<amrex::Real, N_PER_BLOCK> a0,a1,a2,a3,a4,a5,a6,a7;
    amrex::Real* const ptrs[8] = {a0.data(),a1.data(),a2.data(),a3.data(),
                                   a4.data(),a5.data(),a6.data(),a7.data()};
    for (int k = 0; k < 8; ++k)
        for (int i = 0; i < N_PER_BLOCK; ++i)
            ptrs[k][i] = amrex::Real(k * 10000 + i);

    amrex::Print() << "\n  --- ParallelFor (no engine_acc) ---\n";
    bool ok_pf = run_capture_probe<N_PER_BLOCK>(false, a0,a1,a2,a3,a4,a5,a6,a7);

    amrex::Print() << "\n  --- ParallelForRNG (+ engine_acc) ---\n";
    bool ok_rng = run_capture_probe<N_PER_BLOCK>(true,  a0,a1,a2,a3,a4,a5,a6,a7);

    amrex::Print() << "\n  ParallelFor result:    " << (ok_pf  ? "PASS" : "FAIL (overflow)") << "\n";
    amrex::Print() << "  ParallelForRNG result: " << (ok_rng ? "PASS" : "FAIL (overflow)") << "\n";
    if (ok_pf && !ok_rng) {
        amrex::Print() << "  --> engine_acc pushes captures past 2048-byte limit\n";
        amrex::Print() << "  --> This is the root cause of WarpX AddPlasma 0-particle bug\n";
    } else if (!ok_pf) {
        amrex::Print() << "  --> User lambda alone exceeds limit (engine_acc not needed to trigger)\n";
    } else {
        amrex::Print() << "  --> Both variants intact at this capture size\n";
    }

    // Overall: report both but don't fail the suite — this is diagnostic.
    // Return true so the test suite can continue.
    return true;
}

// ---------------------------------------------------------------------------
int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    bool ok = true;
    {
        amrex::Print() << "\n========================================\n";
        amrex::Print() << "  gpu_rng_test: AMReX ParallelForRNG\n";
        amrex::Print() << "========================================\n";

#ifdef AMREX_USE_SYCL
        amrex::Print() << "\n[Sizes] sizeof(amrex::sycl_rng_acc)=" << sizeof(amrex::sycl_rng_acc)
                       << "  (engine_acc contribution to outer kernel arg block)\n";
        // Outer ParallelForRNG kernel arg block = engine_acc + BoxIndexerND<3> + pf(ptr) + user_lambda
        // BoxIndexerND<3>: npts(8) + fdm[2](48) + lo(12) = ~72B (padded)
        // If engine_acc + ~80B + 8B + sizeof(fill_kernel=424B) > 2048B  =>  crash at VA 0x0
        amrex::Print() << "  est. outer-lambda overhead (sans user-f): "
                       << sizeof(amrex::sycl_rng_acc) + 72 + 8
                       << "B  (limit 2048B)\n";
#endif

        ok &= test_range_rng();
        ok &= test_box_rng();
        ok &= test_idcpu_write();
        ok &= test_counting_kernel();
        ok &= test_union_switch_device_ptr();
        ok &= test_capture_size_limit();

        amrex::Print() << "\n----------------------------------------\n";
        amrex::Print() << "  Overall: " << (ok ? "PASS" : "FAIL") << "\n";
        amrex::Print() << "----------------------------------------\n\n";
    }

    amrex::Finalize();
    return ok ? 0 : 1;
}
