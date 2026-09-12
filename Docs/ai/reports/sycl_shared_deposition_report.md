# Shared-Memory Deposition on PVC

**WarpX / Aurora&ndash;PVC &middot; Engineering Report**

Diagnosing a hardware-level slowdown in particle deposition, porting a HIP/CUDA-only
kernel to SYCL, and validating the fix across three DOE GPU platforms.

| | |
|---|---|
| **Period** | 2026-09-01 &rarr; 2026-09-11 |
| **Systems** | Aurora/Sunspot (PVC) &middot; Polaris (A100) &middot; Frontier (MI250X) |
| **Outcome** | merged to fork, pushed, ready to share |

An HTML version of this report (with fuller visual formatting) is available at
[`sycl_shared_deposition_report.html`](sycl_shared_deposition_report.html) in this
same directory.

---

## Summary

WarpX's current-deposition step &mdash; the part of the particle-in-cell loop that
scatters each particle's contribution onto the field grid &mdash; was measured at
roughly **2&ndash;6&times; slower per atomic operation** on Intel's Ponte Vecchio
(PVC) GPUs than on NVIDIA A100 or AMD MI250X, for the same physics and the same
algorithm. That gap is what made deposition the dominant cost in production runs
on Aurora. The root cause turned out to be a genuine PVC hardware characteristic,
not a missing compiler flag &mdash; so the fix was not to speed up the atomic
instruction, but to use far fewer of them. AMReX/WarpX already has a faster
algorithm for this &mdash; shared-memory deposition, which batches many particles'
contributions in fast on-chip memory before one bulk write to the main grid
&mdash; but it had only ever been implemented for CUDA and HIP. It did not exist
for SYCL, the language PVC requires.

This report covers porting that kernel to SYCL, tracking down a hardware-specific
atomic-memory bug along the way, and empirically tuning and validating the result
on Aurora/Sunspot, Polaris, and Frontier before landing it as a four-file patch on
a public branch of the user's WarpX fork.

## Result

| Metric | Value |
|---|---|
| Current-deposition speedup, PVC, tuned | **3.2&times;** |
| Charge-deposition speedup, PVC, tuned | **10.1&times;** |
| Measured step-time speedup, dense case | **2.75&times;** |
| GPU architectures validated | 3 |
| Scheduler jobs across 3 facilities | 40 |
| Upstream unit tests, all passing | 7 |

Speedups are for a dense test case (640 particles/cell/species) representative
of the affected production runs, and include the particle-sorting cost the
shared path uniquely requires; see [Cross-Platform Validation](#cross-platform-validation)
for the full table and the caveats behind each number.

---

## Root Cause: Why PVC was slow, specifically

Direct deposition works by having every particle write its contribution straight
to the shared field grid with an atomic add &mdash; a hardware-guaranteed "don't
lose this update even if another particle writes here at the same instant." A
cross-platform microbenchmark, built specifically to isolate this one operation
from everything else the simulation does, measured that single atomic add at
matched contention levels on real hardware:

*ns per atomic add, double precision, lower is better*

| Contention | A100 | PVC (direct) | Ratio |
|---|---:|---:|---:|
| High (64 targets) | 0.246 | 0.883 | 3.6&times; |
| Moderate | 0.025 | 0.054 | 2.2&times; |
| Low | 0.0073 | 0.021 | 2.9&times; |
| Near-zero | 0.0072 | 0.041 | 5.7&times; |

No compiler flag, cache-hint variant, or hand-tuned instruction sequence closed
this gap &mdash; it was confirmed, via disassembly and hardware instrumentation,
to be a genuine property of PVC's memory fabric under contention, not a software
oversight. That reframed the problem: instead of trying to make the expensive
operation cheaper, the fix was to *do it far less often*.

**Existing gap:** Shared-memory deposition worked on CUDA & HIP only
&rarr; **Consequence:** PVC always used the slow direct path.

AMReX's shared-memory deposition algorithm exists to solve exactly this: batch a
whole tile's worth of particles into fast on-chip local memory first, then issue
one atomic write per grid cell to the real field array instead of one per
particle. It was already implemented and proven &mdash; on CUDA and HIP only. On
SYCL, it simply didn't compile. That gap, not the atomic-instruction cost itself,
was the actual lever available to pull.

## The Bug: A silent address-space mismatch

Porting the kernel to SYCL compiled cleanly on the first attempt but crashed on
real hardware with a GPU-side atomic page fault &mdash; and, unhelpfully, the
crash vanished when reproduced under a debugger, the classic signature of a race
condition. A systematic bisection (returning early from the kernel at
successively later points until the crash reappeared) narrowed the fault to one
specific call, and reading straight from AMReX's own atomic-operations header
explained why:

> **Root cause.** AMReX's SYCL atomic-add function accepts a memory address-space
> as a template parameter, but silently defaults it to `global_space` when the
> caller doesn't specify one. Shared-memory deposition's per-particle writes
> target a buffer that lives in local/shared memory &mdash; a different address
> space entirely. The atomic operation was quietly being told to look for its
> target in the wrong region of memory, which corrupts the address it resolves
> to on PVC and produces exactly the observed crash.

The fix is a small wrapper function that explicitly targets `local_space` on
SYCL, while remaining a pure pass-through on CUDA and HIP with zero behavior
change &mdash; confirmed not just by reading the code, but by disassembling the
actual compiled machine code for both the pre-fix and post-fix versions on AMD
hardware and comparing instruction-by-instruction: identical count and sequence
of synchronization barriers and atomic operations, with only register allocation
differing.

---

## Cross-Platform Validation: Tuned, not just fixed

A kernel that merely compiles and produces the right answer is not the same as
one that's actually faster. A literature check before investing further turned
up a real cautionary data point: an outside user testing a separate, related
shared-memory deposition port under review upstream (a still-open WarpX pull
request implementing a different algorithm variant) had reported it running
*20&ndash;40&times; slower* than direct deposition on their hardware &mdash; the
opposite of the speedup its own author had measured elsewhere. That made tuning,
not just correctness, a real risk for this port rather than a formality. It
turned out two parameters &mdash; thread-block size and tile size &mdash;
mattered enormously, and the values already hardcoded as WarpX's defaults
(carried over from earlier CUDA/HIP tuning) were actively harmful or, at best,
neutral on every platform tested here:

*current-deposition kernel time (incl. particle sort), dense test case, seconds &mdash; lower is better*

| Platform | Direct | Shared, old default<br>(tpb=128, tile=6,6,8) | Shared, tuned<br>(tpb=256, tile=4,4,4) |
|---|---:|---:|---:|
| Aurora/Sunspot &middot; PVC | 105.0 | 105.0 (no gain) | **33.2 (3.16&times;)** |
| Polaris &middot; A100 | 2.72 | 3.19 (0.85&times;, a loss) | **2.34 (1.16&times;)** |
| Frontier &middot; MI250X | 7.75 | 3.66 (2.12&times;) | **2.04 (3.81&times;)** |

The same tuned setting &mdash; `tpb=256`, tile size `4×4×4` &mdash; won or tied
for best on all three architectures once swept, letting the fix stand on one
well-justified default rather than three separate special cases. Both parameters
remain ordinary user-settable input-file options, so this is a starting point a
user can still override, not a ceiling. On PVC specifically, charge deposition
benefited even more from the same fix &mdash; **10.1&times;** at this density
&mdash; since its thread-block size happened to already be well-chosen; on
Polaris it produced a real but visibly smaller gain, consistent with A100
tolerating the atomics-heavy direct path better than PVC does in the first
place.

> **Reading these numbers honestly.** The shared-memory path requires an extra
> step direct deposition never does &mdash; sorting particles into per-tile bins
> before depositing &mdash; and at moderate particle density that sorting cost
> can cancel out most of the kernel-level win. The table above already includes
> it. Isolating the deposition kernel alone, without the sorting cost, showed a
> larger apparent PVC win early in this investigation (up to 10.7&times; at this
> density); that number was set aside in favor of the fuller, sort-inclusive
> figure above once the extra cost was traced and measured, since it is the
> fairer comparison for anyone deciding whether to turn this feature on.

### Correctness

Before any performance claim, the shared-memory path's numerical output was
checked field-by-field against the existing, trusted direct-deposition path
across a full simulation run &mdash; electric and magnetic fields, current
density, charge density, and divergence &mdash; landing at roughly `2×10⁻¹²`
relative difference, i.e. ordinary floating-point roundoff from summing numbers
in a different order, not a real discrepancy. Separately, all seven of WarpX's
own current- and charge-deposition unit tests &mdash; added by the WarpX
maintainers to the upstream project the same week, unprompted and unrelated to
this work &mdash; passed cleanly against the finished patch.

---

## Timeline: Sep 1 &ndash; Sep 11, day by day

**Sep 1&ndash;3 &mdash; Environment & baseline**
Synced the working tree to the latest upstream WarpX, rebuilt on Sunspot, and
confirmed via source tracing that the target benchmark genuinely exercised the
atomics-based deposition path in question. Wrote and ran a standalone,
three-way-portable microbenchmark to measure the raw atomic-add cost on PVC,
A100, and MI250X hardware directly, rather than relying on vendor
specifications.

**Sep 5 &mdash; Port begins, crash found** *(investigation)*
Began porting the shared-memory kernel to SYCL. It compiled but crashed on
hardware with a GPU-side atomic fault that disappeared under a debugger. A
scheduled maintenance window on Sunspot paused the investigation mid-bisection.

**Sep 8&ndash;9 &mdash; Root cause, fix, and correctness** *(fix landed)*
Resumed once compute nodes returned. Completed the bisection, traced the crash
to the SYCL atomic address-space default, and fixed it. Verified the corrected
kernel matched direct deposition to floating-point roundoff, then measured a
first, promising speedup on PVC.

**Sep 9&ndash;10 &mdash; Tuning & the real number** *(performance)*
Swept block-size and tile-size settings on PVC at higher, more representative
particle densities, finding the shared default settings were actively
counterproductive. Diagnosed that an initial "step-loop" comparison was
accidentally dominated by one-time disk I/O rather than compute, and re-derived
the honest, compute-only comparison.

**Sep 10&ndash;11 &mdash; Polaris & Frontier** *(cross-platform)*
Rebuilt and ran the same benchmark suite on Polaris (A100) and Frontier
(MI250X) to check whether PVC's fix and tuning generalized. Diagnosed and
worked around a stale system module conflict on Polaris and an intermittent
network path issue reaching Frontier along the way; both were pre-existing
infrastructure quirks, unrelated to the WarpX code itself.

**Sep 11 &mdash; Landed** *(shipped)*
Fast-forwarded to the latest upstream WarpX, rebuilt, re-ran the maintainers'
new unit tests to confirm nothing regressed, then committed the four-file patch
to a new branch and pushed it to the user's public GitHub fork, ready to share.

---

## Compute & Cost Accounting: What it took to get here

This work ran as an interactive, tool-using AI session rather than a fixed
compute allocation, so there is no separate "AI cost" line item distinct from
the HPC time itself &mdash; the assistant submitted and monitored real
scheduler jobs on real allocations, the same as a person would. What follows is
what can be reconstructed from the job scheduler logs and file timestamps left
behind; there is no token-usage or wall-clock-session metering available to
this report, and that gap is called out rather than estimated.

*scheduler jobs submitted, by facility*

| Facility | Scheduler | GPU | Jobs |
|---|---|---|---:|
| Aurora / Sunspot (ALCF) | PBS | Intel PVC | 25 |
| Polaris (ALCF) | PBS | NVIDIA A100 | 7 |
| Frontier (OLCF) | Slurm | AMD MI250X | 8 |

Most individual jobs were short &mdash; single-node, well under the
20&ndash;90 minute walltimes requested, since the physics test case was
deliberately kept small enough to iterate quickly rather than run at production
scale. A meaningful share of the job count reflects legitimate engineering
overhead rather than the core investigation: re-running after a scheduled
facility maintenance window, working around a stale software module on
Polaris, and recovering from a disk-quota limit hit twice while iterating on
parameter sweeps. None of that overhead is unique to using an AI agent for the
work &mdash; the same false starts show up in ordinary manual HPC debugging
&mdash; but it is included here for a complete accounting.

### What the job count bought

- **One hardware-verified root cause** instead of a guess: the atomic-cost
  disparity and the address-space bug were both confirmed by direct
  measurement and disassembly, not inferred from documentation.
- **Three independent confirmations** that the fix and tuning approach
  generalizes, rather than one lucky result on one machine.
- **A tuned default** grounded in eight distinct block-size/tile-size
  combinations measured on real silicon, not carried over unchanged from a
  different vendor's GPU.
- **Zero regressions**, checked against both this project's own correctness
  tests and the WarpX maintainers' independently-written test suite.

---

## Delivered: What's on the fork

Four files changed, 190 lines added and 43 removed, in one commit on branch
`sycl-shared-mem-deposition` of the user's public WarpX fork:

```
Source/Particles/Deposition/CurrentDeposition.H       64 changed
Source/Particles/Deposition/ChargeDeposition.H        74 changed
Source/Particles/Deposition/SharedDepositionUtils.H    77 changed
Source/WarpX.cpp                                       18 changed
```

The branch is public and requires no access grant to clone or check out. A
colleague on Aurora who had been using direct deposition and finding it too
slow has already been given the two-line input-file change (or Python/PICMI
equivalent) needed to switch over.

> **Known open item.** During Frontier/MI250X validation, shared-memory current
> deposition crashed intermittently at the highest particle density tested (640
> ppc/species). Comparing disassembled machine code confirmed the SYCL patch
> does not alter AMD codegen, so this is a pre-existing race condition in the
> original HIP implementation, not something this work introduced &mdash; but
> it was not root-caused or fixed here, and remains open. It does not affect
> the PVC target this work was scoped for.

---

*WarpX &middot; Aurora/PVC deposition performance &middot; Prepared 2026-09-12*
