# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL

import numpy as np
import pytest
from conftest import requires_shared_mem_deposition, rtol
from helpers import N_AXES, add_uniform_particles, make_sim

import pywarpx
from pywarpx import picmi

constants = picmi.constants

# The conservation identities below integrate a density over the grid with a
# Cartesian volume element. RZ deposits with an inverse volume scaling instead,
# so it needs its own test rather than this assertion.
pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in ("1d", "2d", "3d"),
    reason="conservation identity assumes a Cartesian volume element",
)


@pytest.mark.parametrize("particle_shape", ["linear", "quadratic", "cubic"])
def test_charge_deposition_conserves_total_charge(particle_shape):
    """Charge deposition must conserve the total charge of the species.

    Whatever the B-spline order, the shape factors of a macro particle sum to
    one, so summing the deposited charge density over the grid must return the
    total charge carried by the particles. Each order is a separately compiled
    kernel in ``Source/Particles/Deposition/ChargeDeposition.H``.
    """
    sim = make_sim(particle_shape=particle_shape)

    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )

    sim.initialize_inputs()
    sim.initialize_warpx()

    n_per_dim = 4
    weight = 1.0e6
    add_uniform_particles(sim, "electrons", n_per_dim=n_per_dim, weight=weight)

    n_part = n_per_dim ** N_AXES[pywarpx.libwarpx.geometry_dim]
    electrons = sim.particles.get("electrons")

    assert electrons.size == n_part

    # cross-check WarpX's own particle-side sum against the analytic value, so
    # that a failure below can be attributed to the deposition and not to the
    # particles that were added
    expected_charge = -constants.q_e * n_part * weight
    # atol=0.0, so that the relative tolerance is what actually decides. The
    # numpy default of atol=1e-8 is far larger than the ~1e-11 C compared here
    # and would make this assertion vacuous.
    assert np.isclose(
        electrons.sum_particle_charge(local=False),
        expected_charge,
        rtol=rtol(),
        atol=0.0,
    )

    # allocates a nodal MultiFab, resets it, deposits, sums the guard cells and
    # applies the boundary/volume treatment
    rho = electrons.get_charge_density(lev=0, local=False)

    # integrate rho over the domain: sum the unique nodes, times the cell volume.
    # Passing the periodicity matters, as rho is nodal and the nodes on the periodic
    # boundary would otherwise be counted twice
    geom = sim.extension.warpx.Geom(0)
    cell_volume = float(np.prod(geom.data().CellSize()))
    deposited_charge = (
        rho.sum_unique(comp=0, local=False, period=geom.periodicity()) * cell_volume
    )

    assert np.isclose(deposited_charge, expected_charge, rtol=rtol(), atol=0.0)


# Deposited rho from each (shape, shared) case, filled in as the parametrized
# cases run so the second of a pair can compare against the first.
#
# This exists because a test function must run exactly ONE simulation. An
# earlier version ran both the direct and the shared deposition inside one
# test, finalizing in between. That passed on SYCL but aborted the whole
# pytest process on HIP with "ParmParse::Initialize(): already initialized"
# once enough init/finalize cycles had accumulated -- Frontier job 5786040
# died at the third such test, taking the remaining cases with it. One
# simulation per test keeps the cycle count the same as every other test in
# this suite, which is a pattern already known to work on all backends.
_DEPOSITED_RHO = {}


@requires_shared_mem_deposition
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("particle_shape", ["linear", "quadratic", "cubic"])
def test_charge_deposition_shared_mem_matches_direct(particle_shape, shared):
    """Shared-memory charge deposition must agree with the direct deposition.

    Compares the two kernels cell by cell rather than against an analytic
    total, which is what catches a buffer that is indexed or reduced
    incorrectly: a kernel depositing the right total charge into the wrong
    cells satisfies the conservation identity above but fails here. Every
    B-spline order is a separately compiled kernel sizing its shared-memory
    buffer from its own stencil width, so all three are covered.

    Runs one deposition per invocation and caches it; whichever of the
    ``shared`` pair runs second makes the comparison.
    """
    # the autouse lifecycle fixture resets this flag after the test
    pywarpx.warpx.do_shared_mem_charge_deposition = shared
    sim = make_sim(particle_shape=particle_shape)
    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )
    sim.initialize_inputs()
    sim.initialize_warpx()

    add_uniform_particles(sim, "electrons", n_per_dim=4, weight=1.0e6)

    electrons = sim.particles.get("electrons")
    rho = np.asarray(electrons.get_charge_density(lev=0, local=False)[:])

    # a deposition that produced nothing would otherwise compare equal to
    # another empty one, so check this case stands on its own first
    assert np.isfinite(rho).all()
    assert rho.min() < 0.0, "electrons must deposit negative charge somewhere"

    _DEPOSITED_RHO[(particle_shape, shared)] = rho

    counterpart = _DEPOSITED_RHO.get((particle_shape, not shared))
    if counterpart is None:
        return  # the other half of the pair has not run yet

    direct = _DEPOSITED_RHO[(particle_shape, False)]
    shared_rho = _DEPOSITED_RHO[(particle_shape, True)]
    assert shared_rho.shape == direct.shape
    np.testing.assert_allclose(
        shared_rho,
        direct,
        rtol=rtol(),
        atol=np.abs(direct).max() * rtol(),
    )


def test_charge_deposition_is_negative_for_electrons():
    """Electrons must deposit a negative charge density everywhere."""
    sim = make_sim()
    sim.add_species(
        picmi.Species(particle_type="electron", name="electrons"), layout=None
    )

    sim.initialize_inputs()
    sim.initialize_warpx()

    add_uniform_particles(sim, "electrons")

    electrons = sim.particles.get("electrons")
    rho = electrons.get_charge_density(lev=0, local=False)

    assert rho.max(0) <= 0.0
    assert rho.min(0) < 0.0
