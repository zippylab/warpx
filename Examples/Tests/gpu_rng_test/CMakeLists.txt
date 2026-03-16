# gpu_rng_test: standalone AMReX diagnostic for ParallelForRNG on GPU.
#
# This test does NOT use the WarpX binary. It links directly against AMReX
# and calls ParallelForRNG in the same way that WarpX's AddPlasma does.
# It is therefore independent of WarpX_DIMS and WarpX_APP.

# AMReX::amrex is always present in the WarpX build (FetchContent or found)
if(NOT TARGET AMReX::amrex)
    message(STATUS "gpu_rng_test: skipping (AMReX::amrex target not found)")
    return()
endif()

add_executable(gpu_rng_test main.cpp)
target_link_libraries(gpu_rng_test PRIVATE AMReX::amrex)

set_target_properties(gpu_rng_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpu_rng_test_1rank)

# configure_mpiexec(N) sets MPI_TEST_EXE in the caller's scope
configure_mpiexec(1)
add_test(
    NAME test_gpu_rng_1rank.run
    COMMAND ${MPI_TEST_EXE} $<TARGET_FILE:gpu_rng_test> amrex.verbose=0
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpu_rng_test_1rank
)
set_property(TEST test_gpu_rng_1rank.run APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")

if(WarpX_MPI)
    file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpu_rng_test_12rank)
    configure_mpiexec(12)
    add_test(
        NAME test_gpu_rng_12rank.run
        COMMAND ${MPI_TEST_EXE} $<TARGET_FILE:gpu_rng_test> amrex.verbose=0
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpu_rng_test_12rank
    )
    set_property(TEST test_gpu_rng_12rank.run APPEND PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
