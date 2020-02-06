find_package(Boost REQUIRED unit_test_framework)

add_executable(
    sudoku_cuda_test
    sudoku_cuda_test.cpp
    block_counter_test.cpp
    block_counter_kernels.cu
    related_groups_test.cpp
    related_groups_kernels.cu
    grid_test.cpp
    grid_kernels.cu
    guess_stack_test.cpp
    guess_stack_kernels.cu
    solver_test.cpp
)

target_include_directories(
    sudoku_cuda_test
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(
    sudoku_cuda_test
    PRIVATE sudoku_cuda
            Boost::unit_test_framework
)

# Boost tests must define BOOST_TEST_DYN_LINK if not linking statically.
if (NOT Boost_USE_STATIC_LIBS)
    target_compile_definitions(sudoku_cuda_test PRIVATE BOOST_TEST_DYN_LINK=1)
endif()

# Some CUDA errors are "sticky." Once a process encounters a sticky error,
# subsequent CUDA API calls in the same process will fail. The only way to
# recover from a "sticky" error is to restart the process. To prevent a
# sticky error in one test case from causing failures in the others, we will
# run each suite in its own process.
#
# See: https://stackoverflow.com/a/56330491
#      https://stackoverflow.com/a/31642573

function(add_sudoku_cuda_test TEST_NAME TEST_PATTERN)
    add_test(
        NAME ${TEST_NAME}
        COMMAND sudoku_cuda_test -t "${TEST_PATTERN}"
    )
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 10 RUN_SERIAL TRUE)
endfunction()

add_sudoku_cuda_test(sudoku_cuda_test_block_counter  "BlockCounter*")
add_sudoku_cuda_test(sudoku_cuda_test_related_groups "RelatedGroups*")
add_sudoku_cuda_test(sudoku_cuda_test_grid           "Grid*")
add_sudoku_cuda_test(sudoku_cuda_test_guess_stack    "GuessStack*")
add_sudoku_cuda_test(sudoku_cuda_test_solver         "Solver*")
