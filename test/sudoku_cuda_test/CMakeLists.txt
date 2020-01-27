find_package(Boost REQUIRED unit_test_framework)

add_executable(
    sudoku_cuda_test
    sudoku_cuda_test.cpp
    kernel_test.cpp
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

add_test(
    NAME sudoku_cuda_test
    COMMAND sudoku_cuda_test
)

set_tests_properties(
    sudoku_cuda_test
    PROPERTIES TIMEOUT 10
)
