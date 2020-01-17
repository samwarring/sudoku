# 1.58.0 : WSL
find_package(Boost REQUIRED unit_test_framework)

add_executable(
    sudoku_test
    sudoku_test.cpp
    dimensions_test.cpp
    groups_test.cpp
    potential_test.cpp
    solver_test.cpp
    cell_value_parser_test.cpp
    formatter_test.cpp
    solution_queue_test.cpp
    square_test.cpp
    parallel_solver_test.cpp
)

target_include_directories(
    sudoku_test
    PRIVATE ${CMAKE_CURRENT_SOUCE_DIR}
)

target_link_libraries(
    sudoku_test
    PRIVATE sudoku
            Boost::unit_test_framework
)

# Boost tests must define BOOST_TEST_DYN_LINK if not linking statically.
if (NOT Boost_USE_STATIC_LIBS)
    target_compile_definitions(sudoku_test PRIVATE BOOST_TEST_DYN_LINK=1)
endif()

add_test(
    NAME sudoku_test
    COMMAND sudoku_test
)

set_tests_properties(
    sudoku_test
    PROPERTIES TIMEOUT 10
)
