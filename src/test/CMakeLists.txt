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

add_test(
    NAME sudoku_test
    COMMAND sudoku_test
)

set_tests_properties(
    sudoku_test
    PROPERTIES TIMEOUT 10
)
