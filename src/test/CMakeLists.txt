find_package(Boost REQUIRED unit_test_framework)

add_executable(
    sudoku_test
    sudoku_test.cpp
    dimensions_test.cpp
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
