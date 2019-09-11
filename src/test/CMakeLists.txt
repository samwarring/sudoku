# 1.58.0 : WSL
find_package(Boost REQUIRED unit_test_framework)

add_executable(
    sudoku_test
    sudoku_test.cpp
    dimensions_test.cpp
    groups_test.cpp
)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")
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
