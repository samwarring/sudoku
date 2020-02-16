find_package(Boost REQUIRED program_options)

add_executable(
    sudoku_partition_test
    sudoku_partition_test.cpp
)

target_link_libraries(
    sudoku_partition_test
    PRIVATE sudoku
            Boost::program_options
)
