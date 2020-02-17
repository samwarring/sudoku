add_library(
    sudoku
    dimensions.cpp
    groups.cpp
    potential.cpp
    solver.cpp
    cell_value_parser.cpp
    formatter.cpp
    solution_queue.cpp
    square.cpp
    parallel_solver.cpp
    grid.cpp
    fork.cpp
    block_count_tracker.cpp
    partitions.cpp
    partition_table.cpp
    partition_block_count_tracker.cpp
    partition_grid.cpp
)

target_include_directories(
    sudoku
    PUBLIC ../../include
)

target_compile_features(
    sudoku
    PUBLIC cxx_template_template_parameters
           cxx_range_for
	   cxx_std_14
)

# Using std::thread
find_package(Threads)
target_link_libraries(
    sudoku
    PUBLIC Threads::Threads
)
