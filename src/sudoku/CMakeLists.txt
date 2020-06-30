# Create version.cpp from version.in.cpp
configure_file(
    version.in.cpp
    version.cpp
)

add_library(
    sudoku
    dimensions.cpp
    groups.cpp
    potential.cpp
    solver.cpp
    cell_value_parser.cpp
    formatter.cpp
    solution_queue.cpp
    parallel_solver.cpp
    grid.cpp
    fork.cpp
    block_count_tracker.cpp
    groupwise_block_counter.cpp
    groupwise_empty_solver.cpp
    groupwise_solver.cpp
    inner_rectangular.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
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
