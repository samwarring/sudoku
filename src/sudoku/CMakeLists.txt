add_library(
    sudoku
    dimensions.cpp
    standard.cpp
    groups.cpp
    potential.cpp
    solver.cpp
    cell_value_parser.cpp
    formatter.cpp
    solution_queue.cpp
)

target_include_directories(
    sudoku
    PUBLIC ../../include
)

target_compile_features(
    sudoku
    PUBLIC cxx_template_template_parameters
           cxx_range_for
)
