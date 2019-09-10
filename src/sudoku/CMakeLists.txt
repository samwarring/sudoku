add_library(
    sudoku
    dimensions.cpp
    standard.cpp
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
