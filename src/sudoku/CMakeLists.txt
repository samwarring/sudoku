add_library(
    sudoku
    dimensions.cpp
    standard.cpp
)

target_include_directories(
    sudoku
    PUBLIC ../../include
)
