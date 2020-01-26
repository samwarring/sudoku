add_library(
    sudoku_cuda
    dimensions.cu
)

target_include_directories(
    sudoku_cuda
    PUBLIC ../../include
)
