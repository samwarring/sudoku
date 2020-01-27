add_library(
    sudoku_cuda
    kernel.cu
    result.cpp
)

target_include_directories(
    sudoku_cuda
    PUBLIC ../../include
)

target_link_libraries(
    sudoku_cuda
    PUBLIC sudoku
)
