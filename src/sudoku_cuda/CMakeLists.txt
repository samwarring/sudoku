add_library(
    sudoku_cuda
    kernel.cu
    result.cpp
    dimensions.cu
    grid.cu
    guess_stack.cu
)

target_include_directories(
    sudoku_cuda
    PUBLIC ../../include
)

target_link_libraries(
    sudoku_cuda
    PUBLIC sudoku
)

set_target_properties(
    sudoku_cuda
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
