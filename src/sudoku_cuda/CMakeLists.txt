add_library(
    sudoku_cuda
    sudoku_cuda.cu
    guess_stack.cu
)

target_include_directories(
    sudoku_cuda
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(
    sudoku_cuda
    PUBLIC sudoku
)

set_target_properties(
    sudoku_cuda
    PROPERTIES CUDA_SEPARABLE_COMPILATION TRUE
)
