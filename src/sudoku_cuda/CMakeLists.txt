add_library(
    sudoku_cuda
    sudoku_cuda.cu
)

target_include_directories(
    sudoku_cuda
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(
    sudoku_cuda
    PUBLIC sudoku
)
