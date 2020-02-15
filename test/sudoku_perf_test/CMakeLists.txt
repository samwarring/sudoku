add_executable(
    sudoku_perf_test
    sudoku_perf_test.cpp
)

target_link_libraries(
    sudoku_perf_test
    PRIVATE sudoku
)

target_compile_definitions(
    sudoku_perf_test
    PRIVATE SUDOKU_PERF_TEST_IS_DEBUG=$<CONFIG:Debug>
)
