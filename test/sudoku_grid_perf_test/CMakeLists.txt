find_package(Boost REQUIRED program_options)

add_executable(
    sudoku_grid_perf_test
    sudoku_grid_perf_test.cpp
)

target_compile_definitions(
    sudoku_grid_perf_test
    PRIVATE SUDOKU_CONFIG_IS_DEBUG=$<CONFIG:Debug>
)

target_link_libraries(
    sudoku_grid_perf_test
    PRIVATE sudoku
            Boost::program_options
)
