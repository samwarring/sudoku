find_package(Boost REQUIRED program_options)

add_executable(
    sudoku_cli
    sudoku_cli.cpp
    program_options.cpp
)

# Required for std::optional
set_property(
    TARGET sudoku_cli
    PROPERTY CXX_STANDARD 17
)

target_include_directories(
    sudoku_cli
    PRIVATE ${CMAKE_CURRENT_LISTS_DIR}
)

target_link_libraries(
    sudoku_cli
    PRIVATE sudoku
            Boost::program_options
)

add_test(
    NAME sudoku_cli
    COMMAND sudoku_cli --input 0
)
