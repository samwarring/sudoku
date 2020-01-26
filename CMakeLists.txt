# 3.5.1 : WSL
# 3.9.x : CUDA support
cmake_minimum_required(VERSION 3.5)
project(sudoku VERSION 0.0.0 LANGUAGES CXX CUDA)
enable_testing()
add_subdirectory(src)
add_subdirectory(test)
