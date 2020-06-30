
# 3.10.2 - Ubuntu 18.04
# 3.17.2 - Appveyor (VS2019)
cmake_minimum_required(VERSION 3.10)

# Get version info from CMake cache
set(sudoku_VERSION      "0.0.0" CACHE STRING "sudoku version (X.Y.Z only)")
set(sudoku_VERSION_DESC "none"  CACHE STRING "git describe --long --tags --always")
set(sudoku_BRANCH       "none"  CACHE STRING "current branch")
set(sudoku_COMMIT_DATE  "none"  CACHE STRING "timestamp of current commit")
set(sudoku_BUILD_DATE   "none"  CACHE STRING "timestamp of build")

project(sudoku VERSION ${sudoku_VERSION})
enable_testing()
add_subdirectory(src)
add_subdirectory(test)
