
# 3.10.2 - Ubuntu 18.04
# 3.17.2 - Appveyor (VS2019)
cmake_minimum_required(VERSION 3.10)

# Get version info from CMake cache
set(sudoku_VERSION      "0.0.0" CACHE STRING "sudoku version (X.Y.Z only)")
set(sudoku_VERSION_DESC "none"  CACHE STRING "git describe --long --tags --always")
set(sudoku_BRANCH       "none"  CACHE STRING "current branch")
set(sudoku_COMMIT_DATE  "none"  CACHE STRING "timestamp of current commit")
set(sudoku_BUILD_DATE   "none"  CACHE STRING "timestamp of build")

# Identify compilers
project(sudoku VERSION ${sudoku_VERSION})
enable_testing()

# 1.59.0 - Added BOOST_DATA_TEST_CASE
# 1.65.1 - Ubuntu 18.04
find_package(Boost 1.59 REQUIRED unit_test_framework
                                 program_options)

# Find pthreads or std::thread
find_package(Threads)

# Define paths to files in the root directory
set(sudoku_LICENSE_FILE "${CMAKE_CURRENT_LIST_DIR}/LICENSE")

# Installation relative directories
set(sudoku_INSTALLDIR_ARCHIVE lib/sudoku-${PROJECT_VERSION})
set(sudoku_INSTALLDIR_RUNTIME bin)
set(sudoku_INSTALLDIR_HEADERS include)

# Add targets from subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(pkg)