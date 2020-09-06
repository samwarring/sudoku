# Convert LICENSE to LICENSE.txt (Works better with WIX generator)
configure_file(${sudoku_LICENSE_FILE} LICENSE.txt)

# Variables common to all CPack generators
set(CPACK_PACKAGE_DESCRIPTION   "General-purpose sudoku solver")
set(CPACK_HOMEPAGE_URL          "github.com/samwarring/sudoku")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt")
set(CPACK_PACKAGE_CONTACT       "github.com/samwarring/sudoku/issues")

# Variables specific to CPack WIX generator
set(CPACK_WIX_UPGRADE_GUID      "307B560E-6A80-402E-9CAE-C6B3B87E4D92")

include(CPack)

# Install a sudoku_targets.cmake file containing the imported targets
install(
    EXPORT sudoku_targets
    DESTINATION ${sudoku_INSTALLDIR_SHARE}
)

# Install a sudoku-config.cmake file for downstream projects to load
include(CMakePackageConfigHelpers)

configure_package_config_file(
    sudoku-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sudoku-config.cmake
    INSTALL_DESTINATION ${sudoku_INSTALLDIR_SHARE}
    PATH_VARS sudoku_INSTALLDIR_SHARE
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/sudoku-config.cmake
    DESTINATION ${sudoku_INSTALLDIR_SHARE}
)