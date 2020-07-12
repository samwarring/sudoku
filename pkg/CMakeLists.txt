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