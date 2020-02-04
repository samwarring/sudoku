#include <boost/test/unit_test.hpp>
#include <sudoku/square.h>
#include <sudoku/dimensions.h>
#include <sudoku/grid.h>
#include "grid_kernels.h"

BOOST_AUTO_TEST_CASE(Grid_4x4_setAndClearCellValue)
{
    sudoku::square::Dimensions dims(2);
    GridKernels grid(dims);
    
    grid.setCellValue(0, 3);
    BOOST_CHECK_EQUAL(grid.getCellValue(0), 3);
    BOOST_CHECK_EQUAL(grid.getCellBlockCount(1), 1);
    BOOST_CHECK_EQUAL(grid.getCellBlockCount(5), 1);
    BOOST_CHECK_EQUAL(grid.getValueBlockCount(2, 3), 1);
    BOOST_CHECK_LT(grid.getCellBlockCount(0), 0);

    grid.clearCellValue(0);
    BOOST_CHECK_EQUAL(grid.getCellValue(0), 0);
    BOOST_CHECK_EQUAL(grid.getCellBlockCount(1), 0);
    BOOST_CHECK_EQUAL(grid.getCellBlockCount(5), 0);
    BOOST_CHECK_EQUAL(grid.getValueBlockCount(2, 3), 0);
    BOOST_CHECK_EQUAL(grid.getCellBlockCount(0), 0);
}

BOOST_AUTO_TEST_CASE(Grid_4x4_getMaxCellBlockCountPos)
{
    sudoku::square::Dimensions dims(2);
    GridKernels grid(dims);

    // Position reference:
    //
    //  0  1 |  2  3
    //  4  5 |  6  7
    // ------+------
    //  8  9 | 10 11
    // 12 13 | 14 15

    grid.setCellValue(0, 1);
    grid.setCellValue(6, 2);
    grid.setCellValue(15, 3);

    BOOST_CHECK_EQUAL(grid.getMaxCellBlockCountPos(), 3);
}
