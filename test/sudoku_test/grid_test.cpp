#include <boost/test/unit_test.hpp>
#include <sudoku/grid.h>
#include <sudoku/standard.h>

BOOST_AUTO_TEST_CASE(Grid_initialPotentialsAllUnblocked)
{
    sudoku::standard::Dimensions dims;
    sudoku::Grid grid(dims);

    // Each value of each cell should be available.
    for (size_t cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
        for (size_t cellValue = 1; cellValue <= dims.getMaxCellValue(); ++cellValue) {
            BOOST_REQUIRE(!grid.getCellPotential(cellPos).isBlocked(cellValue));
        }
    }
}

BOOST_AUTO_TEST_CASE(Grid_setValues_valuesBlockedInGroup)
{
    sudoku::square::Dimensions dims(2);
    sudoku::Grid grid(dims);

    // 1 2 | 0 0
    // 0 0 | 0 0
    // ----+----
    // 0 0 | 0 0
    // 0 0 | 0 0
    grid.setCellValue(0, 1);
    grid.setCellValue(1, 2);
    
    // Do some checks to make sure the values are blocked.
    BOOST_CHECK(grid.getCellPotential(3).isBlocked(1)); // cell 8 blocked by cell 0
    BOOST_CHECK(grid.getCellPotential(3).isBlocked(2)); // cell 8 blocked by cell 1
    BOOST_CHECK(grid.getCellPotential(8).isBlocked(1)); // cell 27 blocked by cell 0
    BOOST_CHECK(!grid.getCellPotential(8).isBlocked(2)); // cell 27 not blocked by cell 1
}

BOOST_AUTO_TEST_CASE(Grid_setAndClearValue)
{
    sudoku::square::Dimensions dims(2);
    sudoku::Grid grid(dims);

    grid.setCellValue(0, 1); // 1 x 0 0 <-- x blocked on value 1
                                      // 0 0 0 0
                                      // 0 0 0 0
                                      // 0 0 0 0

    grid.setCellValue(13, 1); // 1 x 0 0 <-- x blocked on value 1
                                       // 0 0 0 0
                                       // 0 0 0 0
                                       // 0 1 0 0
    
    BOOST_REQUIRE(grid.getCellPotential(1).isBlocked(1));
    
    // Remove the 1 at position 0. The cell at 'x' should still
    // be blocked by the 1 at position 13.
    grid.clearCellValue(0, 1);
    BOOST_REQUIRE(grid.getCellPotential(1).isBlocked(1));

    // Remove the 1 at position 13. The cell at 'x' should now be
    // free again.
    grid.clearCellValue(13, 1);
    BOOST_REQUIRE(!grid.getCellPotential(1).isBlocked(1));
}

BOOST_AUTO_TEST_CASE(Grid_restrictValue)
{
    sudoku::square::Dimensions dims(2);
    sudoku::Grid grid(dims);

    BOOST_REQUIRE(!grid.getCellPotential(0).isBlocked(3));
    grid.restrictCellValue(0, 3);
    BOOST_REQUIRE(grid.getCellPotential(0).isBlocked(3));
}
