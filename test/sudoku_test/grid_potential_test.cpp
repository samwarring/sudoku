#include <boost/test/unit_test.hpp>
#include <sudoku/grid_potential.h>
#include <sudoku/standard.h>

BOOST_AUTO_TEST_CASE(GridPotential_initialPotentialsAllUnblocked)
{
    sudoku::standard::Dimensions dims;
    sudoku::GridPotential gridPotential(dims);

    // Each value of each cell should be available.
    for (size_t cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
        for (size_t cellValue = 1; cellValue <= dims.getMaxCellValue(); ++cellValue) {
            BOOST_REQUIRE(!gridPotential.getCellPotential(cellPos).isBlocked(cellValue));
        }
    }
}

BOOST_AUTO_TEST_CASE(GridPotential_setValues_valuesBlockedInGroup)
{
    sudoku::square::Dimensions dims(2);
    sudoku::GridPotential gridPotential(dims);

    // 1 2 | 0 0
    // 0 0 | 0 0
    // ----+----
    // 0 0 | 0 0
    // 0 0 | 0 0
    gridPotential.setCellValue(0, 1);
    gridPotential.setCellValue(1, 2);
    
    // Do some checks to make sure the values are blocked.
    BOOST_CHECK(gridPotential.getCellPotential(3).isBlocked(1)); // cell 8 blocked by cell 0
    BOOST_CHECK(gridPotential.getCellPotential(3).isBlocked(2)); // cell 8 blocked by cell 1
    BOOST_CHECK(gridPotential.getCellPotential(8).isBlocked(1)); // cell 27 blocked by cell 0
    BOOST_CHECK(!gridPotential.getCellPotential(8).isBlocked(2)); // cell 27 not blocked by cell 1
}

BOOST_AUTO_TEST_CASE(GridPotential_setAndClearValue)
{
    sudoku::square::Dimensions dims(2);
    sudoku::GridPotential gridPotential(dims);

    gridPotential.setCellValue(0, 1); // 1 x 0 0 <-- x blocked on value 1
                                      // 0 0 0 0
                                      // 0 0 0 0
                                      // 0 0 0 0

    gridPotential.setCellValue(13, 1); // 1 x 0 0 <-- x blocked on value 1
                                       // 0 0 0 0
                                       // 0 0 0 0
                                       // 0 1 0 0
    
    BOOST_REQUIRE(gridPotential.getCellPotential(1).isBlocked(1));
    
    // Remove the 1 at position 0. The cell at 'x' should still
    // be blocked by the 1 at position 13.
    gridPotential.clearCellValue(0, 1);
    BOOST_REQUIRE(gridPotential.getCellPotential(1).isBlocked(1));

    // Remove the 1 at position 13. The cell at 'x' should now be
    // free again.
    gridPotential.clearCellValue(13, 1);
    BOOST_REQUIRE(!gridPotential.getCellPotential(1).isBlocked(1));
}

BOOST_AUTO_TEST_CASE(GridPotential_restrictValue)
{
    sudoku::square::Dimensions dims(2);
    sudoku::GridPotential gridPotential(dims);

    BOOST_REQUIRE(!gridPotential.getCellPotential(0).isBlocked(3));
    gridPotential.restrictCellValue(0, 3);
    BOOST_REQUIRE(gridPotential.getCellPotential(0).isBlocked(3));
}
