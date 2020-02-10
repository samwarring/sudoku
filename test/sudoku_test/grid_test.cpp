#include <boost/test/unit_test.hpp>
#include <sudoku/grid.h>
#include <sudoku/standard.h>

struct GridTestSuiteFixture
{
    sudoku::square::Dimensions dims{2};
    sudoku::Grid grid{dims};

    void fillGrid()
    {
        // Fill grid with a valid solution
        std::vector<sudoku::CellValue> cellValues{
            1, 2, 3, 4,
            3, 4, 1, 2,
            2, 1, 4, 3,
            4, 3, 2, 1
        };
        for (sudoku::CellCount cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
            grid.setCellValue(cellPos, cellValues[cellPos]);
        }
    }
};

BOOST_FIXTURE_TEST_SUITE(GridTestSuite, GridTestSuiteFixture)

    BOOST_AUTO_TEST_CASE(initialPotentialsAllUnblocked)
    {
        // Each value of each cell should be available.
        for (sudoku::CellCount cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
            for (sudoku::CellValue cellValue = 1; cellValue <= dims.getMaxCellValue(); ++cellValue) {
                BOOST_REQUIRE(!grid.getCellPotential(cellPos).isBlocked(cellValue));
            }
        }
    }

    BOOST_AUTO_TEST_CASE(setAndGetValue)
    {
        grid.setCellValue(3, 3);
        BOOST_REQUIRE_EQUAL(grid.getCellValue(3), 3);
    }

    BOOST_AUTO_TEST_CASE(setValues_valuesBlockedInGroup)
    {
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

    BOOST_AUTO_TEST_CASE(setAndClearValue)
    {
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
        grid.clearCellValue(0);
        BOOST_REQUIRE(grid.getCellPotential(1).isBlocked(1));

        // Remove the 1 at position 13. The cell at 'x' should now be
        // free again.
        grid.clearCellValue(13);
        BOOST_REQUIRE(!grid.getCellPotential(1).isBlocked(1));
    }

    BOOST_AUTO_TEST_CASE(restrictValue)
    {
        BOOST_REQUIRE(!grid.getCellPotential(0).isBlocked(3));
        grid.restrictCellValue(0, 3);
        BOOST_REQUIRE(grid.getCellPotential(0).isBlocked(3));
    }

    BOOST_AUTO_TEST_CASE(getMaxBlockEmptyCell_emptyGrid)
    {
        auto cellPos = grid.getMaxBlockEmptyCell();
        BOOST_REQUIRE_LT(cellPos, dims.getCellCount());
    }

    BOOST_AUTO_TEST_CASE(getMaxBlockEmptyCell_fullGrid)
    {
        fillGrid();
        BOOST_REQUIRE_EQUAL(grid.getMaxBlockEmptyCell(), dims.getCellCount());
    }

    BOOST_AUTO_TEST_CASE(getMaxBlockEmptyCell_EmptyCellCompletelyBlocked)
    {
        // 1 x 0 0  (x is completely blocked)
        // 0 2 0 0
        // 0 3 0 0
        // 0 4 0 0
        grid.setCellValue(0, 1);
        grid.setCellValue(5, 2);
        grid.setCellValue(9, 3);
        grid.setCellValue(13, 4);
        BOOST_REQUIRE_EQUAL(grid.getMaxBlockEmptyCell(), 1);
    }

    BOOST_AUTO_TEST_CASE(isFull_emptyGrid)
    {
        BOOST_REQUIRE(!grid.isFull());
    }

    BOOST_AUTO_TEST_CASE(isFull_fullGrid)
    {
        fillGrid();
        BOOST_REQUIRE(grid.isFull());
    }

    BOOST_AUTO_TEST_CASE(isFull_almostFull)
    {
        fillGrid();
        grid.clearCellValue(0);
        BOOST_REQUIRE(!grid.isFull());
    }

BOOST_AUTO_TEST_SUITE_END()
