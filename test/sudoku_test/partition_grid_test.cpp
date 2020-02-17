#include <algorithm>
#include <set>
#include <boost/test/unit_test.hpp>
#include <sudoku/partition_grid.h>
#include <sudoku/standard.h>

struct PartitionGridTestFixture
{
    sudoku::standard::Dimensions dims;
    sudoku::PartitionCount partitionCount{ 4 };
    std::vector<sudoku::PartitionCount> partitionIds
    {
        sudoku::partitionRoundRobin(dims.getCellCount(), partitionCount)
    };
    sudoku::PartitionTable partitionTable{ dims, partitionCount, partitionIds };
    sudoku::PartitionGrid grid{ dims, partitionTable };
};

BOOST_FIXTURE_TEST_SUITE(PartitionGridTestSuite, PartitionGridTestFixture)

    BOOST_AUTO_TEST_CASE(SetCellValue_GetMaxBlockEmptyCell)
    {
        // Prepare cell values:
        //
        // 1  0  0  0  0  0  0  0  * <-- max block empty cell
        //                   0  0  0
        //                   2  0  0
        //                         0
        //                         0
        //                         0
        //                         0
        //                         0
        //                         3
        grid.setCellValue(0, 1);
        grid.setCellValue(24, 2);
        grid.setCellValue(80, 3);

        BOOST_CHECK_EQUAL(grid.getMaxBlockEmptyCell(), 8);
        BOOST_CHECK_EQUAL(grid.getCellBlockCount(8), 3);
        BOOST_CHECK_EQUAL(grid.getNextAvailableValue(8, 0), 4);
    }

    BOOST_AUTO_TEST_CASE(ClearCellValue_getMaxBlockEmptyCell)
    {
        // Prepare cell values
        //
        // 1 2 3  0 0 0 ... <-- first subsquare is filled.
        // 4 5 6  0 0 0 ...     remaining cells are empty.
        // 7 8 9  0 0 0 ...
        // 0 0 0  0 0 0 ...
        // ...
        grid.setCellValue(0, 1);
        grid.setCellValue(1, 2);
        grid.setCellValue(2, 3);
        grid.setCellValue(9, 4);
        grid.setCellValue(10, 5);
        grid.setCellValue(11, 6);
        grid.setCellValue(18, 7);
        grid.setCellValue(19, 8);
        grid.setCellValue(20, 9);

        // Max block empty cell must be outside of the first subsquare
        auto maxBlockEmptyCell = grid.getMaxBlockEmptyCell();
        std::set<sudoku::CellCount> subsquarePositions{0, 1, 2, 9, 10, 11, 18, 19, 20};
        auto match = subsquarePositions.find(maxBlockEmptyCell);
        BOOST_REQUIRE(match == subsquarePositions.end());

        // Clear the center of the subsquare (previously set to 5)
        grid.clearCellValue(10);

        // New max block empty cell should be the cell we just cleared.
        // Its next available value should be 5.
        maxBlockEmptyCell = grid.getMaxBlockEmptyCell();
        BOOST_CHECK_EQUAL(maxBlockEmptyCell, 10);
        BOOST_CHECK_EQUAL(grid.getNextAvailableValue(10, 0), 5);
        BOOST_CHECK_EQUAL(grid.getCellBlockCount(maxBlockEmptyCell), 8);
    }

BOOST_AUTO_TEST_SUITE_END()
