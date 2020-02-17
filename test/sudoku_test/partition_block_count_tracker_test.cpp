#include <set>
#include <boost/test/unit_test.hpp>
#include <sudoku/partition_block_count_tracker.h>
#include <sudoku/standard.h>

struct PartitionBlockCountTrackerTestFixture
{
    sudoku::square::Dimensions dims{2};
    std::vector<sudoku::PartitionCount> partitionIds{
        0, 0, 1, 1,
        0, 0, 1, 1,
        2, 2, 3, 3,
        2, 2, 3, 3
    };
    sudoku::PartitionTable partitionTable{dims, 4, partitionIds};
    std::vector<sudoku::PartitionBlockCountTracker> trackers;

    PartitionBlockCountTrackerTestFixture()
    {
        for (sudoku::PartitionCount partitionId = 0; partitionId < 4; ++partitionId) {
            trackers.emplace_back(partitionTable, partitionId, dims.getMaxCellValue());
        }
    }
};

BOOST_FIXTURE_TEST_SUITE(PartitionBlockCountTrackerTestSuite, PartitionBlockCountTrackerTestFixture)

    BOOST_AUTO_TEST_CASE(SetCellValue_relatedCellsAreBlocked)
    {
        // Set global pos 0 to value 1
        trackers[1].setCellValue(0, 1);

        // 2 of the cells in partition 1 are now blocked on cell value 1.
        // Which 2 depend on how the partition table assigned the indices.
        // Let's not assume which 2 cells are blocked - only that we have 2.
        auto countRelatedCellsBlockedOn1 = 0;
        for (sudoku::CellCount localPos = 0; localPos < 4; ++localPos) {
            if (trackers[1].isBlocked(localPos, 1)) {
                ++countRelatedCellsBlockedOn1;
            }
        }
        BOOST_REQUIRE_EQUAL(countRelatedCellsBlockedOn1, 2);

        // Similarly, 2 of the cells in partition 1 now have a cell block count of 1.
        std::set<sudoku::CellBlockCount> expectedBlockCounts{0, 0, 1, 1};
        std::set<sudoku::CellBlockCount> actualBlockCounts;
        for (sudoku::CellCount localPos = 0; localPos < 4; ++localPos) {
            actualBlockCounts.insert(trackers[1].getCellBlockCount(localPos));
        }
        BOOST_REQUIRE_EQUAL_COLLECTIONS(expectedBlockCounts.begin(), expectedBlockCounts.end(),
                                        actualBlockCounts.begin(), actualBlockCounts.end());
    }

    BOOST_AUTO_TEST_CASE(SetCellValue_getMaxBlockEmptyCell)
    {
        // 0 0 1 1
        // 0 0 * 1 <-- global position 6 is assigned
        // 2 2 3 3
        // 2 2 3 3
        trackers[0].setCellValue(6, 4);
        trackers[1].setCellValue(6, 4);
        trackers[2].setCellValue(6, 4);
        trackers[3].setCellValue(6, 4);

        // All of partition 2 is unrelated to global position 6.
        // The max block empty cell will have block count = 0.
        auto maxBlockEmptyCell2 = trackers[2].getMaxBlockEmptyCell();
        BOOST_REQUIRE_EQUAL(trackers[2].getCellBlockCount(maxBlockEmptyCell2), 0);

        // Partitions 0,1,3 each cover at least one cell related
        // to global position 6.
        std::vector<sudoku::PartitionCount> partitionIds{0, 1, 3};
        for (auto partitionId : partitionIds) {
            sudoku::PartitionBlockCountTracker& tracker = trackers[partitionId];
            auto maxBlockEmptyCell = tracker.getMaxBlockEmptyCell();
            BOOST_REQUIRE_EQUAL(tracker.getCellBlockCount(maxBlockEmptyCell), 1);

            // Verify the maxBlockEmpty cell is actually related to global pos 6.
            auto globalMaxBlockEmptyCell = partitionTable.getCellPosition(partitionId, maxBlockEmptyCell);
            auto globalRelatedCells = dims.getRelatedCells(6);
            auto match = std::find(globalRelatedCells.begin(), globalRelatedCells.end(), globalMaxBlockEmptyCell);
            BOOST_REQUIRE(match != globalRelatedCells.end());
        }
    }

    BOOST_AUTO_TEST_CASE(SetAndClearValue_getMaxBlockEmptyCell)
    {
        // Fill in all of partition 2 with values.
        trackers[2].setCellValue(8, 1);
        trackers[2].setCellValue(9, 2);
        trackers[2].setCellValue(12, 3);
        trackers[2].setCellValue(13, 4);

        // By clearing one of the above cells, that cell should become
        // the max block empty cell for the partition.
        trackers[2].clearCellValue(12, 3);
        auto localMaxBlockEmptyCell = trackers[2].getMaxBlockEmptyCell();
        BOOST_REQUIRE_EQUAL(partitionTable.getCellPosition(2, localMaxBlockEmptyCell), 12);
    }

    BOOST_AUTO_TEST_CASE(BlockCornerCell_getMaxBlockEmptyCell)
    {
        // Set up sudoku so corner cell is max-blocked
        // 1  0  0  x <-- partition 1 contains the global max block empty cell
        // 0  0  2  0
        // 0  0  0  0
        // 0  0  0  3

        for (sudoku::PartitionCount partitionId = 0; partitionId < 4; ++partitionId) {
            trackers[partitionId].setCellValue(0, 1);
            trackers[partitionId].setCellValue(6, 2);
            trackers[partitionId].setCellValue(15, 3);
        }

        std::vector<sudoku::CellCount> localMaxBlockEmptyCells(4);
        for (sudoku::PartitionCount partitionId = 0; partitionId < 4; ++partitionId) {
            localMaxBlockEmptyCells[partitionId] = trackers[partitionId].getMaxBlockEmptyCell();
        }

        // Partition 1 should have selected the corer cell, because that cell
        // has a block count of 3.
        auto cellBlockCount1 = trackers[1].getCellBlockCount(localMaxBlockEmptyCells[1]);
        BOOST_REQUIRE_EQUAL(cellBlockCount1, 3);

        // The other partitions selected their own maxBlockEmpty cell, but the block counts
        // for those selected cells should all be less than 3.
        std::vector<sudoku::PartitionCount> otherPartitionIds{0, 2, 3};
        for (auto partitionId : otherPartitionIds) {
            auto otherBlockCount = trackers[partitionId].getCellBlockCount(localMaxBlockEmptyCells[partitionId]);
            BOOST_REQUIRE_LT(otherBlockCount, cellBlockCount1);
        }
    }

BOOST_AUTO_TEST_SUITE_END()
