#include <boost/test/unit_test.hpp>
#include <sudoku/block_count_tracker.h>
#include <sudoku/square.h>

void alterBlockCount(sudoku::BlockCountTracker& tracker, size_t cellPos, int delta)
{
    if (delta > 0) {
        for (int i = 0; i < delta; ++i) {
            tracker.incrementBlockCount(cellPos);
        }
    }
    else {
        for (int i = 0; i < (-delta); ++i) {
            tracker.derementBlockCount(cellPos);
        }
    }
}

BOOST_AUTO_TEST_CASE(BlockCountTracker_incrementBlockCount)
{
    sudoku::square::Dimensions dims(2);
    sudoku::BlockCountTracker tracker(dims);

    tracker.incrementBlockCount(4);  // blockCount(4)=1
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 4);

    tracker.incrementBlockCount(10); // blockCount(4)=1, blockCount(10)=1
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 4);

    tracker.incrementBlockCount(10); // blockCount(4)=1, blockCount(10)=2
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 10);

    alterBlockCount(tracker, 0, 3);  // blockCount(0)=3, blockCount(4)=1, blockCount(10)=3
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 0);
}

BOOST_AUTO_TEST_CASE(BlockCountTracker_decrementBlockCount)
{
    sudoku::square::Dimensions dims(2);
    sudoku::BlockCountTracker tracker(dims);

    alterBlockCount(tracker, 1, 4);
    alterBlockCount(tracker, 2, 4);
    alterBlockCount(tracker, 3, 4); // blockCount(1)=4, blockCount(2)=4, blockCount(3)=4
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 1);

    tracker.derementBlockCount(1);
    tracker.derementBlockCount(2);  // blockCount(1)=3, blockCount(2)=3, blockCount(3)=4
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 3);

    alterBlockCount(tracker, 3, -2);
    alterBlockCount(tracker, 1, -1); // blockCount(1)=2, blockCount(2)=3, blockCount(3)=2
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 2);
}

BOOST_AUTO_TEST_CASE(BlockCountTracker_occupyAndEmptyCell)
{
    sudoku::square::Dimensions dims(2);
    sudoku::BlockCountTracker tracker(dims);

    alterBlockCount(tracker, 6, 3);
    alterBlockCount(tracker, 7, 2);
    alterBlockCount(tracker, 8, 1);
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 6);

    tracker.markCellOccupied(6);
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 7);

    tracker.markCellOccupied(7);
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 8);

    tracker.markCellEmpty(6);
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 6);
}

BOOST_AUTO_TEST_CASE(BlockCountTracker_getBlockCount)
{
    sudoku::square::Dimensions dims(2);
    sudoku::BlockCountTracker tracker(dims);
    BOOST_REQUIRE_EQUAL(tracker.getBlockCount(7), 0);
    
    alterBlockCount(tracker, 7, 3);
    BOOST_REQUIRE_EQUAL(tracker.getBlockCount(7), 3);

    alterBlockCount(tracker, 7, 1);
    BOOST_REQUIRE_EQUAL(tracker.getBlockCount(7), 4);

    alterBlockCount(tracker, 7, -2);
    BOOST_REQUIRE_EQUAL(tracker.getBlockCount(7), 2);
}

BOOST_AUTO_TEST_CASE(BlockCountTracker_rightChildGreaterInHeap)
{
    sudoku::Dimensions dims(3, 3, {});
    sudoku::BlockCountTracker tracker(dims);
    alterBlockCount(tracker, 0, 3);  //     0(=3)
    alterBlockCount(tracker, 1, 1);  //     |   |
    alterBlockCount(tracker, 2, 2);  // 1(=1)   2(=2)
    
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 0);
    tracker.markCellOccupied(0); // right child should swap up
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 2);
    tracker.markCellOccupied(2);  // left child should swap up
    BOOST_REQUIRE_EQUAL(tracker.getMaxBlockEmptyCell(), 1);
}
