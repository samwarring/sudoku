#include <algorithm>
#include <set>
#include <boost/test/unit_test.hpp>
#include <sudoku/partition_table.h>
#include <sudoku/standard.h>

BOOST_AUTO_TEST_CASE(PartitionTable_getPartitionSizes)
{
    sudoku::standard::Dimensions dims;
    auto partitionIds = sudoku::partitionRoundRobin(dims.getCellCount(), 4);
    sudoku::PartitionTable partitionTable(dims, 4, partitionIds);

    std::vector<sudoku::CellCount> partitionSizes{
        partitionTable.getPartitionSize(0),
        partitionTable.getPartitionSize(1),
        partitionTable.getPartitionSize(2),
        partitionTable.getPartitionSize(3),
    };
    std::sort(partitionSizes.begin(), partitionSizes.end());
    std::vector<sudoku::CellCount> expectedSizes{ 20, 20, 20, 21 };
    
    BOOST_REQUIRE_EQUAL_COLLECTIONS(expectedSizes.begin(), expectedSizes.end(),
                                    partitionSizes.begin(), partitionSizes.end());
}

BOOST_AUTO_TEST_CASE(PartitionTable_getPartitionId)
{
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::PartitionCount> partitionIds{
        0, 0, 1, 1,
        0, 0, 1, 1,
        2, 2, 3, 3,
        2, 2, 3, 3
    };
    sudoku::PartitionTable partitionTable(dims, 4, partitionIds);
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(0), 0);
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(6), 1);
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(9), 2);
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(15), 3);
}

BOOST_AUTO_TEST_CASE(PartitionTable_getRelatedIndicesForPartition)
{
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::PartitionCount> partitionIds{
        0, 0, 1, 1,
        0, 0, 1, 1,
        2, 2, 3, 3,
        2, 2, 3, 3 // <-- test corner cell
    };
    sudoku::PartitionTable partitionTable(dims, 4, partitionIds);

    auto related0 = partitionTable.getRelatedIndicesForPartition(0, 15);
    auto related1 = partitionTable.getRelatedIndicesForPartition(1, 15);
    auto related2 = partitionTable.getRelatedIndicesForPartition(2, 15);
    auto related3 = partitionTable.getRelatedIndicesForPartition(3, 15);

    // Cell 15 has no related cells in partition 0.
    BOOST_CHECK_EQUAL(related0.size(), 0);

    // Cell 15 has 2 related cells each in partitions 1 and 2.
    BOOST_CHECK_EQUAL(related1.size(), 2);
    BOOST_CHECK_EQUAL(related2.size(), 2);

    // Cell 15 is related to all cells in partition 3 (except itself)
    BOOST_CHECK_EQUAL(related3.size(), 3);
}

BOOST_AUTO_TEST_CASE(PartitionTable_getCellPosition)
{
    sudoku::square::Dimensions dims(4);
    auto partitionIds = sudoku::partitionRRTrail(dims.getCellCount(), 5);
    sudoku::PartitionTable partitionTable(dims, 5, partitionIds);

    auto testPosition = [&](auto cellPos) {
        auto partitionId = partitionTable.getPartitionId(cellPos);
        auto partitionIndex = partitionTable.getPartitionIndex(cellPos);
        BOOST_REQUIRE_EQUAL(cellPos, partitionTable.getCellPosition(partitionId, partitionIndex));
    };

    testPosition(0);
    testPosition(42);
    testPosition(123);
    testPosition(7);
    testPosition(255);
}

BOOST_AUTO_TEST_CASE(PartitionTable_getPartitionIndex)
{
    std::vector<sudoku::PartitionCount> partitionIds{
        0, 0, 0, 1,
        1, 1, 2, 2,
        2, 1, 3, 2,
        3, 3, 0, 3
    };
    sudoku::square::Dimensions dims(2);
    sudoku::PartitionTable partitionTable(dims, 4, partitionIds);

    // Check that partition 1 covers positions 3,4,5,9
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(3), 1);
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(4), 1);
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(5), 1);
    BOOST_REQUIRE_EQUAL(partitionTable.getPartitionId(9), 1);

    // Check that indices of 3,4,5,9 cover values 0,1,2,3
    std::set<sudoku::CellCount> partitionIndices;
    partitionIndices.insert(partitionTable.getPartitionIndex(3));
    partitionIndices.insert(partitionTable.getPartitionIndex(4));
    partitionIndices.insert(partitionTable.getPartitionIndex(5));
    partitionIndices.insert(partitionTable.getPartitionIndex(9));
    std::set<sudoku::CellCount> expected{0, 1, 2, 3};
    BOOST_REQUIRE_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                    partitionIndices.begin(), partitionIndices.end());
}