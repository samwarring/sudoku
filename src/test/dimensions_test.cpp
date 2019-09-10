#include <boost/test/unit_test.hpp>
#include <sudoku/dimensions.h>

void requireEqualVector(const std::vector<size_t>& actual, const std::vector<size_t>& expected)
{
    BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        BOOST_REQUIRE_EQUAL(actual[i], expected[i]);
    }
}

BOOST_AUTO_TEST_CASE(computeRowGroups_0x0)
{
    auto rowGroups = sudoku::computeRowGroups(0, 0);
    BOOST_REQUIRE(rowGroups.empty());
}

BOOST_AUTO_TEST_CASE(computeRowGroups_0x5)
{
    auto rowGroups = sudoku::computeRowGroups(0, 5);
    BOOST_REQUIRE(rowGroups.empty());
}

BOOST_AUTO_TEST_CASE(computeRowGroups_5x0)
{
    auto rowGroups = sudoku::computeRowGroups(5, 0);
    BOOST_REQUIRE(rowGroups.empty());
}

BOOST_AUTO_TEST_CASE(computeRowGroups_1x1)
{
    auto rowGroups = sudoku::computeRowGroups(1, 1);
    BOOST_REQUIRE_EQUAL(rowGroups.size(), 1);
    requireEqualVector(rowGroups[0], {0});
}

BOOST_AUTO_TEST_CASE(computeRowGroups_2x2)
{
    auto rowGroups = sudoku::computeRowGroups(2, 2);
    BOOST_REQUIRE_EQUAL(rowGroups.size(), 2);
    requireEqualVector(rowGroups[0], {0, 1});
    requireEqualVector(rowGroups[1], {2, 3});
}

BOOST_AUTO_TEST_CASE(computeRowGroups_3x4)
{
    auto rowGroups = sudoku::computeRowGroups(3, 4);
    BOOST_REQUIRE_EQUAL(rowGroups.size(), 3);
    requireEqualVector(rowGroups[0], {0, 1, 2, 3});
    requireEqualVector(rowGroups[1], {4, 5, 6, 7});
    requireEqualVector(rowGroups[2], {8, 9, 10, 11});
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_0x0)
{
    auto columnGroups = sudoku::computeColumnGroups(0, 0);
    BOOST_REQUIRE(columnGroups.empty());
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_0x5)
{
    auto columnGroups = sudoku::computeColumnGroups(0, 5);
    BOOST_REQUIRE(columnGroups.empty());
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_5x0)
{
    auto columnGroups = sudoku::computeColumnGroups(5, 0);
    BOOST_REQUIRE(columnGroups.empty());
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_1x1)
{
    auto columnGroups = sudoku::computeColumnGroups(1, 1);
    BOOST_REQUIRE_EQUAL(columnGroups.size(), 1);
    requireEqualVector(columnGroups[0], {0});
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_2x2)
{
    auto columnGroups = sudoku::computeColumnGroups(2, 2);
    BOOST_REQUIRE_EQUAL(columnGroups.size(), 2);
    requireEqualVector(columnGroups[0], {0, 2});
    requireEqualVector(columnGroups[1], {1, 3});
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_3x4)
{
    auto columnGroups = sudoku::computeColumnGroups(3, 4);
    BOOST_REQUIRE_EQUAL(columnGroups.size(), 4);
    requireEqualVector(columnGroups[0], {0, 4, 8});
    requireEqualVector(columnGroups[1], {1, 5, 9});
    requireEqualVector(columnGroups[2], {2, 6, 10});
    requireEqualVector(columnGroups[3], {3, 7, 11});
}

BOOST_AUTO_TEST_CASE(computeGroupsFromMap_emptyString)
{
    auto groups = sudoku::computeGroupsFromMap("");
    BOOST_REQUIRE_EQUAL(groups.size(), 0);
}

BOOST_AUTO_TEST_CASE(computeGroupsFromMap_simple)
{
    auto groups = sudoku::computeGroupsFromMap(
        "0 0 1 1 1 2 "
        "0 0 1 1 1 2 "
        "0 0 0 0 1 2 "
        "0 0 1 1 1 2 "
    );
    BOOST_REQUIRE_EQUAL(groups.size(), 3);
    requireEqualVector(groups[0], {0, 1, 6, 7, 12, 13, 14, 15, 18, 19});
    requireEqualVector(groups[1], {2, 3, 4, 8, 9, 10, 16, 20, 21, 22});
    requireEqualVector(groups[2], {5, 11, 17, 23});
}

BOOST_AUTO_TEST_CASE(joinGroups_empty)
{
    auto groups = sudoku::joinGroups({});
    BOOST_REQUIRE(groups.empty());
}

BOOST_AUTO_TEST_CASE(joinGroups_emptyVectors)
{
    auto groups = sudoku::joinGroups({ {}, {}, {} });
    BOOST_REQUIRE(groups.empty());
}

BOOST_AUTO_TEST_CASE(joinGroups_simple)
{
    std::vector<std::vector<size_t>> groups1 = {  // 2 groups
        {0, 1, 2, 3},
        {4, 5, 6, 7}
    };
    std::vector<std::vector<size_t>> groups2 = {  // 4 groups
        {0, 4},
        {1, 5},
        {2, 6},
        {3, 7}
    };
    auto allGroups = sudoku::joinGroups({groups1, groups2});  // 2 + 4 = 6 groups
    BOOST_REQUIRE_EQUAL(allGroups.size(), 6);
    requireEqualVector(allGroups[0], {0, 1, 2, 3});
    requireEqualVector(allGroups[1], {4, 5, 6, 7});
    requireEqualVector(allGroups[2], {0, 4});
    requireEqualVector(allGroups[3], {1, 5});
    requireEqualVector(allGroups[4], {2, 6});
    requireEqualVector(allGroups[5], {3, 7});
}

BOOST_AUTO_TEST_CASE(dimensions_4x4)
{
    auto rowGroups = sudoku::computeRowGroups(4, 4);        // Groups 0-3
    auto columnGroups = sudoku::computeColumnGroups(4, 4);  // Groups 4-7
    auto squareGroups = sudoku::computeGroupsFromMap(       // Groups 8-11
        " 0 0 1 1 "
        " 0 0 1 1 "
        " 2 2 3 3 "
        " 2 2 3 3 "
    );
    auto allGroups = sudoku::joinGroups({rowGroups, columnGroups, squareGroups});
    auto dims = sudoku::Dimensions(16, 4, allGroups);

    // Verify sudoku limits
    BOOST_REQUIRE_EQUAL(dims.getCellCount(), 16);
    BOOST_REQUIRE_EQUAL(dims.getMaxCellValue(), 4);
    
    // Groups should be the same as what was passed to constructor
    for (size_t groupNum = 0; groupNum < allGroups.size(); ++groupNum) {
        requireEqualVector(allGroups[groupNum], dims.getCellsInGroup(groupNum));
    }

    // Should have assigned group numbers to each cell
    std::vector<std::vector<size_t>> expectedGroupsForEachCell {
        {0, 4, 8},  {0, 5, 8},  {0, 6, 9},  {0, 7, 9},
        {1, 4, 8},  {1, 5, 8},  {1, 6, 9},  {1, 7, 9},
        {2, 4, 10}, {2, 5, 10}, {2, 6, 11}, {2, 7, 11},
        {3, 4, 10}, {3, 5, 10}, {3, 6, 11}, {3, 7, 11}
    };
    for (size_t cellPos = 0; cellPos < expectedGroupsForEachCell.size(); ++cellPos) {
        requireEqualVector(dims.getGroupsForCell(cellPos), expectedGroupsForEachCell[cellPos]);
    }
}
