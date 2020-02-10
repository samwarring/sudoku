#include <boost/test/unit_test.hpp>
#include <sudoku/groups.h>
#include "util.h"

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
    BOOST_REQUIRE_EQUAL_VECTORS(rowGroups[0], {0});
}

BOOST_AUTO_TEST_CASE(computeRowGroups_2x2)
{
    auto rowGroups = sudoku::computeRowGroups(2, 2);
    BOOST_REQUIRE_EQUAL(rowGroups.size(), 2);
    BOOST_REQUIRE_EQUAL_VECTORS(rowGroups[0], {0, 1});
    BOOST_REQUIRE_EQUAL_VECTORS(rowGroups[1], {2, 3});
}

BOOST_AUTO_TEST_CASE(computeRowGroups_3x4)
{
    auto rowGroups = sudoku::computeRowGroups(3, 4);
    BOOST_REQUIRE_EQUAL(rowGroups.size(), 3);
    BOOST_REQUIRE_EQUAL_VECTORS(rowGroups[0], {0, 1, 2, 3});
    BOOST_REQUIRE_EQUAL_VECTORS(rowGroups[1], {4, 5, 6, 7});
    BOOST_REQUIRE_EQUAL_VECTORS(rowGroups[2], {8, 9, 10, 11});
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
    BOOST_REQUIRE_EQUAL_VECTORS(columnGroups[0], {0});
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_2x2)
{
    auto columnGroups = sudoku::computeColumnGroups(2, 2);
    BOOST_REQUIRE_EQUAL(columnGroups.size(), 2);
    BOOST_REQUIRE_EQUAL_VECTORS(columnGroups[0], {0, 2});
    BOOST_REQUIRE_EQUAL_VECTORS(columnGroups[1], {1, 3});
}

BOOST_AUTO_TEST_CASE(computeColumnGroups_3x4)
{
    auto columnGroups = sudoku::computeColumnGroups(3, 4);
    BOOST_REQUIRE_EQUAL(columnGroups.size(), 4);
    BOOST_REQUIRE_EQUAL_VECTORS(columnGroups[0], {0, 4, 8});
    BOOST_REQUIRE_EQUAL_VECTORS(columnGroups[1], {1, 5, 9});
    BOOST_REQUIRE_EQUAL_VECTORS(columnGroups[2], {2, 6, 10});
    BOOST_REQUIRE_EQUAL_VECTORS(columnGroups[3], {3, 7, 11});
}

BOOST_AUTO_TEST_CASE(computeSquareGroups_9x9)
{
    auto squareGroups = sudoku::computeSquareGroups(3);
    BOOST_REQUIRE_EQUAL(squareGroups.size(), 9);
    BOOST_REQUIRE_EQUAL_VECTORS(squareGroups[0], {0, 1, 2, 9, 10, 11, 18, 19, 20});
    BOOST_REQUIRE_EQUAL_VECTORS(squareGroups[5], {33, 34, 35, 42, 43, 44, 51, 52, 53});
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
    BOOST_REQUIRE_EQUAL_VECTORS(groups[0], {0, 1, 6, 7, 12, 13, 14, 15, 18, 19});
    BOOST_REQUIRE_EQUAL_VECTORS(groups[1], {2, 3, 4, 8, 9, 10, 16, 20, 21, 22});
    BOOST_REQUIRE_EQUAL_VECTORS(groups[2], {5, 11, 17, 23});
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
    std::vector<std::vector<sudoku::CellCount>> groups1 = {  // 2 groups
        {0, 1, 2, 3},
        {4, 5, 6, 7}
    };
    std::vector<std::vector<sudoku::CellCount>> groups2 = {  // 4 groups
        {0, 4},
        {1, 5},
        {2, 6},
        {3, 7}
    };
    auto allGroups = sudoku::joinGroups({groups1, groups2});  // 2 + 4 = 6 groups
    BOOST_REQUIRE_EQUAL(allGroups.size(), 6);
    BOOST_REQUIRE_EQUAL_VECTORS(allGroups[0], {0, 1, 2, 3});
    BOOST_REQUIRE_EQUAL_VECTORS(allGroups[1], {4, 5, 6, 7});
    BOOST_REQUIRE_EQUAL_VECTORS(allGroups[2], {0, 4});
    BOOST_REQUIRE_EQUAL_VECTORS(allGroups[3], {1, 5});
    BOOST_REQUIRE_EQUAL_VECTORS(allGroups[4], {2, 6});
    BOOST_REQUIRE_EQUAL_VECTORS(allGroups[5], {3, 7});
}
