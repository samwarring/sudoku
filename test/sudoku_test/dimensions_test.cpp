#include <boost/test/unit_test.hpp>
#include <sudoku/dimensions.h>
#include <sudoku/groups.h>
#include <iostream>
#include "util.h"

BOOST_AUTO_TEST_CASE(Dimensions_4x4)
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
    for (sudoku::GroupCount groupNum = 0; groupNum < allGroups.size(); ++groupNum) {
        BOOST_REQUIRE_EQUAL_VECTORS(allGroups[groupNum], dims.getCellsInGroup(groupNum));
    }

    // Should have assigned group numbers to each cell
    std::vector<std::vector<sudoku::CellCount>> expectedGroupsForEachCell {
        {0, 4, 8},  {0, 5, 8},  {0, 6, 9},  {0, 7, 9},
        {1, 4, 8},  {1, 5, 8},  {1, 6, 9},  {1, 7, 9},
        {2, 4, 10}, {2, 5, 10}, {2, 6, 11}, {2, 7, 11},
        {3, 4, 10}, {3, 5, 10}, {3, 6, 11}, {3, 7, 11}
    };
    for (sudoku::CellCount cellPos = 0; cellPos < expectedGroupsForEachCell.size(); ++cellPos) {
        BOOST_REQUIRE_EQUAL_VECTORS(dims.getGroupsForCell(cellPos), expectedGroupsForEachCell[cellPos]);
    }
}

BOOST_AUTO_TEST_CASE(Dimensions_constructor_NoGroupsSuccessful)
{
    sudoku::Dimensions grouplessDims(81, 9, {});  // Does not throw.
}

BOOST_AUTO_TEST_CASE(Dimensions_constructor_GroupTooLarge)
{
    // If group sizes are less than max cell value, then dimensions
    // are valid and should not throw an exception.
    sudoku::Dimensions a(16, 4, { {0}, {1} });
    sudoku::Dimensions b(16, 4, { {0, 1}, {2, 3} });
    sudoku::Dimensions c(16, 4, { {0, 1, 2}, {3, 4, 5} });
    sudoku::Dimensions d(16, 4, { {0, 1, 2, 3}, {4, 5, 6, 7} });
    
    // These dimensions have groups with size 5. We cannot possibly
    // assign 4 unique values across a group with 5 positions.
    // This should throw an exception.
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(16, 4, { {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9} }),
        sudoku::DimensionsException
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_constructor_CellPositionOutOfRange)
{
    // The 2nd group contains position 81 which is too large an index
    // for a size-81 array (max is 80).
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(81, 9, { {0, 1, 2}, {79, 80, 81} }),
        sudoku::DimensionsException
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_constructor_CellCountIsZero)
{
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(0, 9, {}),
        sudoku::DimensionsException
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_constructor_MaxCellValueIsZero)
{
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(81, 0, {}),
        sudoku::DimensionsException
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_validateCellValues_notEnoughValues)
{
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(4, 4, {}).validateCellValues({1, 2, 3}),
        std::exception
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_validateCellValues_tooManyValues)
{
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(4, 4, {}).validateCellValues({0, 0, 0, 0, 0}),
        std::exception
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_validateCellValues_valueTooLarge)
{
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(4, 4, {}).validateCellValues({2, 3, 4, 5}),
        std::exception
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_validateCellValues_repeatedGroupValue)
{
    BOOST_REQUIRE_THROW(
        sudoku::Dimensions(4, 4, { {0, 1, 2, 3} }).validateCellValues({1, 0, 0, 1}),
        std::exception
    );
}

BOOST_AUTO_TEST_CASE(Dimensions_validateCellValues_valid)
{
    sudoku::Dimensions dims(4, 4, { {0, 1, 2, 3} });
    dims.validateCellValues({1, 2, 3, 4});
}
