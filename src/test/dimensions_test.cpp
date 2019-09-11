#include <boost/test/unit_test.hpp>
#include <sudoku/dimensions.h>
#include <sudoku/groups.h>
#include <iostream>
#include "util.h"

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
        BOOST_REQUIRE_EQUAL_VECTORS(allGroups[groupNum], dims.getCellsInGroup(groupNum));
    }

    // Should have assigned group numbers to each cell
    std::vector<std::vector<size_t>> expectedGroupsForEachCell {
        {0, 4, 8},  {0, 5, 8},  {0, 6, 9},  {0, 7, 9},
        {1, 4, 8},  {1, 5, 8},  {1, 6, 9},  {1, 7, 9},
        {2, 4, 10}, {2, 5, 10}, {2, 6, 11}, {2, 7, 11},
        {3, 4, 10}, {3, 5, 10}, {3, 6, 11}, {3, 7, 11}
    };
    for (size_t cellPos = 0; cellPos < expectedGroupsForEachCell.size(); ++cellPos) {
        BOOST_REQUIRE_EQUAL_VECTORS(dims.getGroupsForCell(cellPos), expectedGroupsForEachCell[cellPos]);
    }
}
