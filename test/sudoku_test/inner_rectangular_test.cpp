#include <boost/test/unit_test.hpp>
#include <sudoku/inner_rectangular.h>

BOOST_AUTO_TEST_CASE(RectangularDimensions_2x3)
{
    //  0  1  2 |  3  4  5
    //  6  7  8 |  9 10 11
    // ---------+---------
    // 12 13 14 | 15 16 17
    // 18 19 20 | 21 22 23
    // ---------+---------
    // 24 25 26 | 27 28 29
    // 30 31 32 | 33 34 35

    sudoku::inner_rectangular::Dimensions dims(2, 3);
    BOOST_REQUIRE_EQUAL(dims.getCellCount(), 36);
    BOOST_REQUIRE_EQUAL(dims.getMaxCellValue(), 6);
    BOOST_REQUIRE_EQUAL(dims.getNumGroups(), 18);
    
    auto groupNums = dims.getGroupsForCell(20);
    std::vector<sudoku::GroupCount> expectedGroups{ 3, 8, 14 };
    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        groupNums.begin(), groupNums.end(),
        expectedGroups.begin(), expectedGroups.end()
    );
}

BOOST_AUTO_TEST_CASE(RectangularDimensions_2x3_Formatter)
{
    sudoku::inner_rectangular::Dimensions dims(2, 3);
    sudoku::inner_rectangular::Formatter fmt(dims);
    std::vector<sudoku::CellValue> cellValues(dims.getCellCount(), 0);
    cellValues[0] = 3;
    cellValues[20] = 6;
    cellValues[33] = 4;
    auto formatted = fmt.format(cellValues);
    std::string expected =
        "3 0 0 | 0 0 0\n"
        "0 0 0 | 0 0 0\n"
        "------+------\n"
        "0 0 0 | 0 0 0\n"
        "0 0 6 | 0 0 0\n"
        "------+------\n"
        "0 0 0 | 0 0 0\n"
        "0 0 0 | 4 0 0";
    BOOST_REQUIRE_EQUAL(formatted, expected);
}