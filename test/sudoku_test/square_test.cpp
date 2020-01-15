#include <vector>
#include <boost/test/unit_test.hpp>
#include <sudoku/square.h>

BOOST_AUTO_TEST_CASE(SquareFormatter_Root1)
{
    sudoku::square::Dimensions dims(1);
    sudoku::square::Formatter fmt(dims);
    std::vector<size_t> cellValues{ 1 };
    std::string expected = "1";
    std::string actual = fmt.format(cellValues);
    BOOST_REQUIRE_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_CASE(SquareFormatter_Root2)
{
    sudoku::square::Dimensions dims(2);
    sudoku::square::Formatter fmt(dims);
    std::vector<size_t> cellValues{ 
        1, 2, 3, 4,
        4, 3, 2, 1,
        3, 4, 1, 2,
        0, 0, 0, 0
    };
    std::string expected = 
        "1 2 | 3 4\n"
        "4 3 | 2 1\n"
        "----+----\n"
        "3 4 | 1 2\n"
        "0 0 | 0 0";
    std::string actual = fmt.format(cellValues);
    BOOST_REQUIRE_EQUAL(expected, actual);
}

BOOST_AUTO_TEST_CASE(SquareFormatter_Root3)
{
    sudoku::square::Dimensions dims(3);
    sudoku::square::Formatter fmt(dims);
    std::vector<size_t> cellValues(81, 0);
    std::string expected = 
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "------+-------+------\n"
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "------+-------+------\n"
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "0 0 0 | 0 0 0 | 0 0 0\n"
        "0 0 0 | 0 0 0 | 0 0 0";
    std::string actual = fmt.format(cellValues);
    BOOST_REQUIRE_EQUAL(expected, actual);
}
