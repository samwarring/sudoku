#include <boost/test/unit_test.hpp>
#include <sudoku/cell_value_parser.h>
#include "util.h"

BOOST_AUTO_TEST_CASE(parseCellValues_emptyString)
{
    auto cellValues = sudoku::parseCellValues(81, 9, "");
    std::vector<size_t> expected(81);
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
}

BOOST_AUTO_TEST_CASE(parseCellValues_remainingEmpty)
{
    auto cellValues = sudoku::parseCellValues(81, 9, "0 1 2");
    std::vector<size_t> expected{ 0, 1, 2 };
    expected.resize(81);
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
}

BOOST_AUTO_TEST_CASE(parseCellValues_valueOutOfRange)
{
    BOOST_REQUIRE_THROW(
        sudoku::parseCellValues(81, 4, "0 1 2 3 4 5"),
        sudoku::CellValueParseException
    );
}

BOOST_AUTO_TEST_CASE(parseCellValues_ignoreChars)
{
    auto cellValues = sudoku::parseCellValues(2, 1, "0 .,.,.\n\t\r\n 1");
    std::vector<size_t> expected({0, 1});
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
}

BOOST_AUTO_TEST_CASE(parseCellValues_ignoreTrailingChars)
{
    auto cellValues = sudoku::parseCellValues(3, 3, "1 2 3XYZ");
    std::vector<size_t> expected({1, 2, 3});
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
}

BOOST_AUTO_TEST_CASE(parseCellValues_noSpace)
{
    auto cellValues = sudoku::parseCellValues(5, 9, "8910");
    std::vector<size_t> expected{8, 9, 1, 0, 0};
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
}

BOOST_AUTO_TEST_CASE(parseCellValues_doubleDigits)
{
    sudoku::Dimensions dims(10, 10, {});
    auto cellValues = sudoku::parseCellValues(dims, "10 9 8 0 1 10 3");
    std::vector<size_t> expected{10, 9, 8, 0, 1, 10, 3, 0, 0, 0};
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
}

BOOST_AUTO_TEST_CASE(parseCellValues_unrecognizedChar)
{
    BOOST_REQUIRE_THROW(
        sudoku::parseCellValues(4, 4, "1 2 3 % 4"),
        sudoku::CellValueParseException
    );
}
