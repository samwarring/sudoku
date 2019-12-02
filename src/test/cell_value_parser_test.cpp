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

BOOST_AUTO_TEST_CASE(parseCellValues_hexValues)
{
    auto cellValues = sudoku::parseCellValues(16, 15, "0 1 2 3 4 5 6 7 8 9 a b c d e f");
    std::vector<size_t> expected{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
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
    auto cellValues = sudoku::parseCellValues(5, 10, "8 9 10 a");
    std::vector<size_t> expected{8, 9, 1, 0, 10};
    BOOST_REQUIRE_EQUAL_VECTORS(cellValues, expected);
}

BOOST_AUTO_TEST_CASE(parseCellValues_nullString)
{
    BOOST_REQUIRE_THROW(
        sudoku::parseCellValues(81, 9, nullptr),
        sudoku::CellValueParseException
    );
}

BOOST_AUTO_TEST_CASE(parseCellValues_unrecognizedChar)
{
    BOOST_REQUIRE_THROW(
        sudoku::parseCellValues(4, 4, "1 2 3 % 4"),
        sudoku::CellValueParseException
    );
}

BOOST_AUTO_TEST_CASE(parseCellValues_maxCellValueTooLarge)
{
    BOOST_REQUIRE_THROW(
        sudoku::parseCellValues(256, 16, ""),
        sudoku::CellValueParseException
    );
}