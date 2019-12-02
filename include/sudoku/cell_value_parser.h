#ifndef INCLUDED_SUDOKU_CELL_VALUE_PARSER_H
#define INCLUDED_SUDOKU_CELL_VALUE_PARSER_H

#include <stdexcept>
#include <vector>

namespace sudoku
{
    class CellValueParseException : public std::runtime_error { using runtime_error::runtime_error; };

    std::vector<size_t> parseCellValues(size_t cellCount, size_t maxCellValue, const char* valueString);
}

#endif
