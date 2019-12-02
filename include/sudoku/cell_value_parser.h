#ifndef INCLUDED_SUDOKU_CELL_VALUE_PARSER_H
#define INCLUDED_SUDOKU_CELL_VALUE_PARSER_H

#include <stdexcept>
#include <vector>

namespace sudoku
{
    class CellValueParseException : public std::runtime_error { using runtime_error::runtime_error; };

    /**
     * Parse a vector of cell values from a C-string.
     * 
     * \param cellCount Maximum number of cells to parse. Once this function parses `cellCount`
     *                  characters, the rest of the `valueString` is ignored. If `valueString`
     *                  contains less than `cellCount` values, then the remaining values are
     *                  set to 0.
     * 
     * \param maxCellValue Maximum cell value allowed in the sudoku. If `valueString` contains
     *                     a value beyond `maxCellValue`, this function throws an exception. This
     *                     function supports maxCellValues up to 15 using characters [0-9a-f].
     * 
     * \param valueString The string to parse cell values from. Whitespace characters, periods (.),
     *                    and commas (,) are ignored.
     * 
     * \throw CellValueParseException if (1) valueString is NULL, (2) valueString contains an
     *                                unrecognized character, (3) valueString contains a value
     *                                beyond maxCellValue, (4) maxCellValue > 15.
     */
    std::vector<size_t> parseCellValues(size_t cellCount, size_t maxCellValue, const char* valueString);
}

#endif
