#ifndef INCLUDED_SUDOKU_CELL_VALUE_PARSER_H
#define INCLUDED_SUDOKU_CELL_VALUE_PARSER_H

#include <stdexcept>
#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/types.h>

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
     * \param ignoreChars The parser ignores these characters in a value string.
     * 
     * \throw CellValueParseException if (1) valueString contains an unrecognized character,
     *                                (2) valueString contains a value beyond maxCellValue.
     * 
     * \note if maxCellValue representable by >1 digits, then values must be separated by at least
     *       one ignoreChar. If representable by 1 digit, then values can appear consecutively.
     */
    std::vector<CellValue> parseCellValues(
        CellCount cellCount,
        CellValue maxCellValue,
        const std::string& valueString,
        const std::string& ignoreChars="., \t\n\r"
    );

    std::vector<CellValue> parseCellValues(const Dimensions& dims, const std::string& valueString);
}

#endif
