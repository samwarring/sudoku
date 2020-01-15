#include <sstream>
#include <sudoku/cell_value_parser.h>

namespace sudoku
{
    std::vector<size_t> parseCellValues(
        size_t cellCount,
        size_t maxCellValue,
        const std::string& valueString,
        const std::string& ignoreChars)
    {
        std::vector<size_t> cellValues(cellCount);
        size_t cellPos = 0;

        // Get number of digits to represent maxCellValue
        std::ostringstream ossDigits;
        ossDigits << maxCellValue;
        const size_t maxDigits = ossDigits.str().length();

        size_t curCellValue = 0;
        size_t curNumDigits = 0;
        auto addValue = [&]() {
            if (curCellValue > maxCellValue) {
                throw CellValueParseException("Parsed a cell value beyond maxCellValue");
            }
            cellValues[cellPos++] = curCellValue;
            curCellValue = 0;
            curNumDigits = 0;
        };

        for (size_t i = 0; i < valueString.length() && cellPos < cellCount; ++i) {
            char ch = valueString[i];
            if (ignoreChars.find(ch) != std::string::npos) {
                // Found an ignore character.
                if (maxDigits > 1 && curNumDigits > 0) {
                    addValue();
                }
                continue;
            }
            if (!isdigit(ch)) {
                // Found an unrecognized character.
                throw CellValueParseException("Parsed an unrecognized character");
            }
            // Found a new digit for the current cell value.
            ++curNumDigits;
            if (curNumDigits > maxDigits) {
                throw CellValueParseException("Too many digits for cell value");
            }
            curCellValue = (curCellValue * 10) + (ch - '0');
            // If single-digit values, add the value now.
            if (maxDigits == 1) {
                addValue();
            }
        }

        // If we terminated the loop by reaching end of string, we need
        // to add the final current cell value if necessary.
        if (curNumDigits > 0 && cellPos < cellCount) {
            addValue();
        }

        return cellValues;
    }

    std::vector<size_t> parseCellValues(const sudoku::Dimensions& dims, const std::string& valueString)
    {
        return parseCellValues(dims.getCellCount(), dims.getMaxCellValue(), valueString);
    }
}
