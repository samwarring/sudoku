#include <sudoku/cell_value_parser.h>

namespace sudoku
{
    std::vector<size_t> parseCellValues(size_t cellCount, size_t maxCellValue, const char* valueString)
    {
        if (0xf < maxCellValue) {
            throw CellValueParseException("Cannot parse cell values with more than 15 symbols (1-f)");
        }
        
        if (!valueString) {
            throw CellValueParseException("Cannot parse a null value string");
        }

        std::vector<size_t> cellValues(cellCount);
        size_t cellPos = 0;
        for (const char* ch = valueString; *ch != 0 && cellPos < cellCount; ++ch) {
            switch (*ch) {
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    cellValues[cellPos++] = (*ch - '0');
                    break;
                case 'a':
                case 'b':
                case 'c':
                case 'd':
                case 'e':
                case 'f':
                    cellValues[cellPos++] = (*ch - 'a' + 10);
                    break;
                case ' ':
                case '.':
                case ',':
                case '\t':
                case '\n':
                case '\r':
                    continue; // loop to next character
                default:
                    throw CellValueParseException("Unrecognized character in cell value string");
            }
            if (cellValues[cellPos - 1] > maxCellValue) {
                throw CellValueParseException("Parsed a cell value beyond maxCellValue");
            }
        }

        return cellValues;
    }

    std::vector<size_t> parseCellValues(const sudoku::Dimensions& dims, const char* valueString)
    {
        return parseCellValues(dims.getCellCount(), dims.getMaxCellValue(), valueString);
    }
}
