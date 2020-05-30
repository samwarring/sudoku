#include <sstream>
#include <sudoku/inner_rectangular.h>
#include <sudoku/groups.h>

namespace sudoku
{
    namespace inner_rectangular
    {
        static std::string join(const std::vector<std::string>& strings, std::string sep)
        {
            std::ostringstream oss;
            for (size_t i = 0; i < strings.size(); ++i) {
                oss << strings[i];
                if (i < strings.size() - 1) {
                    oss << sep;
                }
            }
            return oss.str();
        }

        std::string Formatter::computeFormatString(const Dimensions& dims)
        {
            auto innerRowCount = dims.getInnerRowCount();
            auto innerColumnCount = dims.getInnerColumnCount();
            auto totalRows = innerRowCount * innerColumnCount;
            
            // Compute max digits for placeholder
            std::ostringstream ossDigits;
            ossDigits << static_cast<size_t>(dims.getMaxCellValue());
            const size_t maxDigits = ossDigits.str().length();
            std::string placeholder(maxDigits, '0');

            // Compue horizontal divider string
            std::string divGroup(maxDigits * innerColumnCount + (innerColumnCount - 1), '-');
            std::vector<std::string> divs(innerRowCount, divGroup);
            std::string horizontalDiv = join(divs, "-+-");

            // Compute format string for one row
            std::vector<std::string> placeholdersInGroup(innerColumnCount, placeholder);
            std::string placeholdersInGroupJoined = join(placeholdersInGroup, " ");
            std::vector<std::string> groupsInRow(innerRowCount, placeholdersInGroupJoined);
            std::string formattedRow = join(groupsInRow, " | ");

            // Compute format string for one band
            std::vector<std::string> rowsInBand(innerRowCount, formattedRow);
            std::string formattedBand = join(rowsInBand, "\n");

            // Compue format string for entire grid
            std::vector<std::string> bandsInGrid(innerColumnCount, formattedBand);
            std::string formattedGrid = join(bandsInGrid, "\n" + horizontalDiv + "\n");
            return formattedGrid;
        }
    }
}
