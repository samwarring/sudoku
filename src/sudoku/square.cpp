#include <string>
#include <sstream>
#include <sudoku/square.h>
#include <sudoku/dimensions.h>

namespace sudoku
{
    namespace square
    {
        std::string Formatter::computeFormatString(const Dimensions& dims)
        {
            const size_t root = dims.getRoot();
            std::ostringstream ossResult;

            // Compute max digits for placeholder
            std::ostringstream ossDigits;
            ossDigits << dims.getMaxCellValue();
            const size_t maxDigits = ossDigits.str().length();

            // Compute placeholder string from max digits
            const std::string placeholder(maxDigits, '0');

            // Compute horizontal separator string to be drawn
            // between each subsquare-row
            std::ostringstream ossHorizontalSep;
            for (size_t cellCol = 0; cellCol < root * root; ++cellCol) {
                // Cover the placeholder
                for (size_t digitNum = 0; digitNum < maxDigits; ++digitNum) {
                    ossHorizontalSep << '-';
                }
                // If not the last column, cover the space
                if (cellCol < root * root - 1) {
                    ossHorizontalSep << '-';

                    // If starting a new square, cover the vertical separator
                    if (cellCol % root == root - 1) {
                        ossHorizontalSep << "+-";
                    }
                }
            }
            const std::string horizontalSep = ossHorizontalSep.str();
            
            // Iterate through each cell by (row, col)
            for (size_t cellRow = 0; cellRow < root * root; ++cellRow) {
                for (size_t cellCol = 0; cellCol < root * root; ++cellCol) {
                    ossResult << placeholder;

                    // Separate consecutive columns by a space (unless last column in row)
                    if (cellCol < (root * root - 1)) {
                        ossResult << ' ';

                        // If starting a new square column, insert vertical separator.
                        if (cellCol % root == root - 1) {
                            ossResult << "| ";
                        }
                    }
                }

                // Separate rows by newline (unless this is the last row)
                if (cellRow < (root * root - 1)) {
                    ossResult << '\n';

                    // If about to start a new square-row, insert horizontal separator.
                    if (cellRow % root == (root - 1)) {
                        ossResult << horizontalSep << '\n';
                    }
                }
            }

            return ossResult.str();
        }
    }
}
