#include <sstream>
#include <algorithm>
#include <iomanip>
#include <sudoku/formatter.h>

namespace sudoku
{
    Formatter::Formatter(const sudoku::Dimensions& dims, std::string formatString, std::string placeholders)
        : dims_(dims)
        , formatString_(std::move(formatString))
        , placeholders_(placeholders)
    {
        // Number of digits required to represent the max cell value.
        std::ostringstream sout;
        sout << dims_.getMaxCellValue();
        maxDigits_ = sout.str().length();

        // Validate the format string
        size_t numPlaceholders = 0;
        for(size_t i = 0; i < formatString_.length(); ++i) {
            
            // If char is placeholder, then following maxDigits-1 chars should
            // also be placeholder.
            if (isPlaceholder(formatString_[i])) {
                for (int j = 1; j < maxDigits_ && i + j < formatString_.length(); ++j) {
                    if (!isPlaceholder(formatString_[i + j])) {
                        throw FormatterException("Encountered format string with invalid placeholder");
                    }
                }

                // Found required number of placeholder chars for a single cell.
                i += (maxDigits_ - 1);
                numPlaceholders++;
            }
        }
        if (numPlaceholders != dims_.getCellCount()) {
            throw FormatterException("Formatter dimensions do not match formatString");
        }
    }

    std::string Formatter::format(const std::vector<size_t>& cellValues) const
    {
        if (cellValues.size() != dims_.getCellCount()) {
            throw FormatterException("Number of cell values unequal to formatter dimensions");
        }

        std::ostringstream sout;
        size_t cellValuePos = 0;
        for (size_t i = 0; i < formatString_.length(); ++i) {
            char ch = formatString_[i];
            if (isPlaceholder(ch)) {
                sout << std::setw(maxDigits_) << std::right << cellValues[cellValuePos++];
                i += (maxDigits_ - 1);
            }
            else {
                sout << ch;
            }
        }
        return sout.str();
    }

    bool Formatter::isPlaceholder(char ch) const
    {
        return placeholders_.find(ch) != std::string::npos;
    }
}
