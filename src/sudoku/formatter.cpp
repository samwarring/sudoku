#include <sstream>
#include <algorithm>
#include <sudoku/formatter.h>

namespace sudoku
{
    Formatter::Formatter(const sudoku::Dimensions& dims, std::string formatString, std::string placeholders)
        : dims_(dims)
        , formatString_(std::move(formatString))
        , placeholders_(placeholders)
    {
        // Validate the format string
        size_t numPlaceholders = 0;
        for(size_t i = 0; i < formatString_.length(); ++i) {
            if (isPlaceholder(formatString_[i])) {
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
                sout << cellValues[cellValuePos++];
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
