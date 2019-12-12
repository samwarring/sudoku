#include <sstream>
#include <algorithm>
#include <sudoku/formatter.h>

namespace sudoku
{
    Formatter::Formatter(const sudoku::Dimensions& dims, std::string formatString)
        : dims_(dims)
        , formatString_(std::move(formatString))
    {
        // Validate the format string
        size_t numPlaceholders = std::count_if(cbegin(formatString_), cend(formatString_), isPlaceholder);
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

    bool Formatter::isPlaceholder(char ch)
    {
        switch (ch) {
            case '0':
            case '_':
            case '.':
                return true;
            default:
                return false;
        }
    }
}
