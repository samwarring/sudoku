#ifndef INCLUDED_SUDOKU_FORMATTER_H
#define INCLUDED_SUDOKU_FORMATTER_H

#include <stdexcept>
#include <string>
#include <vector>
#include <sudoku/dimensions.h>

namespace sudoku
{
    /**
     * Exception raised when the \ref Formatter class is passed invalid data
     */
    class FormatterException : public std::logic_error { using logic_error::logic_error; };

    /**
     * Associates sudoku dimensions with a format string.
     */
    class Formatter
    {
        public:
            /**
             * \param dims Dimensions of the sudoku
             * \param formatString String representation of the formatted sudoku.
             *                     '0', '.', and '_' characters are treated as
             *                     placeholders for sudoku values.
             * \throw FormatterError if the number of placeholders in formatString
             *        does not match the dimensions' cell count.
             */
            Formatter(const Dimensions& dims, std::string formatString);

            /**
             * Formats the cell values as a string.
             * 
             * \throw FormatterError if size of cellValues does not match the
             *        dimensions' cell count.
             */
            std::string format(const std::vector<size_t>& cellValues) const;

        private:
            static bool isPlaceholder(char ch);

            const Dimensions& dims_;
            const std::string formatString_;
    };
}

#endif
