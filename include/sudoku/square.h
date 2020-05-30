#ifndef INCLUDED_SUDOKU_SQUARE_H
#define INCLUDED_SUDOKU_SQUARE_H

#include <vector>
#include <sudoku/inner_rectangular.h>

namespace sudoku
{
    /**
     * Namespace for working with generalized "square" sudokus.
     * These can be 1x1, 4x4, 9x9, 16x16, 25x25, etc. The 9x9
     * case would be the "standard" sudoku.
     */
    namespace square
    {
        class Dimensions : public sudoku::inner_rectangular::Dimensions
        {
            public:
                /**
                 * \param root the square root of the max cell value.
                 *             1 means a 1x1 sudoku; 2 means a 4x4 sudoku;
                 *             3 means a 9x9 sudoku; etc.
                 */
                Dimensions(size_t root)
                    : sudoku::inner_rectangular::Dimensions(root, root)
                    , root_(root)
                {}

                /**
                 * Get the root value of the square
                 */
                size_t getRoot() const { return root_; }

            private:
                size_t root_;
        };

        class Formatter : public sudoku::inner_rectangular::Formatter
        {
            public:
                Formatter(const Dimensions& dims)
                    : sudoku::inner_rectangular::Formatter(dims)
                {}
        };
    }
}

#endif
