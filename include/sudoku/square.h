#ifndef INCLUDED_SUDOKU_SQUARE_H
#define INCLUDED_SUDOKU_SQUARE_H

#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/groups.h>
#include <sudoku/formatter.h>

namespace sudoku
{
    /**
     * Namespace for working with generalized "square" sudokus.
     * These can be 1x1, 4x4, 9x9, 16x16, 25x25, etc. The 9x9
     * case would be the "standard" sudoku.
     */
    namespace square
    {
        class Dimensions : public sudoku::Dimensions
        {
            public:
                /**
                 * \param root the square root of the max cell value.
                 *             1 means a 1x1 sudoku; 2 means a 4x4 sudoku;
                 *             3 means a 9x9 sudoku; etc.
                 */
                Dimensions(size_t root)
                    : sudoku::Dimensions(
                        root * root * root * root,
                        root * root,
                        joinGroups({
                            computeRowGroups(root * root, root * root),
                            computeColumnGroups(root * root, root * root),
                            computeSquareGroups(root)
                        }))
                    , root_(root)
                {}

                /**
                 * Get the root value of the square
                 */
                size_t getRoot() const { return root_; }

            private:
                size_t root_;
        };

        class Formatter : public sudoku::Formatter
        {
            public:
                Formatter(const Dimensions& dims)
                    : sudoku::Formatter(
                        dims,
                        computeFormatString(dims)
                    )
                {}

            private:
                static std::string computeFormatString(const Dimensions& dims);
        };
    }
}

#endif
