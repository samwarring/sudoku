#ifndef INCLUDED_SUDOKU_INNER_RECTANGULAR_H
#define INCLUDED_SUDOKU_INNER_RECTANGULAR_H

#include <sudoku/dimensions.h>
#include <sudoku/formatter.h>
#include <sudoku/groups.h>

namespace sudoku
{
    namespace inner_rectangular
    {
        class Dimensions : public sudoku::Dimensions
        {
            public:

                Dimensions(CellCount innerRowCount, CellCount innerColumnCount)
                    : sudoku::Dimensions(
                        innerRowCount * innerRowCount * innerColumnCount * innerColumnCount,
                        innerRowCount * innerColumnCount,
                        joinGroups({
                            computeRowGroups(innerRowCount * innerColumnCount,
                                            innerRowCount * innerColumnCount),
                            computeColumnGroups(innerRowCount * innerColumnCount,
                                                innerRowCount * innerColumnCount),
                            computeInnerRectangularGroups(innerRowCount, innerColumnCount)
                        })
                    ),
                    innerRowCount_(innerRowCount),
                    innerColumnCount_(innerColumnCount)
                {}

                CellCount getInnerRowCount() const { return innerRowCount_; }

                CellCount getInnerColumnCount() const { return innerColumnCount_; }

            private:

                CellCount innerRowCount_;
                CellCount innerColumnCount_;
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