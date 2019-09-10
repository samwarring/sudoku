#ifndef INCLUDED_SUDOKU_STANDARD_H
#define INCLUDED_SUDOKU_STANDARD_H

#include <vector>
#include <sudoku/dimensions.h>


namespace sudoku
{
    /**
     * This namespace defines specializations of \ref sudoku classes which
     * are specific to solving standard 9x9 sudokus.
     */
    namespace standard
    {
        class Dimensions : public sudoku::Dimensions
        {
            public:

                Dimensions() : sudoku::Dimensions(81, 9, computeStandardGroups()) {}

                static std::vector<std::vector<size_t>> computeStandardGroups();
        };
    }
}

#endif
