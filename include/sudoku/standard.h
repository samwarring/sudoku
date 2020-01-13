#ifndef INCLUDED_SUDOKU_STANDARD_H
#define INCLUDED_SUDOKU_STANDARD_H

#include <vector>
#include <sudoku/square.h>

namespace sudoku
{
    /**
     * This namespace defines specializations of \ref sudoku classes which
     * are specific to solving standard 9x9 sudokus.
     */
    namespace standard
    {
        class Dimensions : public square::Dimensions
        {
            public:
                Dimensions() : square::Dimensions(3) {}
        };
    }
}

#endif
