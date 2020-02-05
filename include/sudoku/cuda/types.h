#ifndef INCLUDED_SUDOKU_CUDA_TYPES_H
#define INCLUDED_SUDOKU_CUDA_TYPES_H

namespace sudoku
{
    namespace cuda
    {
        using CellCount = unsigned;
        using CellValue = unsigned;
        using CellBlockCount = int;
        using ValueBlockCount = unsigned;
        using GroupCount = unsigned;
        struct Guess
        {
            CellCount cellPos;
            CellValue cellValue;
        };
        enum class Result : int
        {
            NO_SOLUTION,
            FOUND_SOLUTION,
            TIMED_OUT
        };
    }
}

#endif
