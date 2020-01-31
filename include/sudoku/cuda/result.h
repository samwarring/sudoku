#ifndef INCLUDED_SUDOKU_CUDA_RESULT_H
#define INCLUDED_SUDOKU_CUDA_RESULT_H

#include <iostream>
#include <string>

namespace sudoku
{
    namespace cuda
    {
        enum class Result : int
        {
            ERROR_INVALID_ARG = -1,
            ERROR_NOT_SET = 0,
            OK_FOUND_SOLUTION = 1,
            OK_TIMED_OUT = 2,
            OK_NO_SOLUTION = 3,
        };

        std::string toString(Result result);
    }
}

namespace std
{
    ostream& operator<<(ostream& out, sudoku::cuda::Result result);
}

#endif
