#ifndef INCLUDED_SUDOKU_CUDA_UTIL_H
#define INCLUDED_SUDOKU_CUDA_UTIL_H

#include <cmath>

namespace sudoku
{
    namespace cuda
    {
        template <typename T>
        bool isPowerOf2(T value)
        {
            return value > 0 && (value & (value - 1)) == 0;
        }

        template <typename T>
        T nearestPowerOf2(T value)
        {
            return static_cast<T>(pow(2, ceil( log(value)/log(2) )));
        }
    }
}

#endif