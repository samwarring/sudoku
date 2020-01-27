#ifndef INCLUDED_SUDOKU_CUDA_KERNEL_H
#define INCLUDED_SUDOKU_CUDA_KERNEL_H

#include <sudoku/cuda/result.h>

namespace sudoku
{
    namespace cuda
    {
        void computeNextSolutionKernelWrapper(unsigned blockCount, unsigned threadsPerBlock, Result* results);
    }
}

#endif
