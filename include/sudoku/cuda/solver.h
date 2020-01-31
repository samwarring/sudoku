#ifndef INCLUDED_SUDOKU_CUDA_SOLVER_H
#define INCLUDED_SUDOKU_CUDA_SOLVER_H

#include <sudoku/cuda/dimensions.h>
#include <sudoku/cuda/grid.h>
#include <sudoku/cuda/guess_stack.h>
#include <sudoku/cuda/host_and_device.h>
#include <sudoku/cuda/result.h>

namespace sudoku
{
    namespace cuda
    {
        class Solver
        {
            public:
                CUDA_HOST_AND_DEVICE
                Solver(Dimensions dims, Grid grid, GuessStack guessStack);

                CUDA_HOST_AND_DEVICE
                Result computeNextSolution(size_t maxGuessCount);

            private:
                Dimensions dims_;
                Grid grid_;
                GuessStack guessStack_;
        };
    }
}

#endif
