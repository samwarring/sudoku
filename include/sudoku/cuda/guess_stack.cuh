#ifndef INCLUDED_SUDOKU_CUDA_GUESS_STACK_CUH
#define INCLUDED_SUDOKU_CUDA_GUESS_STACK_CUH

#include <sudoku/cuda/types.h>

namespace sudoku
{
    namespace cuda
    {
        class GuessStack
        {
        private:
            CellCount  guessStackSize_;
            CellCount* sharedGuessStack_;
            CellCount* globalGuessStackSize_;
            CellCount* globalGuessStack_;

        public:
            __device__ GuessStack(CellCount* globalGuessStack, CellCount* globalGuessStackSize,
                                  CellCount* sharedGuessStack);

            __device__ ~GuessStack();

            __device__ CellCount getSize() const;

            __device__ void push(CellCount cellPos);

            /// \warning Do not call this method if stack size is 0
            __device__ CellCount pop();
        };
    }
}

#endif
