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
                                  CellCount* sharedGuessStack) 
                : guessStackSize_(*globalGuessStackSize)
                , sharedGuessStack_(sharedGuessStack)
                , globalGuessStackSize_(globalGuessStackSize)
                , globalGuessStack_(globalGuessStack)
            {
                // Copy guess stack to shared buffer.
                if (threadIdx.x < guessStackSize_) {
                    sharedGuessStack_[threadIdx.x] = globalGuessStack_[threadIdx.x];
                }
                __syncthreads();
            }

            __device__ ~GuessStack()
            {
                // Copy shared guess stack back to global buffer.
                if (threadIdx.x < guessStackSize_) {
                    globalGuessStack_[threadIdx.x] = sharedGuessStack_[threadIdx.x];
                }
                if (threadIdx.x == 0) {
                    *globalGuessStackSize_ = guessStackSize_;
                }
            }

            __device__ CellCount getSize() const { return guessStackSize_; }

            __device__ void push(CellCount cellPos)
            {
                // Write position to guess stack.
                if (threadIdx.x == 0) {
                    sharedGuessStack_[guessStackSize_] = cellPos;
                }
                __syncthreads();
                guessStackSize_++;
            }

            /// \warning Do not call this method if stack size is 0
            __device__ CellCount pop()
            {
                // Get top of stack.
                guessStackSize_--;
                return sharedGuessStack_[guessStackSize_];
            }    
        };
    }
}

#endif
