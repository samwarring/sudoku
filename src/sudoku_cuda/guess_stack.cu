#include <sudoku/cuda/guess_stack.cuh>

namespace sudoku
{
    namespace cuda
    {
        __device__ GuessStack::GuessStack(CellCount* globalGuessStack,
                                          CellCount* globalGuessStackSize,
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

        __device__ GuessStack::~GuessStack()
        {
            // Copy shared guess stack back to global buffer.
            if (threadIdx.x < guessStackSize_) {
                globalGuessStack_[threadIdx.x] = sharedGuessStack_[threadIdx.x];
            }
            if (threadIdx.x == 0) {
                *globalGuessStackSize_ = guessStackSize_;
            }
        }

        __device__ CellCount GuessStack::getSize() const
        {
            return guessStackSize_;
        }

        __device__ void GuessStack::push(CellCount cellPos)
        {
            // Write position to guess stack.
            if (threadIdx.x == 0) {
                sharedGuessStack_[guessStackSize_] = cellPos;
            }
            __syncthreads();
            guessStackSize_++;
        }

        __device__ CellCount GuessStack::pop()
        {
            // Get top of stack.
            guessStackSize_--;
            return sharedGuessStack_[guessStackSize_];
        }
    }
}
