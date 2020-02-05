#include <sudoku/cuda/error_check.h>
#include <sudoku/cuda/guess_stack.cuh>
#include "guess_stack_kernels.h"

using namespace sudoku::cuda;

__global__ void guessStackPushKernel(Guess* globalGuessStack, CellCount* globalGuessStackSize, CellCount cellPos)
{
    extern __shared__ Guess sharedGuessStack[];
    GuessStack guessStack(globalGuessStack, globalGuessStackSize, sharedGuessStack);
    guessStack.push(cellPos, 1);
}

__global__ void guessStackPopKernel(Guess* globalGuessStack, CellCount* globalGuessStackSize, CellCount* outPos)
{
    extern __shared__ Guess sharedGuessStack[];
    GuessStack guessStack(globalGuessStack, globalGuessStackSize, sharedGuessStack);
    auto prevGuess = guessStack.pop();
    if (threadIdx.x == 0) {
        *outPos = prevGuess.cellPos;
    }
}

GuessStackKernels::GuessStackKernels(CellCount maxStackSize)
    : hostGuessStack_(maxStackSize, {0, 0})
    , hostGuessStackSize_(1, 0)
    , deviceGuessStack_(hostGuessStack_)
    , deviceGuessStackSize_(hostGuessStackSize_)
    , threadCount_(maxStackSize)
    , sharedMemSize_(sizeof(Guess) * maxStackSize)
{}

void GuessStackKernels::copyToHost()
{
    hostGuessStack_ = deviceGuessStack_.copyToHost();
    hostGuessStackSize_ = deviceGuessStackSize_.copyToHost();
}

void GuessStackKernels::push(CellCount cellPos)
{
    guessStackPushKernel<<<1, threadCount_, sharedMemSize_>>>(
        deviceGuessStack_.get(), deviceGuessStackSize_.get(), cellPos
    );
    ErrorCheck::lastError();
    copyToHost();
}

CellCount GuessStackKernels::pop()
{
    DeviceBuffer<CellCount> outPos(1);
    guessStackPopKernel<<<1, threadCount_, sharedMemSize_>>>(
        deviceGuessStack_.get(), deviceGuessStackSize_.get(), outPos.get()
    );
    ErrorCheck::lastError();
    copyToHost();
    return outPos.copyToHost()[0];
}
