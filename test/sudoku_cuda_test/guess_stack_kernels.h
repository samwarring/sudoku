#ifndef INCLUDED_GUESS_STACK_KERNELS_H
#define INCLUDED_GUESS_STACK_KERNELS_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/types.h>

class GuessStackKernels
{
private:
    std::vector<sudoku::cuda::CellCount> hostGuessStack_;
    std::vector<sudoku::cuda::CellCount> hostGuessStackSize_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::CellCount> deviceGuessStack_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::CellCount> deviceGuessStackSize_;
    unsigned threadCount_;
    unsigned sharedMemSize_;

    void copyToHost();

public:
    GuessStackKernels(sudoku::cuda::CellCount maxStackSize);

    void push(sudoku::cuda::CellCount cellPos);

    sudoku::cuda::CellCount pop();

    sudoku::cuda::CellCount getSize() const { return hostGuessStackSize_[0]; }
};

#endif
