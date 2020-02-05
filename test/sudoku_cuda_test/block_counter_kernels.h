#ifndef INCLUDED_BLOCK_COUNTER_KERNELS_H
#define INCLUDED_BLOCK_COUNTER_KERNELS_H

#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/types.h>

struct BlockCounterKernelArgs
{
    sudoku::cuda::CellCount cellCount;
    sudoku::cuda::CellValue maxCellValue;
    sudoku::cuda::CellBlockCount* cellBlockCounts;
    sudoku::cuda::ValueBlockCount* valueBlockCounts;
};

class BlockCounterKernels
{
private:
    unsigned cellCountPow2_;
    BlockCounterKernelArgs args_;
    std::vector<sudoku::cuda::CellBlockCount> hostCellBlockCounts_;
    std::vector<sudoku::cuda::ValueBlockCount> hostValueBlockCounts_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::CellBlockCount> deviceCellBlockCounts_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::ValueBlockCount> deviceValueBlockCounts_;
    unsigned sharedMemSize_;

    void copyToHost();

public:
    struct Pair
    {
        sudoku::cuda::CellCount cellPos;
        sudoku::cuda::CellBlockCount cellBlockCount;
    };

    BlockCounterKernels(sudoku::cuda::CellCount cellCount, sudoku::cuda::CellValue maxCellValue);

    sudoku::cuda::CellBlockCount getCellBlockCount(sudoku::cuda::CellCount cellpos) const;

    sudoku::cuda::ValueBlockCount getValueBlockCount(sudoku::cuda::CellCount cellPos, sudoku::cuda::CellValue cellValue) const;

    void block(sudoku::cuda::CellCount blockPos, sudoku::cuda::CellValue blockValue);

    void unblock(sudoku::cuda::CellCount unblockPos, sudoku::cuda::CellValue unblockValue);

    void markOccupied(sudoku::cuda::CellCount occupiedPos);

    void markFree(sudoku::cuda::CellCount freePos);

    Pair getMaxBlockCountPair();
};

#endif
