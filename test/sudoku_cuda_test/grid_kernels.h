#ifndef INCLUDED_GRID_KERNELS_H
#define INCLUDED_GRID_KERNELS_H

#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/types.h>
#include <sudoku/grid.h>
#include "block_counter_kernels.h"
#include "related_groups_kernels.h"

struct GridKernelArgs
{
    sudoku::cuda::CellValue* cellValues;
    sudoku::cuda::CellCount cellCount;
};

class GridKernels
{
private:
    const sudoku::Dimensions& dims_;
    std::vector<sudoku::cuda::CellValue> hostCellValues_;
    std::vector<sudoku::cuda::CellBlockCount> hostCellBlockCounts_;
    std::vector<sudoku::cuda::CellValue> hostValueBlockCounts_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::GroupCount> groupCounts_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::GroupCount> groupIds_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::CellBlockCount> deviceCellBlockCounts_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::ValueBlockCount> deviceValueBlockCounts_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::CellValue> deviceCellValues_;
    RelatedGroupsKernelArgs relatedGroupsArgs_;
    BlockCounterKernelArgs blockCounterArgs_;
    GridKernelArgs gridArgs_;
    sudoku::cuda::CellCount cellCountPow2_;
    size_t sharedGroupUpdatesSize_;

    void copyToHost();

public:
    GridKernels(const sudoku::Grid& grid);

    void initBlockCounts();

    void setCellValue(sudoku::cuda::CellCount cellPos, sudoku::cuda::CellValue cellValue);

    void clearCellValue(sudoku::cuda::CellCount cellPos);
    
    sudoku::cuda::CellCount getMaxCellBlockCountPos();

    sudoku::cuda::CellValue getCellValue(sudoku::cuda::CellCount cellPos) const;

    sudoku::cuda::CellBlockCount getCellBlockCount(sudoku::cuda::CellCount cellPos) const;

    sudoku::cuda::ValueBlockCount getValueBlockCount(sudoku::cuda::CellCount cellPos, sudoku::cuda::CellValue cellValue) const;

};

#endif
