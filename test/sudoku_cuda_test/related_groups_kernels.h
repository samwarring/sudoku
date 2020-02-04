#ifndef INCLUDED_SUDOKU_CUDA_RELATED_GROUPS_KERNELS_H
#define INCLUDED_SUDOKU_CUDA_RELATED_GROUPS_KERNELS_H

#include <sudoku/cuda/types.h>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/dimensions.h>

struct RelatedGroupsKernelArgs
{
    sudoku::cuda::CellCount cellCount;
    sudoku::cuda::GroupCount totalGroupCount;
    sudoku::cuda::GroupCount* groupCounts;
    sudoku::cuda::GroupCount* groupIds;
};

class RelatedGroupsKernels
{
private:
    sudoku::cuda::DeviceBuffer<sudoku::cuda::GroupCount> groupCounts_;
    sudoku::cuda::DeviceBuffer<sudoku::cuda::GroupCount> groupIds_;
    RelatedGroupsKernelArgs args_;

public:
    RelatedGroupsKernels(const sudoku::Dimensions& dims);

    std::vector<sudoku::cuda::CellValue> broadcastAndReceive(sudoku::cuda::CellCount cellPos,
                                                             sudoku::cuda::CellValue cellValue);
};

#endif
