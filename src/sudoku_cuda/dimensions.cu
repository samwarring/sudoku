#include <sudoku/cuda/dimensions.h>

namespace sudoku
{
    namespace cuda
    {
        CUDA_HOST_AND_DEVICE
        Dimensions::Dimensions(KernelParams kernelParams)
        {
            cellCount_ = kernelParams.cellCount;
            maxCellValue_ = kernelParams.maxCellValue;
            groupCount_ = kernelParams.groupCount;
            groupValues_ = kernelParams.groupValues;
            groupOffsets_ = kernelParams.groupOffsets;
            groupsForCellValues_ = kernelParams.groupsForCellValues;
            groupsForCellOffsets_ = kernelParams.groupsForCellOffsets;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getCellCount() const
        {
            return cellCount_;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getMaxCellValue() const
        {
            return maxCellValue_;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getGroupCount() const
        {
            return groupCount_;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getCellsInGroupCount(size_t groupNum) const
        {
            CUDA_HOST_ASSERT(groupNum < groupCount_);
            return groupOffsets_[groupNum + 1] - groupOffsets_[groupNum];
        }

        CUDA_HOST_AND_DEVICE
        const size_t* Dimensions::getCellsInGroup(size_t groupNum) const
        {
            CUDA_HOST_ASSERT(groupNum < groupCount_);
            return groupValues_ + groupOffsets_[groupNum];
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getGroupsForCellCount(size_t cellPos) const
        {
            CUDA_HOST_ASSERT(cellPos < cellCount_);
            return groupsForCellOffsets_[cellPos + 1] - groupsForCellOffsets_[cellPos];
        }

        CUDA_HOST_AND_DEVICE
        const size_t* Dimensions::getGroupsForCell(size_t cellPos) const
        {
            CUDA_HOST_ASSERT(cellPos < cellCount_);
            return groupsForCellValues_ + groupsForCellOffsets_[cellPos];
        }
    }
}
