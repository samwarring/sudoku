#include <sudoku/cuda/dimensions.h>

namespace sudoku
{
    namespace cuda
    {
        Dimensions::HostData::HostData(const sudoku::Dimensions& dims)
        {
            serialize(dims);
            data_.cellCount = dims.getCellCount();
            data_.maxCellValue = dims.getMaxCellValue();
            data_.groupCount = dims.getNumGroups();
            data_.groupValues = groupValues_.data();
            data_.groupOffsets = groupOffsets_.data();
            data_.groupsForCellValues = groupsForCellValues_.data();
            data_.groupsForCellOffsets = groupsForCellOffsets_.data();
        }

        void Dimensions::HostData::serialize(const sudoku::Dimensions& dims)
        {
            // Concatenate groups.
            for (size_t gn = 0; gn < dims.getNumGroups(); ++gn) {
                groupOffsets_.push_back(groupValues_.size());
                for (auto cellPos : dims.getCellsInGroup(gn)) {
                    groupValues_.push_back(cellPos);
                }
            }
            groupOffsets_.push_back(groupValues_.size());

            // Concatenate groups for each cell.
            for (size_t cp = 0; cp < dims.getCellCount(); ++cp) {
                groupsForCellOffsets_.push_back(groupsForCellValues_.size());
                for (auto gn : dims.getGroupsForCell(cp)) {
                    groupsForCellValues_.push_back(gn);
                }
            }
            groupsForCellOffsets_.push_back(groupsForCellValues_.size());
        }

        size_t Dimensions::HostData::getAllocatedSize() const
        {
            return (
                (sizeof(size_t) * groupValues_.size()) +
                (sizeof(size_t) * groupOffsets_.size()) +
                (sizeof(size_t) * groupsForCellValues_.size()) + 
                (sizeof(size_t) * groupsForCellOffsets_.size())
            );
        }

        Dimensions::DeviceData::DeviceData(const Dimensions::HostData& hostData)
            : groupValues_(hostData.groupValues_)
            , groupOffsets_(hostData.groupOffsets_)
            , groupsForCellValues_(hostData.groupsForCellValues_)
            , groupsForCellOffsets_(hostData.groupsForCellOffsets_)
            , data_(hostData.data_)
        {
            data_.groupValues = groupValues_.begin();
            data_.groupOffsets = groupOffsets_.begin();
            data_.groupsForCellValues = groupsForCellValues_.begin();
            data_.groupsForCellOffsets = groupsForCellOffsets_.begin();
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getCellCount() const
        {
            return data_.cellCount;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getMaxCellValue() const
        {
            return data_.maxCellValue;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getGroupCount() const
        {
            return data_.groupCount;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getCellsInGroupCount(size_t groupNum) const
        {
            CUDA_HOST_ASSERT(groupNum < data_.groupCount);
            return data_.groupOffsets[groupNum + 1] - data_.groupOffsets[groupNum];
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getCellInGroup(size_t groupNum, size_t itemNum) const
        {
            CUDA_HOST_ASSERT(itemNum < getCellsInGroupCount(groupNum));
            return *(data_.groupValues + data_.groupOffsets[groupNum] + itemNum);
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getGroupsForCellCount(size_t cellPos) const
        {
            CUDA_HOST_ASSERT(cellPos < data_.cellCount);
            return data_.groupsForCellOffsets[cellPos + 1] - data_.groupsForCellOffsets[cellPos];
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getGroupForCell(size_t cellPos, size_t itemNum) const
        {
            CUDA_HOST_ASSERT(itemNum < getGroupsForCellCount(cellPos));
            return *(data_.groupsForCellValues + data_.groupsForCellOffsets[cellPos] + itemNum);
        }
    }
}
