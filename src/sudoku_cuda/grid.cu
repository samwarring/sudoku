#include <sudoku/cuda/grid.h>

namespace sudoku
{
    namespace cuda
    {
        Grid::HostData::HostData(const std::vector<sudoku::Grid>& grids)
        {
            serialize(grids);
            data_.cellValues = cellValues_.data();
            data_.restrictions = restrictions_.data();
            data_.restrictionsOffsets = restrictionsOffsets_.data();
            data_.blockCounts = blockCounts_.data();
        }

        void Grid::HostData::serialize(const std::vector<sudoku::Grid>& grids)
        {
            assert(grids.size() > 0);

            // Concatenate cell values
            for (const auto& grid : grids) {
                cellValues_.insert(
                    cellValues_.end(),
                    grid.getCellValues().cbegin(),
                    grid.getCellValues().cend()
                );
            }

            // Concatenate restrictions
            for (const auto& grid : grids) {
                restrictionsOffsets_.push_back(restrictions_.size());
                for (auto restr : grid.getRestrictions()) {
                    restrictions_.push_back(restr.first);
                    restrictions_.push_back(restr.second);
                }
            }
            restrictionsOffsets_.push_back(restrictions_.size());

            // Allocate space for block counts (initialize to 0)
            const auto& dims = grids[0].getDimensions();
            blockCounts_.resize(grids.size() * dims.getCellCount() * (1 + dims.getMaxCellValue()));
        }

        Grid::DeviceData::DeviceData(const HostData& hostData)
            : cellValues_(hostData.cellValues_)
            , restrictions_(hostData.restrictions_)
            , restrictionsOffsets_(hostData.restrictionsOffsets_)
            , blockCounts_(hostData.blockCounts_)
        {
            data_.cellValues = cellValues_.begin();
            data_.restrictions = restrictions_.begin();
            data_.restrictionsOffsets = restrictionsOffsets_.begin();
            data_.blockCounts = blockCounts_.begin();
        }

        CUDA_HOST_AND_DEVICE
        size_t Grid::getCellValue(size_t cellPos)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            return *getCell(cellPos);
        }

        CUDA_HOST_AND_DEVICE
        void Grid::setCellValue(size_t cellPos, size_t cellValue)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            CUDA_HOST_ASSERT(0 < cellValue && cellValue <= dims_->getMaxCellValue());
            *getCell(cellPos) = cellValue;
            blockRelatedCells(cellPos, cellValue);
        }

        CUDA_HOST_AND_DEVICE
        void Grid::clearCellValue(size_t cellPos)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            unblockRelatedCells(cellPos, *getCell(cellPos));
            *getCell(cellPos) = 0;
        }

        CUDA_HOST_AND_DEVICE
        size_t* Grid::getCell(size_t cellPos)
        {
            return data_.cellValues + (threadNum_ * dims_->getCellCount()) + cellPos;
        }

        CUDA_HOST_AND_DEVICE
        size_t* Grid::getBlockCount(size_t cellPos, size_t cellValue)
        {
            CUDA_HOST_ASSERT(0 < cellValue && cellValue <= dims_->getMaxCellValue());
            return getBlockCount(cellPos) + cellValue;
        }

        CUDA_HOST_AND_DEVICE
        size_t* Grid::getBlockCount(size_t cellPos)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            return data_.blockCounts + (cellPos * (dims_->getMaxCellValue() + 1));
        }

        CUDA_HOST_AND_DEVICE
        void Grid::blockCellValue(size_t cellPos, size_t cellValue)
        {
            size_t* cellBlockCount = getBlockCount(cellPos);
            size_t* valueBlockCount = getBlockCount(cellPos, cellValue);
            *cellBlockCount += (size_t)(*valueBlockCount == 0);
            (*valueBlockCount)++;
        }

        CUDA_HOST_AND_DEVICE
        void Grid::unblockCellValue(size_t cellPos, size_t cellValue)
        {
            size_t* cellBlockCount = getBlockCount(cellPos);
            size_t* valueBlockCount = getBlockCount(cellPos, cellValue);
            *cellBlockCount -= (size_t)(*valueBlockCount == 1);
            (*valueBlockCount)--;
        }

        CUDA_HOST_AND_DEVICE
        void Grid::blockRelatedCells(size_t cellPos, size_t cellValue)
        {
            const size_t relatedGroupCount = dims_->getGroupsForCellCount(cellPos);
            for (size_t i = 0; i < relatedGroupCount; ++i) {
                const size_t groupNum = dims_->getGroupForCell(cellPos, i);
                const size_t groupSize = dims_->getCellsInGroupCount(groupNum);
                for (size_t j = 0; j < groupSize; ++j) {
                    size_t relatedPos = dims_->getCellInGroup(groupNum, j);
                    blockCellValue(relatedPos, cellValue);
                }
            }
        }

        CUDA_HOST_AND_DEVICE
        void Grid::unblockRelatedCells(size_t cellPos, size_t cellValue)
        {
            const size_t relatedGroupCount = dims_->getGroupsForCellCount(cellPos);
            for (size_t i = 0; i < relatedGroupCount; ++i) {
                const size_t groupNum = dims_->getGroupForCell(cellPos, i);
                const size_t groupSize = dims_->getCellsInGroupCount(groupNum);
                for (size_t j = 0; j < groupSize; ++j) {
                    size_t relatedPos = dims_->getCellInGroup(groupNum, j);
                    unblockCellValue(relatedPos, cellValue);
                }
            }
        }
    }
}
