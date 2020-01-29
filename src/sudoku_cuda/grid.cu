#include <sudoku/cuda/grid.h>

namespace sudoku
{
    namespace cuda
    {
        Grid::HostData::HostData(Dimensions dims, const std::vector<sudoku::Grid>& grids)
        {
            serialize(dims, grids);
            data_.cellValues = cellValues_.data();
            data_.blockCounts = blockCounts_.data();

            // Now that the grid has been serialized, we can construct cuda::Grid objects
            // from the serialized data. The cell values and restrictions have been copied
            // but the "potentials" are still all 0. Initialize them now.
            initBlockCounts(dims, grids);
        }

        void Grid::HostData::serialize(Dimensions dims, const std::vector<sudoku::Grid>& grids)
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

            // Allocate space for block counts (initialize to 0)
            blockCounts_.resize(grids.size() * dims.getCellCount() * (1 + dims.getMaxCellValue()));
        }

        void Grid::HostData::initBlockCounts(Dimensions dims, const std::vector<sudoku::Grid>& grids)
        {
            for (size_t i = 0; i < grids.size(); ++i) {
                Grid grid(dims, data_, i);
                grid.initBlockCounts();
                for (auto restr : grids[i].getRestrictions()) {
                    grid.blockCellValue(restr.first, restr.second);
                }
            }
        }

        Grid::DeviceData::DeviceData(const HostData& hostData)
            : cellValues_(hostData.cellValues_)
            , blockCounts_(hostData.blockCounts_)
        {
            data_.cellValues = cellValues_.begin();
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
        size_t Grid::getNextAvailableValue(size_t cellPos, size_t cellValue)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            CUDA_HOST_ASSERT(cellValue <= dims_->getMaxCellValue());
            for (size_t v = cellValue + 1; v <= dims_->getMaxCellValue(); ++v) {
                if (*getBlockCount(cellPos, v) == 0) {
                    return v;
                }
            }
            return 0;
        }

        CUDA_HOST_AND_DEVICE
        size_t Grid::getMaxBlockEmptyCell()
        {
            size_t maxBlockCount = 0;
            size_t maxBlockPos = dims_->getCellCount();
            for (size_t cellPos = 0; cellPos < dims_->getCellCount(); ++cellPos) {
                size_t blockCount = *getBlockCount(cellPos);
                size_t cellValue = getCellValue(cellPos);
                if (cellValue == 0 && blockCount >= maxBlockCount) {
                    maxBlockCount = blockCount;
                    maxBlockPos = cellPos;
                    if (maxBlockCount == dims_->getMaxCellValue()) {
                        break;
                    }
                }
            }
            return maxBlockPos;
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

        void Grid::initBlockCounts()
        {
            for (size_t cellPos = 0; cellPos < dims_->getCellCount(); ++cellPos) {
                const size_t cellValue = getCellValue(cellPos);
                if (cellValue > 0) {
                    blockRelatedCells(cellPos, cellValue);
                }
            }
        }
    }
}
