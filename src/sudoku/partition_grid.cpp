#include <sudoku/partition_grid.h>

namespace sudoku
{
    PartitionGrid::PartitionGrid(const Dimensions& dims, const PartitionTable& partitionTable)
        : dims_(&dims)
        , partitionTable_(&partitionTable)
        , cellValues_(dims.getCellCount()) // init to empty grid (for now...)
        , partitionCount_(partitionTable.getPartitionCount())
    {
        for (PartitionCount partitionId = 0; partitionId < partitionCount_; ++partitionId) {
            trackers_.emplace_back(partitionTable, partitionId, dims.getMaxCellValue());
        }
    }

    void PartitionGrid::setCellValue(CellCount cellPos, CellValue cellValue)
    {
        // TODO: run each partition in its own thread.
        for (auto& tracker : trackers_) {
            tracker.setCellValue(cellPos, cellValue);
        }
        cellValues_[cellPos] = cellValue;
    }

    void PartitionGrid::clearCellValue(CellCount cellPos)
    {
        auto oldValue = cellValues_[cellPos];
        cellValues_[cellPos] = 0;

        // TODO: run each partition in its own thread.
        for (auto& tracker : trackers_) {
            tracker.clearCellValue(cellPos, oldValue);
        }
    }

    CellCount PartitionGrid::getMaxBlockEmptyCell() const
    {
        CellCount maxBlockCountPos = dims_->getCellCount();
        CellBlockCount maxBlockCount = -1;

        for (PartitionCount partitionId = 0; partitionId < partitionCount_; ++partitionId) {
            auto localPos = trackers_[partitionId].getMaxBlockEmptyCell();
            auto blockCount = trackers_[partitionId].getCellBlockCount(localPos);
            if (blockCount > maxBlockCount) {
                maxBlockCount = blockCount;
                maxBlockCountPos = partitionTable_->getCellPosition(partitionId, localPos);
            }
        }

        return maxBlockCountPos;
    }

    CellValue PartitionGrid::getNextAvailableValue(CellCount cellPos, CellValue minCellValue) const
    {
        auto partitionId = partitionTable_->getPartitionId(cellPos);
        auto localPos = partitionTable_->getPartitionIndex(cellPos);
        return trackers_[partitionId].getNextAvailableValue(localPos, minCellValue);
    }

    CellBlockCount PartitionGrid::getCellBlockCount(CellCount cellPos) const
    {
        auto partitionId = partitionTable_->getPartitionId(cellPos);
        auto localPos = partitionTable_->getPartitionIndex(cellPos);
        return trackers_[partitionId].getCellBlockCount(localPos);
    }
}
