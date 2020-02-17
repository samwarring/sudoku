#include <sudoku/partition_block_count_tracker.h>

namespace sudoku
{
    PartitionBlockCountTracker::PartitionBlockCountTracker(const PartitionTable& partitionTable,
                                                           PartitionCount partitionId,
                                                           CellValue maxCellValue)
        : partitionTable_(&partitionTable)
        , partitionId_(partitionId)
        , cellPotentials_(partitionTable.getPartitionSize(partitionId), maxCellValue)
        , blockCountTracker_(partitionTable.getPartitionSize(partitionId), maxCellValue)
    {}

    void PartitionBlockCountTracker::setCellValue(CellCount cellPos, CellValue cellValue)
    {
        // Update related cells covered by this partition.
        for (auto relatedLocalPos : partitionTable_->getRelatedIndicesForPartition(partitionId_, cellPos)) {
            if (cellPotentials_[relatedLocalPos].block(cellValue)) {
                blockCountTracker_.incrementBlockCount(relatedLocalPos);
            }
        }

        // Mark the cell as occupied if its covered by this partition.
        if (partitionTable_->getPartitionId(cellPos) == partitionId_) {
            blockCountTracker_.markCellOccupied(partitionTable_->getPartitionIndex(cellPos));
        }
    }

    void PartitionBlockCountTracker::clearCellValue(CellCount cellPos, CellValue cellValue)
    {
        // Update related cells covered by this partition.
        for (auto relatedLocalPos : partitionTable_->getRelatedIndicesForPartition(partitionId_, cellPos)) {
            if (cellPotentials_[relatedLocalPos].unblock(cellValue)) {
                blockCountTracker_.derementBlockCount(relatedLocalPos);
            }
        }

        // Mark the cell as free if its covered by this partition.
        if (partitionTable_->getPartitionId(cellPos) == partitionId_) {
            blockCountTracker_.markCellEmpty(partitionTable_->getPartitionIndex(cellPos));
        }
    }

    CellCount PartitionBlockCountTracker::getMaxBlockEmptyCell() const
    {
        return blockCountTracker_.getMaxBlockEmptyCell();
    }

    CellBlockCount PartitionBlockCountTracker::getCellBlockCount(CellCount localPos) const
    {
        return blockCountTracker_.getBlockCount(localPos);
    }

    bool PartitionBlockCountTracker::isBlocked(CellCount localPos, CellValue cellValue) const
    {
        return cellPotentials_[localPos].isBlocked(cellValue);
    }

    CellValue PartitionBlockCountTracker::getNextAvailableValue(CellCount localPos, CellValue minCellValue) const
    {
        return cellPotentials_[localPos].getNextAvailableValue(minCellValue);
    }
}
