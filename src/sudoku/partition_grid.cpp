#include <thread>
#include <sudoku/partition_grid.h>

namespace sudoku
{
    PartitionGrid::PartitionGrid(const Dimensions& dims, const PartitionTable& partitionTable)
        : dims_(&dims)
        , partitionTable_(&partitionTable)
        , cellValues_(dims.getCellCount()) // init to empty grid (for now...)
        , partitionCount_(partitionTable.getPartitionCount())
        , startBarrier_(partitionCount_)
        , endBarrier_(partitionCount_)
        , broadcastCellPos_(dims.getCellCount())
        , broadcastCellValue_(dims.getMaxCellValue() + 1)
        , broadcastOperation_(true)
        , broadcastTerminate_(false)
    {
        for (PartitionCount partitionId = 0; partitionId < partitionCount_; ++partitionId) {
            trackers_.emplace_back(partitionTable, partitionId, dims.getMaxCellValue());
        }

        // Start threads
        for (PartitionCount partitionId = 1; partitionId < partitionCount_; ++partitionId) {
            threads_.emplace_back([this, partitionId](){
                while (true) {
                    startBarrier_.wait();
                    if (broadcastTerminate_) {
                        return;
                    }
                    if (broadcastOperation_) {
                        trackers_[partitionId].setCellValue(broadcastCellPos_, broadcastCellValue_);
                    }
                    else {
                        trackers_[partitionId].clearCellValue(broadcastCellPos_, broadcastCellValue_);
                    }
                    endBarrier_.wait();
                }
            });
        }
    }

    PartitionGrid::~PartitionGrid()
    {
        broadcastTerminate_ = true;
        startBarrier_.wait();
        for (auto& t : threads_) {
            t.join();
        }
    }

    void PartitionGrid::setCellValue(CellCount cellPos, CellValue cellValue)
    {
        // Broadcast the parameters
        broadcastOperation_ = true;
        broadcastCellPos_ = cellPos;
        broadcastCellValue_ = cellValue;

        // Run this thread synchronized with the workers.
        startBarrier_.wait();
        trackers_[0].setCellValue(cellPos, cellValue);
        endBarrier_.wait();

        // Update cell value
        cellValues_[cellPos] = cellValue;
    }

    void PartitionGrid::clearCellValue(CellCount cellPos)
    {
        // Broadcast the parameters.
        broadcastOperation_ = false;
        broadcastCellPos_ = cellPos;
        broadcastCellValue_ = cellValues_[cellPos];

        // Run this thread synchronized with the workers.
        startBarrier_.wait();
        trackers_[0].clearCellValue(cellPos, cellValues_[cellPos]);
        endBarrier_.wait();

        // Update cell value
        cellValues_[cellPos] = 0;
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
