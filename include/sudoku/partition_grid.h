#ifndef INCLUDED_SUDOKU_PARTITION_GRID_H
#define INCLUDED_SUDOKU_PARTITION_GRID_H

#include <thread>
#include <sudoku/barrier.h>
#include <sudoku/dimensions.h>
#include <sudoku/partition_block_count_tracker.h>

namespace sudoku
{
    class PartitionGrid
    {
        public:
            PartitionGrid(const Dimensions& dims, const PartitionTable& partitionTable);

            ~PartitionGrid();

            void setCellValue(CellCount cellPos, CellValue cellValue);

            void clearCellValue(CellCount cellPos);

            CellCount getMaxBlockEmptyCell() const;

            CellValue getNextAvailableValue(CellCount cellPos, CellValue minCellValue) const;

            CellBlockCount getCellBlockCount(CellCount cellPos) const;

        private:
            const Dimensions* dims_;
            const PartitionTable* partitionTable_;
            std::vector<CellValue> cellValues_;
            PartitionCount partitionCount_;
            std::vector<PartitionBlockCountTracker> trackers_;
            std::vector<std::thread> threads_;
            SpinBarrier startBarrier_;
            SpinBarrier endBarrier_;
            CellCount broadcastCellPos_;
            CellValue broadcastCellValue_;
            bool broadcastOperation_; ///< true=setCellValue, false=clearCellValue
            bool broadcastTerminate_;
    };
}

#endif
