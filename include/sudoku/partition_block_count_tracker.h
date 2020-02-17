#ifndef INCLUDED_SUDOKU_PARTITION_BLOCK_COUNT_TRACKER_H
#define INCLUDED_SUDOKU_PARTITION_BLOCK_COUNT_TRACKER_H

#include <sudoku/partition_table.h>
#include <sudoku/potential.h>
#include <sudoku/block_count_tracker.h>

namespace sudoku
{
    /**
     * Tracks cell block counts and value block counts for a single
     * partition of the grid.
     */
    class PartitionBlockCountTracker
    {
        public:
            PartitionBlockCountTracker(const PartitionTable& partitionTable,
                                       PartitionCount partitionId,
                                       CellValue maxCellValue);

            /**
             * Increment block counts for related cells covered by this partition.
             */
            void setCellValue(CellCount globalPos, CellValue cellValue);

            /**
             * Decrement block counts for related cells covered by this partition.
             */
            void clearCellValue(CellCount globalPos, CellValue cellValue);

            /**
             * Get the local position corresponding to the max block empty cell
             * covered by this partition.
             */
            CellCount getMaxBlockEmptyCell() const;

            /**
             * Get number of values blocked for the given local position.
             */
            CellBlockCount getCellBlockCount(CellCount localPos) const;

            /**
             * Check if cell value is blocked for the given local position.
             */
            bool isBlocked(CellCount localPos, CellValue cellValue) const;

            /**
             * Get next available value greater than minCellValue for the given
             * local position.
             */
            CellValue getNextAvailableValue(CellCount localPos, CellValue minCellValue) const;

        private:
            const PartitionTable* partitionTable_;
            const PartitionCount partitionId_;
            std::vector<Potential> cellPotentials_;
            BlockCountTracker blockCountTracker_;
    };
}

#endif
