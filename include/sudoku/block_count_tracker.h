#ifndef INCLUDED_SUDOKU_BLOCK_COUNT_TRACKER_H
#define INCLUDED_SUDOKU_BLOCK_COUNT_TRACKER_H

#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/types.h>

namespace sudoku
{
    /**
     * Tracks how many values are blocked for each cell. Time complexities
     * for the methods use N = cell count.
     */
    class BlockCountTracker
    {
        public:
            BlockCountTracker(const sudoku::Dimensions& dims);

            /**
             * Get an empty cell with the highest block count. If there are no
             * empty cells, return an arbitrary cell. O(1).
             */
            CellCount getMaxBlockEmptyCell() const { return posHeap_[1]; }

            /**
             * The cell is newly blocked by a related cell. O(log N).
             */
            void incrementBlockCount(CellCount cellPos);

            /**
             * The cell is newly unblocked by all related cells. O(log N).
             */
            void derementBlockCount(CellCount cellPos);

            /**
             * The cell has been assigned a value (no longer empty). O(log N).
             */
            void markCellOccupied(CellCount cellPos);

            /**
             * The cell's value has been cleared. O(log N).
             */
            void markCellEmpty(CellCount cellPos);

            /**
             * Get the number of values blocked for an empty cell.
             * 
             * \warning If the cell is occupied, the block count is undefined.
             *          Caller should first check that the cell is empty.
             */
            int getBlockCount(CellCount cellPos) const;

        private:
            /// Avoid confusion between cell position and heap position
            using HeapIndex = CellCount;

            /**
             * Heap layout convenience functions.
             */
            static HeapIndex getLeftHeapChild(HeapIndex heapIndex);
            static HeapIndex getRightHeapChild(HeapIndex heapIndex);
            static HeapIndex getHeapParent(HeapIndex heapIndex);

            /**
             * Re-heapify after a cell position has increased its block count.
             */
            void onBlockCountIncrease(CellCount cellPos);

            /**
             * Re-heapify after a cell position has decreased its block count.
             */
            void onBlockCountDecrease(CellCount cellPos);

            /**
             * Swap heap at given index with its parent (if necessary).
             * Also, update posIndex accordingly.
             * 
             * \return true if a swap occurred
             */
            bool trySwap(HeapIndex childIndex, HeapIndex parentIndex);

            CellBlockCount occupiedBlockCount_;       ///< subtracted from the block count when a cell is occupied
            std::vector<HeapIndex> posIndex_;         ///< [cellPos] -> location in posHeap_
            std::vector<CellCount> posHeap_;          ///< [0] -> cell pos with greatest block count
            std::vector<CellBlockCount> blockCounts_; ///< [cellPos] -> block count
    };
}

#endif
