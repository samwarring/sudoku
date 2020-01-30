#ifndef INCLUDED_SUDOKU_BLOCK_COUNT_TRACKER_H
#define INCLUDED_SUDOKU_BLOCK_COUNT_TRACKER_H

#include <vector>
#include <sudoku/dimensions.h>

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
            size_t getMaxBlockEmptyCell() const { return posHeap_[1]; }

            /**
             * The cell is newly blocked by a related cell. O(log N).
             */
            void incrementBlockCount(size_t cellPos);

            /**
             * The cell is newly unblocked by all related cells. O(log N).
             */
            void derementBlockCount(size_t cellPos);

            /**
             * The cell has been assigned a value (no longer empty). O(log N).
             */
            void markCellOccupied(size_t cellPos);

            /**
             * The cell's value has been cleared. O(log N).
             */
            void markCellEmpty(size_t cellPos);

        private:
            /**
             * Heap layout convenience functions.
             */
            static size_t getLeftHeapChild(size_t heapIndex);
            static size_t getRightHeapChild(size_t heapIndex);
            static size_t getHeapParent(size_t heapIndex);

            /**
             * Re-heapify after a cell position has increased its block count.
             */
            void onBlockCountIncrease(size_t cellPos);

            /**
             * Re-heapify after a cell position has decreased its block count.
             */
            void onBlockCountDecrease(size_t cellPos);

            /**
             * Swap heap at given index with its parent (if necessary).
             * Also, update posIndex accordingly.
             * 
             * \return true if a swap occurred
             */
            bool trySwap(size_t childIndex, size_t parentIndex);

            int occupiedBlockCount_;       ///< subtracted from the block count when a cell is occupied
            std::vector<size_t> posIndex_; ///< [cellPos] -> location in posHeap_
            std::vector<size_t> posHeap_;  ///< [0] -> cell pos with greatest block count
            std::vector<int> blockCounts_; ///< [cellPos] -> block count
    };
}

#endif
