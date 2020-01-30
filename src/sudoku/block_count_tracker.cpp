#include <cassert>
#include <numeric>
#include <sudoku/block_count_tracker.h>

namespace sudoku
{
    BlockCountTracker::BlockCountTracker(const sudoku::Dimensions& dims)
        : occupiedBlockCount_(static_cast<int>(dims.getMaxCellValue() + 1))
        , posIndex_(dims.getCellCount())
        , posHeap_(dims.getCellCount() + 1)
        , blockCounts_(dims.getCellCount())
    {
        // Don't use the first position in the heap.
        // posHeap[1] is the _real_ max of the heap.
        // This simplifies implementation.
        std::iota(posIndex_.begin(), posIndex_.end(), 1);
        std::iota(posHeap_.begin() + 1, posHeap_.end(), 0);
        posHeap_[0] = static_cast<size_t>(~0);
    }

    void BlockCountTracker::incrementBlockCount(size_t cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] != -1);
        blockCounts_[cellPos]++;
        onBlockCountIncrease(cellPos);
    }

    void BlockCountTracker::derementBlockCount(size_t cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] != 0);
        blockCounts_[cellPos]--;
        onBlockCountDecrease(cellPos);
    }

    void BlockCountTracker::markCellOccupied(size_t cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] >= 0);
        blockCounts_[cellPos] -= occupiedBlockCount_;
        onBlockCountDecrease(cellPos);
    }

    void BlockCountTracker::markCellEmpty(size_t cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] < 0);
        blockCounts_[cellPos] += occupiedBlockCount_;
        onBlockCountIncrease(cellPos);
    }

    int BlockCountTracker::getBlockCount(size_t cellPos) const
    {
        assert(blockCounts_[cellPos] >= 0);
        return blockCounts_[cellPos];
    }

    size_t BlockCountTracker::getLeftHeapChild(size_t heapIndex)
    {
        return heapIndex * 2;
    }

    size_t BlockCountTracker::getRightHeapChild(size_t heapIndex)
    {
        return (heapIndex * 2) + 1;
    }

    size_t BlockCountTracker::getHeapParent(size_t heapIndex)
    {
        return heapIndex / 2;
    }

    void BlockCountTracker::onBlockCountIncrease(size_t cellPos)
    {
        size_t heapIndex = posIndex_[cellPos];
        size_t parentIndex = getHeapParent(heapIndex);
        while (parentIndex && trySwap(heapIndex, parentIndex)) {
            heapIndex = parentIndex;
            parentIndex = getHeapParent(parentIndex);
        }
    }

    void BlockCountTracker::onBlockCountDecrease(size_t cellPos)
    {
        size_t heapIndex = posIndex_[cellPos];
        size_t leftChildIndex = getLeftHeapChild(heapIndex);
        size_t rightChildIndex = getRightHeapChild(heapIndex);
        for(;;) {
            if (leftChildIndex >= posHeap_.size()) {
                // No children. Don't swap anything.
                return;
            }
            else if (rightChildIndex >= posHeap_.size()) {
                // Only 1 left child. Attempt one swap and return.
                trySwap(leftChildIndex, heapIndex);
                return;
            }
            else if (trySwap(leftChildIndex, heapIndex)) {
                // left child was swapped.
                heapIndex = leftChildIndex;
                leftChildIndex = getLeftHeapChild(heapIndex);
                rightChildIndex = getRightHeapChild(heapIndex);
                continue;
            }
            else if (trySwap(rightChildIndex, heapIndex)) {
                // right child was swapped.
                heapIndex = rightChildIndex;
                leftChildIndex = getLeftHeapChild(heapIndex);
                rightChildIndex = getRightHeapChild(heapIndex);
                continue;
            }
            // No children, or nothing was swapped.
            return;
        }
    }

    bool BlockCountTracker::trySwap(size_t childIndex, size_t parentIndex)
    {
        auto childCellPos = posHeap_[childIndex];
        auto parentCellPos = posHeap_[parentIndex];
        if (blockCounts_[parentCellPos] < blockCounts_[childCellPos]) {
            posHeap_[parentIndex] = childCellPos;
            posHeap_[childIndex] = parentCellPos;
            posIndex_[parentCellPos] = childIndex;
            posIndex_[childCellPos] = parentIndex;
            return true;
        }
        return false;
    }
}
