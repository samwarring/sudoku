#include <algorithm>
#include <cassert>
#include <numeric>
#include <sudoku/block_count_tracker.h>

namespace sudoku
{
    BlockCountTracker::BlockCountTracker(const sudoku::Dimensions& dims)
        : occupiedBlockCount_(static_cast<CellBlockCount>(dims.getMaxCellValue() + 1))
        , posIndex_(dims.getCellCount())
        , posHeap_(dims.getCellCount() + 1)
        , blockCounts_(dims.getCellCount())
    {
        // Don't use the first position in the heap.
        // posHeap[1] is the _real_ max of the heap.
        // This simplifies implementation.
        std::iota(posIndex_.begin(), posIndex_.end(), 1);
        std::iota(posHeap_.begin() + 1, posHeap_.end(), 0);
        posHeap_[0] = static_cast<CellCount>(~0);
    }

    void BlockCountTracker::incrementBlockCount(CellCount cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] != -1);
        blockCounts_[cellPos]++;
        onBlockCountIncrease(cellPos);
    }

    void BlockCountTracker::derementBlockCount(CellCount cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] != 0);
        blockCounts_[cellPos]--;
        onBlockCountDecrease(cellPos);
    }

    void BlockCountTracker::markCellOccupied(CellCount cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] >= 0);
        blockCounts_[cellPos] -= occupiedBlockCount_;
        onBlockCountDecrease(cellPos);
    }

    void BlockCountTracker::markCellEmpty(CellCount cellPos)
    {
        assert(cellPos < posIndex_.size());
        assert(blockCounts_[cellPos] < 0);
        blockCounts_[cellPos] += occupiedBlockCount_;
        onBlockCountIncrease(cellPos);
    }

    CellBlockCount BlockCountTracker::getBlockCount(CellCount cellPos) const
    {
        assert(blockCounts_[cellPos] >= 0);
        return blockCounts_[cellPos];
    }

    BlockCountTracker::HeapIndex BlockCountTracker::getLeftHeapChild(HeapIndex heapIndex)
    {
        return heapIndex * 2;
    }

    BlockCountTracker::HeapIndex BlockCountTracker::getRightHeapChild(HeapIndex heapIndex)
    {
        return (heapIndex * 2) + 1;
    }

    BlockCountTracker::HeapIndex BlockCountTracker::getHeapParent(HeapIndex heapIndex)
    {
        return heapIndex / 2;
    }

    void BlockCountTracker::onBlockCountIncrease(CellCount cellPos)
    {
        HeapIndex heapIndex = posIndex_[cellPos];
        HeapIndex parentIndex = getHeapParent(heapIndex);
        while (parentIndex && trySwap(heapIndex, parentIndex)) {
            heapIndex = parentIndex;
            parentIndex = getHeapParent(parentIndex);
        }
    }

    void BlockCountTracker::onBlockCountDecrease(CellCount cellPos)
    {
        HeapIndex heapIndex = posIndex_[cellPos];
        HeapIndex leftChildIndex = getLeftHeapChild(heapIndex);
        HeapIndex rightChildIndex = getRightHeapChild(heapIndex);
        for(;;) {
            if (leftChildIndex >= posHeap_.size()) {
                // 0 children
                break;
            }
            else if (rightChildIndex >= posHeap_.size()) {
                // 1 child
                trySwap(leftChildIndex, heapIndex);
                break;
            }
            else {
                // 2 children
                auto leftValue = blockCounts_[posHeap_[leftChildIndex]];
                auto rightValue = blockCounts_[posHeap_[rightChildIndex]];
                if (leftValue > rightValue && trySwap(leftChildIndex, heapIndex)) {
                    // left child is greatest of parent and children
                    heapIndex = leftChildIndex;
                    leftChildIndex = getLeftHeapChild(heapIndex);
                    rightChildIndex = getRightHeapChild(heapIndex);
                    continue;
                }
                else if (trySwap(rightChildIndex, heapIndex)) {
                    // right child is greatest of parent and children
                    heapIndex = rightChildIndex;
                    leftChildIndex = getLeftHeapChild(heapIndex);
                    rightChildIndex = getRightHeapChild(heapIndex);
                    continue;
                }
                // parent greater than both children.
                break;
            }
        }
    }

    bool BlockCountTracker::trySwap(HeapIndex childIndex, HeapIndex parentIndex)
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
