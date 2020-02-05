#ifndef INCLUDED_SUDOKU_CUDA_GRID_CUH
#define INCLUDED_SUDOKU_CUDA_GRID_CUH

#include <sudoku/cuda/types.h>
#include <sudoku/cuda/block_counter.cuh>
#include <sudoku/cuda/related_groups.cuh>

namespace sudoku
{
    namespace cuda
    {
        template <CellValue MAX_CELL_VALUE, GroupCount MAX_GROUPS_FOR_CELL>
        class Grid
        {
        private:
            RelatedGroups<MAX_GROUPS_FOR_CELL>* relatedGroups_;
            BlockCounter<MAX_CELL_VALUE>* blockCounter_;
            CellValue* globalCellValues_;
            CellCount cellCount_;
            CellValue cellValue_;

        public:
            __device__ Grid(RelatedGroups<MAX_GROUPS_FOR_CELL>& relatedGroups,
                            BlockCounter<MAX_CELL_VALUE>& blockCounter,
                            CellValue* globalCellValues, CellCount cellCount)
                : relatedGroups_(&relatedGroups)
                , blockCounter_(&blockCounter)
                , globalCellValues_(globalCellValues)
                , cellCount_(cellCount)
            {
                // Copy cell value from global buffer.
                if (threadIdx.x < cellCount) {
                    cellValue_ = globalCellValues[threadIdx.x];
                }
            }

            __device__ ~Grid()
            {
                // Copy cell value back to global buffer.
                if (threadIdx.x < cellCount_) {
                    globalCellValues_[threadIdx.x] = cellValue_;
                }
            }

            __device__ void initBlockCounts()
            {
                for (CellCount cellPos = 0; cellPos < cellCount_; ++cellPos) {
                    CellValue cellValue = globalCellValues_[cellPos];
                    if (cellValue > 0) {
                        setCellValue(cellPos, cellValue);
                    }
                }
            }

            __device__ void setCellValue(CellCount cellPos, CellValue cellValue)
            {
                // Save the new cell value.
                if (threadIdx.x == cellPos) {
                    cellValue_ = cellValue;
                    blockCounter_->markOccupied();
                }

                // Notify related groups of value change
                relatedGroups_->broadcast(cellPos, cellValue);

                // Related groups record value change
                GroupCount groupCount = relatedGroups_->getGroupCount();
                for (GroupCount groupIter = 0; groupIter < groupCount; ++groupIter) {
                    CellValue receivedValue = relatedGroups_->getBroadcast(groupIter);
                    if (receivedValue > 0) {
                        blockCounter_->block(receivedValue);
                    }
                }
            }

            __device__ void clearCellValue(CellCount cellPos)
            {
                // Notify related groups of value change.
                relatedGroups_->broadcast(cellPos, cellValue_);

                // Related groups record value change
                GroupCount groupCount = relatedGroups_->getGroupCount();
                for (GroupCount groupIter = 0; groupIter < groupCount; ++groupIter) {
                    CellValue receivedValue = relatedGroups_->getBroadcast(groupIter);
                    if (receivedValue > 0) {
                        blockCounter_->unblock(receivedValue);
                    }
                }

                // Update cell value
                if (threadIdx.x == cellPos) {
                    cellValue_ = 0;
                    blockCounter_->markFree();
                }
            }
        };
    }
}

#endif
