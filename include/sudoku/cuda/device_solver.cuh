#ifndef INCLUDED_SUDOKU_CUDA_DEVICE_SOLVER_CUH
#define INCLUDED_SUDOKU_CUDA_DEVICE_SOLVER_CUH

#include <sudoku/cuda/types.h>
#include <sudoku/cuda/block_counter.cuh>
#include <sudoku/cuda/grid.cuh>
#include <sudoku/cuda/guess_stack.cuh>
#include <sudoku/cuda/related_groups.cuh>

namespace sudoku
{
    namespace cuda
    {
        template <CellValue MAX_CELL_VALUE, GroupCount MAX_GROUPS_FOR_CELL>
        class DeviceSolver
        {
        public:
            using BlockCounter = BlockCounter<MAX_CELL_VALUE>;
            using RelatedGroups = RelatedGroups<MAX_GROUPS_FOR_CELL>;
            using Grid = Grid<MAX_CELL_VALUE, MAX_GROUPS_FOR_CELL>;

        private:
            RelatedGroups* relatedGroups_;
            BlockCounter*  blockCounter_;
            Grid*          grid_;
            GuessStack*    guessStack_;
            CellCount      cellCount_;
            CellValue      maxCellValue_;
            CellValue*     sharedBroadcastValue_;

        public:
            __device__ DeviceSolver(RelatedGroups& relatedGroups, BlockCounter& blockCounter, Grid& grid,
                                    GuessStack& guessStack, CellCount cellCount, CellValue maxCellValue,
                                    CellValue* sharedBroadcastValue)
                : relatedGroups_(&relatedGroups)
                , blockCounter_(&blockCounter)
                , grid_(&grid)
                , guessStack_(&guessStack)
                , cellCount_(cellCount)
                , maxCellValue_(maxCellValue)
                , sharedBroadcastValue_(sharedBroadcastValue)
            {}

            __device__ CellCount nextCellPosition()
            {
                auto pair = blockCounter_->getMaxCellBlockCountPair();
                if (pair.cellBlockCount < 0) {
                    return cellCount_;
                }
                return pair.cellPos;
            }

            __device__ CellValue nextAvailableValue(CellCount cellPos, CellValue minCellValue)
            {
                if (threadIdx.x == cellPos) {
                    *sharedBroadcastValue_ = blockCounter_->nextAvailableValue(minCellValue);
                }
                __syncthreads();
                return *sharedBroadcastValue_;
            }

            __device__ unsigned computeNextSolution(unsigned guessCount, Result& result)
            {
                CellCount cellPos = nextCellPosition();
                CellValue minCellValue = 0;

                while(cellPos < cellCount_) {

                    if (guessCount == 0) {
                        result = Result::TIMED_OUT;
                        return guessCount;
                    }

                    CellValue cellValue = nextAvailableValue(cellPos, minCellValue);
                    if (cellValue > maxCellValue_) {
                        if (guessStack_->getSize() == 0) {
                            result = Result::NO_SOLUTION;
                            return guessCount;
                        }
                        auto prevGuess = guessStack_->pop();
                        cellPos = prevGuess.cellPos;
                        minCellValue = prevGuess.cellValue;
                        grid_->clearCellValue(cellPos);
                        continue;
                    }

                    grid_->setCellValue(cellPos, cellValue);
                    guessStack_->push(cellPos, cellValue);
                    cellPos = nextCellPosition();
                    minCellValue = 0;
                    guessCount--;
                }

                result = Result::FOUND_SOLUTION;
                return guessCount;
            }
        };
    }
}

#endif
