#ifndef INCLUDED_SUDOKU_CUDA_BLOCK_COUNTER_CUH
#define INCLUDED_SUDOKU_CUDA_BLOCK_COUNTER_CUH

#include <sudoku/cuda/types.h>

namespace sudoku
{
    namespace cuda
    {
        template <unsigned MAX_CELL_VALUE>
        class BlockCounter
        {
        private:
            CellBlockCount*  globalCellBlockCounts_;
            ValueBlockCount* globalValueBlockCounts_;
            CellBlockCount   cellBlockCount_;
            ValueBlockCount  valueBlockCounts_[MAX_CELL_VALUE];
            CellCount        cellCount_;
            CellValue        maxCellValue_;

            __device__ unsigned getValueBlockCountOffset(CellValue cellValue)
            {
                return (cellCount_ * (cellValue-1)) + threadIdx.x;
            }

        public:
            struct Pair
            {
                CellCount cellPos;
                CellBlockCount cellBlockCount;
            };

            static unsigned getValueBlockCountOffset(CellCount cellCount, CellCount cellPos, CellValue cellValue)
            {
                return (cellCount * (cellValue - 1)) + cellPos;
            }
            
            __device__ BlockCounter(CellCount cellCount, CellValue maxCellValue,
                                    CellBlockCount* globalCellBlockCounts,
                                    ValueBlockCount* globalValueBlockCounts)
            {
                // Read block count + value block counts from global memory.
                globalCellBlockCounts_ = globalCellBlockCounts;
                globalValueBlockCounts_ = globalValueBlockCounts;
                cellCount_ = cellCount;
                maxCellValue_ = maxCellValue;
                cellBlockCount_ = -1;
                if (threadIdx.x < cellCount_) {
                    cellBlockCount_ = globalCellBlockCounts_[threadIdx.x];
                    for (CellValue cellValue = 1; cellValue <= maxCellValue_; ++cellValue) {
                        unsigned offset = getValueBlockCountOffset(cellValue);
                        valueBlockCounts_[cellValue - 1] = globalValueBlockCounts_[offset];
                    }
                }
            }

            __device__ ~BlockCounter()
            {
                // Write current block count + value block counts back to global memory.
                if (threadIdx.x < cellCount_) {
                    globalCellBlockCounts_[threadIdx.x] = cellBlockCount_;
                    for (CellValue cellValue = 1; cellValue < maxCellValue_; ++cellValue) {
                        unsigned offset = getValueBlockCountOffset(cellValue);
                        globalValueBlockCounts_[offset] = valueBlockCounts_[cellValue - 1];
                    }
                }
            }

            __device__ void block(CellValue cellValue)
            {
                valueBlockCounts_[cellValue - 1]++;
                if (valueBlockCounts_[cellValue - 1] == 1) {
                    cellBlockCount_++;
                }
            }

            __device__ void unblock(CellValue cellValue)
            {
                valueBlockCounts_[cellValue - 1]--;
                if (valueBlockCounts_[cellValue - 1] == 0) {
                    cellBlockCount_--;
                }
            }

            __device__ int nextAvailableValue(CellValue minCellValue) const
            {
                CellValue cellValue = minCellValue + 1;
                for (; cellValue <= maxCellValue_; ++cellValue) {
                    if (valueBlockCounts_[cellValue - 1] == 0) {
                        break;
                    }
                }
                return cellValue;
            }

            __device__ void markOccupied()
            {
                // Occupied cells have negative cell block counts.
                cellBlockCount_ -= (maxCellValue_ + 1);
            }

            __device__ void markFree()
            {
                // Free cells have positive cell block counts.
                cellBlockCount_ += (maxCellValue_ + 1);
            }

            /// Require len(sharedBuffer) == blockDim.x == power of 2.
            __device__ Pair getMaxCellBlockCountPair(Pair* sharedBuffer) const
            {
                Pair myPair{ threadIdx.x, cellBlockCount_ };
                sharedBuffer[threadIdx.x] = myPair;
                __syncthreads();

                for (unsigned offset = (blockDim.x >> 1); offset != 0; offset >>= 1) {
                    if (threadIdx.x < offset) {
                        Pair lhs = sharedBuffer[threadIdx.x];
                        Pair rhs = sharedBuffer[threadIdx.x + offset];
                        if (rhs.cellBlockCount > lhs.cellBlockCount) {
                            sharedBuffer[threadIdx.x] = rhs;
                        }
                    }
                    __syncthreads();
                }

                return sharedBuffer[0];
            }
        };
    }
}

#endif
