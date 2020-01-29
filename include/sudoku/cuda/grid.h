#ifndef INCLUDED_SUDOKU_CUDA_GRID_H
#define INCLUDED_SUDOKU_CUDA_GRID_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/cuda/result.h>
#include <sudoku/grid.h>

namespace sudoku
{
    namespace cuda
    {
        class Grid
        {
            public:
                /// Provided to kernel.
                struct Data
                {
                    size_t* cellValues;           ///< e.g. { (grid0 cells, 81 values) (grid2 cells, 81 values) ... }
                    size_t* restrictions;         ///< e.g. { (grid0 restrictions, 2 pairs=4values) ... }
                    size_t* restrictionsOffsets;  ///< e.g. { (grid0 restrictions offset=0) ... }
                    size_t* blockCounts;          ///< e.g. { [grid0,cell0,blockCount] [grid0,cell0,value1count] ... }
                };

                // Forward declaration required to use as friend.
                class DeviceData;

                class HostData
                {
                    /// Allow DeviceData to copy the private buffers.
                    friend class DeviceData;

                    public:
                        HostData(const std::vector<sudoku::Grid>& grids);
                        Data getData() const { return data_; }

                    private:
                        void serialize(const std::vector<sudoku::Grid>& grids);
                        std::vector<size_t> cellValues_;
                        std::vector<size_t> restrictions_;
                        std::vector<size_t> restrictionsOffsets_;
                        std::vector<size_t> blockCounts_;
                        Data data_;
                };

                class DeviceData
                {
                    public:
                        DeviceData(const HostData& hostData);
                        Data getData() const { return data_; }

                    private:
                        DeviceBuffer<size_t> cellValues_;
                        DeviceBuffer<size_t> restrictions_;
                        DeviceBuffer<size_t> restrictionsOffsets_;
                        DeviceBuffer<size_t> blockCounts_;
                        Data data_;
                };

            public:
                CUDA_HOST_AND_DEVICE
                Grid(const Dimensions& dims, Data data, size_t threadNum)
                    : dims_(&dims), data_(data), threadNum_(threadNum) {}

                CUDA_HOST_AND_DEVICE
                size_t getCellValue(size_t cellPos);

                CUDA_HOST_AND_DEVICE
                void setCellValue(size_t cellPos, size_t cellValue);

                CUDA_HOST_AND_DEVICE
                void clearCellValue(size_t cellPos);

            private:
                /**
                 * Assign value to a cell (without updating block counts)
                 */
                CUDA_HOST_AND_DEVICE
                size_t* getCell(size_t cellPos);

                /**
                 * Get number of times the given value has been blocked by a
                 * related cell.
                 */
                CUDA_HOST_AND_DEVICE
                size_t* getBlockCount(size_t cellPos, size_t cellValue);

                /**
                 * Get the number of values that are blocked for the cell.
                 */
                CUDA_HOST_AND_DEVICE
                size_t* getBlockCount(size_t cellPos);

                CUDA_HOST_AND_DEVICE
                void blockCellValue(size_t cellPos, size_t cellValue);

                CUDA_HOST_AND_DEVICE
                void unblockCellValue(size_t cellPos, size_t cellValue);

                CUDA_HOST_AND_DEVICE
                void blockRelatedCells(size_t cellPos, size_t cellValue);

                CUDA_HOST_AND_DEVICE
                void unblockRelatedCells(size_t cellPos, size_t cellValue);

                const Dimensions* dims_;
                Data data_;
                size_t threadNum_;
        };
    }
}

#endif
