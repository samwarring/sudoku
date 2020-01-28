#ifndef INCLUDED_SUDOKU_CUDA_KERNEL_H
#define INCLUDED_SUDOKU_CUDA_KERNEL_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/mirror_buffer.h>
#include <sudoku/cuda/result.h>
#include <sudoku/dimensions.h>
#include <sudoku/grid.h>

namespace sudoku
{
    namespace cuda
    {
        struct DimensionParams
        {
            size_t cellCount;
            size_t maxCellValue;
            size_t groupCount;
            std::vector<size_t> groupValues;
            std::vector<size_t> groupOffsets;
            std::vector<size_t> groupsForCellValues;
            std::vector<size_t> groupsForCellOffsets;
            
            DimensionParams(const sudoku::Dimensions& dims);
        };

        struct GridParams
        {
            std::vector<size_t> cellValues;
            std::vector<size_t> restrictions;
            std::vector<size_t> restrictionsOffsets;
            std::vector<size_t> blockCounts;

            GridParams(const std::vector<sudoku::Grid> grids);
        };

        struct KernelParams
        {
            size_t cellCount;             ///< e.g. 81
            size_t maxCellValue;          ///< e.g. 9
            size_t groupCount;            ///< e.g. 27
            const size_t* groupValues;          ///< e.g. { (group0, 9 values) (group1 9 values) ... }
            const size_t* groupOffsets;         ///< e.g. { [group0 offset=0] [group1 offset=9] ... [group27 offset=243] }
            const size_t* groupsForCellValues;  ///< e.g. { (cell0 groups, 3 values) (cell1 groups, 3 values) ... }
            const size_t* groupsForCellOffsets; ///< e.g. { [cell0 offset=0] [cell1 offset=3] ... [cell81 offset=243] }
            size_t* cellValues;           ///< e.g. { (grid0 cells, 81 values) (grid2 cells, 81 values) ... }
            size_t* restrictions;         ///< e.g. { (grid0 restrictions, 2 pairs=4values) ... }
            size_t* restrictionsOffsets;  ///< e.g. { (grid0 restrictions offset=0) ... }
            size_t* blockCounts;          ///< e.g. { [grid0,cell0,blockCount] [grid0,cell0,value1count] ... }
            Result* results;              ///< e.g. { Result::OK_TIMED_OUT, Result::OK_FOUND_SOLUTION, ... }
        };

        class DeviceKernelParams
        {
            public:
                DeviceKernelParams(DimensionParams dimParams, GridParams gridParams, size_t threadCount);

                KernelParams getKernelParams() const { return kernelParams_; }

                Result getThreadResult(size_t threadNum);

            private:
                DeviceBuffer<size_t> groupValues_;
                DeviceBuffer<size_t> groupOffsets_;
                DeviceBuffer<size_t> groupsForCellValues_;
                DeviceBuffer<size_t> groupsForCellOffsets_;
                DeviceBuffer<size_t> cellValues_;
                DeviceBuffer<size_t> restrictions_;
                DeviceBuffer<size_t> restrictionsOffsets_;
                DeviceBuffer<size_t> blockCounts_;
                MirrorBuffer<Result> results_;
                KernelParams kernelParams_;
        };

        void kernelWrapper(unsigned blockCount, unsigned threadsPerBlock, KernelParams params);
    }
}

#endif
