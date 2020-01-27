#ifndef INCLUDED_SUDOKU_CUDA_DIMENSIONS_H
#define INCLUDED_SUDOKU_CUDA_DIMENSIONS_H

#include <sudoku/cuda/host_and_device.h>
#include <sudoku/cuda/kernel.h>

namespace sudoku
{
    namespace cuda
    {
        namespace compute_next_solution_kernel
        {
            class Dimensions
            {
                public:
                    CUDA_HOST_AND_DEVICE
                    Dimensions(Params kernelParams);

                    CUDA_HOST_AND_DEVICE
                    size_t getCellCount() const;

                    CUDA_HOST_AND_DEVICE
                    size_t getMaxCellValue() const;

                    CUDA_HOST_AND_DEVICE
                    size_t getGroupCount() const;

                    CUDA_HOST_AND_DEVICE
                    size_t getCellsInGroupCount(size_t groupNum) const;

                    CUDA_HOST_AND_DEVICE
                    const size_t* getCellsInGroup(size_t groupNum) const;

                    CUDA_HOST_AND_DEVICE
                    size_t getGroupsForCellCount(size_t cellPos) const;

                    CUDA_HOST_AND_DEVICE
                    const size_t* getGroupsForCell(size_t cellPos) const;

                private:
                    size_t cellCount_;
                    size_t maxCellValue_;
                    size_t groupCount_;
                    size_t* groupValues_;
                    size_t* groupOffsets_;
                    size_t* groupsForCellValues_;
                    size_t* groupsForCellOffsets_;
            };
        }
    }
}

#endif
