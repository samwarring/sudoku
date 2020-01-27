#ifndef INCLUDED_SUDOKU_CUDA_DIMENSIONS_H
#define INCLUDED_SUDOKU_CUDA_DIMENSIONS_H

#include <sudoku/cuda/host_and_device.h>
#include <sudoku/cuda/kernel.h>

namespace sudoku
{
    namespace cuda
    {
        class Dimensions
        {
            public:
                CUDA_HOST_AND_DEVICE
                Dimensions(KernelParams kernelParams);

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
                const size_t* groupValues_;
                const size_t* groupOffsets_;
                const size_t* groupsForCellValues_;
                const size_t* groupsForCellOffsets_;
        };
    }
}

#endif
