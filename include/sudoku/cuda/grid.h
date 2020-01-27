#ifndef INCLUDED_SUDOKU_CUDA_GRID_H
#define INCLUDED_SUDOKU_CUDA_GRID_H

#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/dimensions.h>

namespace sudoku
{
    namespace cuda
    {
        class Grid
        {
            public:
                CUDA_HOST_AND_DEVICE
                Grid(const Dimensions& dims, KernelParams kernelParams, size_t threadNum);

                CUDA_HOST_AND_DEVICE
                size_t getCellValue(size_t cellPos);

                CUDA_HOST_AND_DEVICE
                void setCellValue(size_t cellPos, size_t cellValue);

                CUDA_HOST_AND_DEVICE
                void clearCellValue(size_t cellPos);

            private:
                const Dimensions* dims_;
                size_t* cellValues_;
                size_t* restrictions_;
        };
    }
}

#endif
