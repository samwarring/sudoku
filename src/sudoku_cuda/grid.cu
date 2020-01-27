#include <sudoku/cuda/grid.h>

namespace sudoku
{
    namespace cuda
    {
        CUDA_HOST_AND_DEVICE
        Grid::Grid(const Dimensions& dims, KernelParams kernelParams, size_t threadNum) 
            : dims_(&dims)
            , cellValues_(kernelParams.cellValues + (threadNum * dims.getCellCount()))
            , restrictions_(kernelParams.restrictions + kernelParams.restrictionsOffsets[threadNum])
        {
        }

        CUDA_HOST_AND_DEVICE
        size_t Grid::getCellValue(size_t cellPos)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            return cellValues_[cellPos];
        }

        CUDA_HOST_AND_DEVICE
        void Grid::setCellValue(size_t cellPos, size_t cellValue)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            CUDA_HOST_ASSERT(0 < cellValue && cellValue <= dims_->getMaxCellValue());
            cellValues_[cellPos] = cellValue;
        }

        CUDA_HOST_AND_DEVICE
        void Grid::clearCellValue(size_t cellPos)
        {
            CUDA_HOST_ASSERT(cellPos < dims_->getCellCount());
            cellValues_[cellPos] = 0;
        }
    }
}
