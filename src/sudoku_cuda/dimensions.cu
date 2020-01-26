#include <vector>
#include <sudoku/cuda/dimensions.h>

namespace sudoku
{
    namespace cuda
    {
        std::vector<char> Dimensions::serialize(const sudoku::Dimensions& dims)
        {
            return {};
        }

        CUDA_HOST_AND_DEVICE
        Dimensions::Dimensions(void* serializedDimensions, size_t serializedDimsSize)
        {
        }

        CUDA_HOST_AND_DEVICE
        bool Dimensions::isValid() const
        {
            return false;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getCellCount() const
        {
            return 0;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getMaxCellValue() const
        {
            return 0;
        }

        CUDA_HOST_AND_DEVICE
        size_t Dimensions::getGroupCount() const
        {
            return 0;
        }
    }
}