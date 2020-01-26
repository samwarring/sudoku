#ifndef INCLUDED_SUDOKU_CUDA_DIMENSIONS_H
#define INCLUDED_SUDOKU_CUDA_DIMENSIONS_H

#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/cuda/host_and_device.h>

namespace sudoku
{
    namespace cuda
    {
        class Dimensions
        {
            public:

                /**
                 * Called from host to convert a fragmented sudoku::Dimenions object
                 * to a serialized buffer to be read by the GPU device.
                 */
                static std::vector<char> serialize(const sudoku::Dimensions& dims);

                CUDA_HOST_AND_DEVICE
                Dimensions(void* serializedDimensions, size_t serializedDimsSize);

                CUDA_HOST_AND_DEVICE
                bool isValid() const;

                CUDA_HOST_AND_DEVICE
                size_t getCellCount() const;

                CUDA_HOST_AND_DEVICE
                size_t getMaxCellValue() const;

                CUDA_HOST_AND_DEVICE
                size_t getGroupCount() const;
        };
    }
}

#endif