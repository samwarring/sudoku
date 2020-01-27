#ifndef INCLUDED_SUDOKU_CUDA_ERROR_H
#define INCLUDED_SUDOKU_CUDA_ERROR_H

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

namespace sudoku
{
    namespace cuda
    {
        class Error : public std::runtime_error { using std::runtime_error::runtime_error; };

        class ErrorCheck
        {
            public:
                void operator<<(cudaError_t error)
                {
                    if (error != cudaSuccess) {
                        std::ostringstream oss;
                        oss << "CUDA error " << error;
                        throw Error(oss.str());
                    }
                }
        };
    }
}

#endif
