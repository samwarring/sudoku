#ifndef INCLUDED_SUDOKU_CUDA_ERROR_CHECK_H
#define INCLUDED_SUDOKU_CUDA_ERROR_CHECK_H

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

namespace sudoku
{
    namespace cuda
    {
        class CudaException : public std::runtime_error { using std::runtime_error::runtime_error; };

        class ErrorCheck
        {
            public:
                ErrorCheck(cudaError_t error = cudaSuccess)
                {
                    checkError(error);
                }

                ErrorCheck& operator=(cudaError_t error)
                {
                    checkError(error);
                    return *this;
                }

                static void lastError()
                {
                    checkError(cudaGetLastError());
                }

            private:
                static void checkError(cudaError_t error)
                {
                    if (error != cudaSuccess) {
                        std::ostringstream oss;
                        oss << "CUDA error " << error << ": " << cudaGetErrorString(error);
                        throw CudaException(oss.str());
                    }
                }
        };
    }
}

#endif
