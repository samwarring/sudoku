#ifndef INCLUDED_SUDOKU_CUDA_DEVICE_BUFFER_H
#define INCLUDED_SUDOKU_CUDA_DEVICE_BUFFER_H

#include <vector>
#include <sudoku/cuda/error_check.h>

namespace sudoku
{
    namespace cuda
    {
        template <typename T>
        class DeviceBuffer
        {
            public:
                DeviceBuffer(size_t itemCount) : memSize_(sizeof(T) * itemCount)
                {
                    ErrorCheck ec;
                    ec = cudaMalloc(&deviceMem_, memSize_);
                }

                DeviceBuffer(const std::vector<T> hostBuffer) : DeviceBuffer(hostBuffer.size())
                {
                    ErrorCheck ec;
                    ec = cudaMemcpy(deviceMem_, hostBuffer.data(), memSize_, cudaMemcpyHostToDevice);
                }

                T* get() { return deviceMem_; }

                const T* get() const { return deviceMem_; }

                std::vector<T> copyToHost() const
                {
                    ErrorCheck ec;
                    std::vector<T> hostBuffer(memSize_ / sizeof(T));
                    ec = cudaMemcpy(hostBuffer.data(), deviceMem_, memSize_, cudaMemcpyDeviceToHost);
                    return hostBuffer;
                }

                size_t size() const { return memSize_ / sizeof(T); }

            private:
                size_t memSize_;
                T* deviceMem_;
        };
    }
}

#endif
