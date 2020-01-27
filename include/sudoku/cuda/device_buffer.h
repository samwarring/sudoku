#ifndef INCLUDED_SUDOKU_CUDA_DEVICE_BUFFER_H
#define INCLUDED_SUDOKU_CUDA_DEVICE_BUFFER_H

#include <vector>
#include <cuda_runtime.h>
#include <sudoku/cuda/error.h>

namespace sudoku
{
    namespace cuda
    {
        template<typename T>
        class DeviceBuffer
        {
            public:
                DeviceBuffer(const std::vector<T>& hostBuffer) : DeviceBuffer(hostBuffer.size())
                {
                    ErrorCheck() << cudaMemcpy(data_, hostBuffer.data(), sizeof(T) * size_, cudaMemcpyHostToDevice);
                }

                DeviceBuffer(size_t size) : size_(size)
                {
                    ErrorCheck() << cudaMalloc(&data_, sizeof(T) * size);
                }

                ~DeviceBuffer() { ErrorCheck() << cudaFree(data_); }

                size_t size() { return size_; }

                T* begin() { return data_; }

                T* end() { return data_ + size_; }

                DeviceBuffer(const DeviceBuffer<T>&) = delete;
                DeviceBuffer(DeviceBuffer<T>&&) = delete;
                DeviceBuffer<T>& operator=(const DeviceBuffer<T>&) = delete;
                DeviceBuffer<T>& operator=(DeviceBuffer<T>&&) = delete;

            private:
                T* data_;
                size_t size_;
        };
    }
}

#endif
