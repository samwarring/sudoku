#ifndef INCLUDED_SUDOKU_CUDA_MIRROR_BUFFER_H
#define INCLUDED_SUDOKU_CUDA_MIRROR_BUFFER_H

#include <vector>
#include <sudoku/cuda/error.h>
#include <sudoku/cuda/device_buffer.h>

namespace sudoku
{
    namespace cuda
    {
        template<typename T>
        class MirrorBuffer
        {
            public:
                MirrorBuffer(size_t itemCount) : hostData_(itemCount), deviceData_(itemCount) {}

                MirrorBuffer(std::vector<T> items) : hostData_(std::move(items)), deviceData_(hostData_.size())
                {
                    copyToDevice();
                }

                MirrorBuffer(const MirrorBuffer&) = delete;
                MirrorBuffer(MirrorBuffer&&) = delete;
                MirrorBuffer& operator=(const MirrorBuffer&) = delete;
                MirrorBuffer& operator=(MirrorBuffer&&) = delete;

                void copyToDevice()
                {
                    ErrorCheck() << cudaMemcpy(deviceData_.begin(), hostData_.data(),
                                               sizeof(T) * hostData_.size(),
                                               cudaMemcpyHostToDevice);
                }

                void copyToHost()
                {
                    ErrorCheck() << cudaMemcpy(hostData_.data(), deviceData_.begin(),
                                               sizeof(T) * hostData_.size(),
                                               cudaMemcpyDeviceToHost);
                }

                T* getHostData()
                {
                    return hostData_.data();
                }

                const T* getHostData() const
                {
                    return hostData_.data();
                }

                T* getDeviceData()
                {
                    return deviceData_.begin();
                }

                size_t getSize()
                {
                    return hostData_.size();
                }

            private:
                std::vector<T> hostData_;
                DeviceBuffer<T> deviceData_;
        };
    }
}

#endif
