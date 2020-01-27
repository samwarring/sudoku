#ifndef INCLUDED_SUDOKU_CUDA_MIRROR_BUFFER_H
#define INCLUDED_SUDOKU_CUDA_MIRROR_BUFFER_H

#include <stdexcept>
#include <sudoku/cuda/error.h>

namespace sudoku
{
    namespace cuda
    {
        template<typename T>
        class MirrorBuffer
        {
            public:
                MirrorBuffer(size_t itemCount)
                {
                    byteCount_ = sizeof(T) * itemCount;
                    hostData_ = new T[byteCount_];
                    errorCheck_ << cudaMalloc(&deviceData_, byteCount_);
                }

                ~MirrorBuffer()
                {
                    errorCheck_ << cudaFree(deviceData_);
                    delete [] hostData_;
                }

                MirrorBuffer(const MirrorBuffer&) = delete;
                MirrorBuffer(MirrorBuffer&&) = delete;
                MirrorBuffer& operator=(const MirrorBuffer&) = delete;
                MirrorBuffer& operator=(MirrorBuffer&&) = delete;

                void copyToDevice()
                {
                    errorCheck_ << cudaMemcpy(deviceData_, hostData_, byteCount_, cudaMemcpyHostToDevice);
                }

                void copyToHost()
                {
                    errorCheck_ << cudaMemcpy(hostData_, deviceData_, byteCount_, cudaMemcpyDeviceToHost);
                }

                T* getHostData()
                {
                    return hostData_;
                }

                T* getDeviceData()
                {
                    return deviceData_;
                }

                size_t getItemCount() const
                {
                    return byteCount_ / sizeof(T);
                }

                size_t getByteCount() const
                {
                    return byteCount_;
                }

            private:
                size_t byteCount_;
                T* hostData_;
                T* deviceData_;
                ErrorCheck errorCheck_;
        };
    }
}

#endif
