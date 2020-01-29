#ifndef INCLUDED_SUDOKU_CUDA_STACK_H
#define INCLUDED_SUDOKU_CUDA_STACK_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/host_and_device.h>
#include <sudoku/grid.h>

namespace sudoku
{
    namespace cuda
    {
        class GuessStack
        {
            public:
                struct Data
                {
                    size_t* values;
                    size_t* sizes;
                    size_t threadCount;
                };

                class HostData;
                class DeviceData;

            public:
                CUDA_HOST_AND_DEVICE
                GuessStack(Data data, size_t threadNum) : data_(data), threadNum_(threadNum) {}

                CUDA_HOST_AND_DEVICE
                void push(size_t cellPos);

                CUDA_HOST_AND_DEVICE
                size_t pop();

                CUDA_HOST_AND_DEVICE
                size_t getSize();

            private:
                CUDA_HOST_AND_DEVICE
                size_t* size();

                /**
                 * Returns a pointer to the NEXT available stack value - NOT a pointer
                 * to the latest value pushed on the stack.
                 */
                CUDA_HOST_AND_DEVICE
                size_t* top();

                Data data_;
                size_t threadNum_;
        };

        class GuessStack::HostData
        {
            friend class DeviceData;

            public:
                HostData(const std::vector<sudoku::Grid>& grids);
                Data getData() const { return data_; }

            private:
                std::vector<size_t> values_;
                std::vector<size_t> sizes_;
                Data data_;
        };

        class GuessStack::DeviceData
        {
            public:
                DeviceData(const HostData& hostData);
                Data getData() const { return data_; }

            private:
                DeviceBuffer<size_t> values_;
                DeviceBuffer<size_t> sizes_;
                Data data_;
        };
    }
}

#endif
