#ifndef INCLUDED_SUDOKU_CUDA_KERNEL_H
#define INCLUDED_SUDOKU_CUDA_KERNEL_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/mirror_buffer.h>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/cuda/grid.h>
#include <sudoku/cuda/result.h>
#include <sudoku/dimensions.h>
#include <sudoku/grid.h>

namespace sudoku
{
    namespace cuda
    {
        namespace kernel
        {
            struct Data
            {
                Dimensions::Data dimsData;
                Grid::Data gridData;
                Result* results;
            };

            class DeviceData;

            class HostData
            {
                friend class DeviceData;

                public:
                    HostData(const sudoku::Dimensions& dims, const std::vector<sudoku::Grid>& grids);
                    Data getData() const { return data_; }
                    Result getResult(size_t threadNum) const { return hostResults_[threadNum]; }

                private:
                    Dimensions::HostData hostDims_;
                    Grid::HostData hostGrid_;
                    std::vector<Result> hostResults_;
                    Data data_;
            };

            class DeviceData
            {
                public:
                    DeviceData(const HostData& hostData);
                    Data getData() const { return data_; }
                    Result getResult(size_t threadNum);

                private:
                    Dimensions::DeviceData deviceDims_;
                    Grid::DeviceData deviceGrid_;
                    MirrorBuffer<Result> deviceResults_;
                    Data data_;
            };

            void launch(unsigned blockCount, unsigned threadsPerBlock, const DeviceData& deviceData);
        }
    }
}

#endif
