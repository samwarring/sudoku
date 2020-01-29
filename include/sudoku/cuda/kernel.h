#ifndef INCLUDED_SUDOKU_CUDA_KERNEL_H
#define INCLUDED_SUDOKU_CUDA_KERNEL_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/cuda/grid.h>
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
            };

            class DeviceData;

            class HostData
            {
                friend class DeviceData;

                public:
                    HostData(const sudoku::Dimensions& dims, const std::vector<sudoku::Grid>& grids);
                    Data getData() const { return data_; }

                private:
                    Dimensions::HostData hostDims_;
                    Grid::HostData hostGrid_;
                    Data data_;
            };

            class DeviceData
            {
                public:
                    DeviceData(const HostData& hostData);
                    Data getData() const { return data_; }

                private:
                    Dimensions::DeviceData deviceDims_;
                    Grid::DeviceData deviceGrid_;
                    Data data_;
            };

            void launch(unsigned blockCount, unsigned threadsPerBlock, const DeviceData& deviceData);
        }
    }
}

#endif
