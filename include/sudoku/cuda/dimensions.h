#ifndef INCLUDED_SUDOKU_CUDA_DIMENSIONS_H
#define INCLUDED_SUDOKU_CUDA_DIMENSIONS_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/host_and_device.h>
#include <sudoku/dimensions.h>

namespace sudoku
{
    namespace cuda
    {
        class Dimensions
        {
            public:
                /// Provided to kernel.
                struct Data
                {
                    size_t cellCount;
                    size_t maxCellValue;
                    size_t groupCount;
                    const size_t* groupValues;          ///< e.g. { (group0, 9 values) (group1 9 values) ... }
                    const size_t* groupOffsets;         ///< e.g. { [group0 offset=0] [group1 offset=9] ... [group27 offset=243] }
                    const size_t* groupsForCellValues;  ///< e.g. { (cell0 groups, 3 values) (cell1 groups, 3 values) ... }
                    const size_t* groupsForCellOffsets; ///< e.g. { [cell0 offset=0] [cell1 offset=3] ... [cell81 offset=243] }
                };

                class HostData;
                class DeviceData;

            public:
                CUDA_HOST_AND_DEVICE
                Dimensions(Data data) : data_(data) {}

                CUDA_HOST_AND_DEVICE
                size_t getCellCount() const;

                CUDA_HOST_AND_DEVICE
                size_t getMaxCellValue() const;

                CUDA_HOST_AND_DEVICE
                size_t getGroupCount() const;

                CUDA_HOST_AND_DEVICE
                size_t getCellsInGroupCount(size_t groupNum) const;

                CUDA_HOST_AND_DEVICE
                size_t getCellInGroup(size_t groupNum, size_t itemNum) const;

                CUDA_HOST_AND_DEVICE
                size_t getGroupsForCellCount(size_t cellPos) const;

                CUDA_HOST_AND_DEVICE
                size_t getGroupForCell(size_t cellPos, size_t itemNum) const;

            private:
                Data data_;
        };

        class Dimensions::HostData
        {
            /// Allow DeviceData to copy the private buffers.
            friend class DeviceData;

            public:
                HostData(const sudoku::Dimensions& dims);
                Data getData() const { return data_; }

            private:
                void serialize(const sudoku::Dimensions& dims);
                std::vector<size_t> groupValues_;
                std::vector<size_t> groupOffsets_;
                std::vector<size_t> groupsForCellValues_;
                std::vector<size_t> groupsForCellOffsets_;
                Data data_;
        };

        class Dimensions::DeviceData
        {
            public:
                DeviceData(const HostData& hostData);
                Data getData() const { return data_; }

            private:
                DeviceBuffer<size_t> groupValues_;
                DeviceBuffer<size_t> groupOffsets_;
                DeviceBuffer<size_t> groupsForCellValues_;
                DeviceBuffer<size_t> groupsForCellOffsets_;
                Data data_;
        };
    }
}

#endif
