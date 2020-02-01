#include <algorithm>
#include <cassert>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/cuda/error.h>
#include <sudoku/cuda/grid.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/solver.h>

namespace sudoku
{
    namespace cuda
    {
        namespace kernel
        {
            __global__ static void kernel(Data data)
            {
                Dimensions dims(data.dimsData);
                Grid grid(dims, data.gridData, threadIdx.x);
                GuessStack guessStack(data.guessStackData, threadIdx.x);
                Solver solver(dims, grid, guessStack);
                data.results[threadIdx.x] = solver.computeNextSolution(100000);
            }

            void launch(unsigned blockCount, unsigned threadsPerBlock, DeviceData& deviceData)
            {
                kernel<<<blockCount, threadsPerBlock>>>(deviceData.getData());
                ErrorCheck() << cudaGetLastError();
                deviceData.copyToHost();
            }

            HostData::HostData(const sudoku::Dimensions& dims, const std::vector<sudoku::Grid>& grids)
                : hostDims_(dims)
                , hostGrid_(hostDims_.getData(), grids)
                , hostGuessStack_(grids)
                , hostResults_(grids.size(), Result::ERROR_NOT_SET)
            {
                data_.dimsData = hostDims_.getData();
                data_.gridData = hostGrid_.getData();
                data_.guessStackData = hostGuessStack_.getData();
                data_.results = hostResults_.data();
            }

            size_t HostData::getAllocatedSize() const
            {
                return (
                    hostDims_.getAllocatedSize() +
                    hostGrid_.getAllocatedSize() +
                    hostGuessStack_.getAllocatedSize() +
                    (sizeof(Result) * hostResults_.size())
                );
            }

            DeviceData::DeviceData(const HostData& hostData)
                : deviceDims_(hostData.hostDims_)
                , deviceGrid_(hostData.hostGrid_)
                , deviceGuessStack_(hostData.hostGuessStack_)
                , deviceResults_(hostData.hostResults_)
            {
                data_.dimsData = deviceDims_.getData();
                data_.gridData = deviceGrid_.getData();
                data_.guessStackData = deviceGuessStack_.getData();
                data_.results = deviceResults_.getDeviceData();
            }

            void DeviceData::copyToHost()
            {
                deviceGrid_.copyToHost();
                deviceResults_.copyToHost();
            }

            Result DeviceData::getResult(size_t threadNum) const
            {
                return deviceResults_.getHostData()[threadNum];
            }

            std::vector<size_t> DeviceData::getCellValues(size_t threadNum) const
            {
                return deviceGrid_.getCellValues(threadNum);
            }
        }
    }
}
