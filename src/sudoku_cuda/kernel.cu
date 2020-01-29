#include <algorithm>
#include <cassert>
#include <sudoku/cuda/error.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/cuda/grid.h>

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
                GuessStack guesses(data.guessStackData, threadIdx.x);
                data.results[threadIdx.x] = (threadIdx.x % 2 ? Result::OK_FOUND_SOLUTION : Result::OK_TIMED_OUT);
            }

            void launch(unsigned blockCount, unsigned threadsPerBlock, const DeviceData& deviceData)
            {
                kernel<<<blockCount, threadsPerBlock>>>(deviceData.getData());
                ErrorCheck() << cudaGetLastError();
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

            Result DeviceData::getResult(size_t threadNum)
            {
                deviceResults_.copyToHost();
                return deviceResults_.getHostData()[threadNum];
            }
        }
    }
}
