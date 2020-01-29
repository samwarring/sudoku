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
                size_t threadNum = threadIdx.x;
                Dimensions dims(data.dimsData);
                Grid grid(dims, data.gridData, threadNum);
            }
    
            void launch(unsigned blockCount, unsigned threadsPerBlock, const DeviceData& deviceData)
            {
                kernel<<<blockCount, threadsPerBlock>>>(deviceData.getData());
                ErrorCheck() << cudaGetLastError();
            }
    
            HostData::HostData(const sudoku::Dimensions& dims, const std::vector<sudoku::Grid>& grids)
                : hostDims_(dims)
                , hostGrid_(grids)
            {
                data_.dimsData = hostDims_.getData();
                data_.gridData = hostGrid_.getData();
            }
    
            DeviceData::DeviceData(const HostData& hostData)
                : deviceDims_(hostData.hostDims_)
                , deviceGrid_(hostData.hostGrid_)
            {
                data_.dimsData = deviceDims_.getData();
                data_.gridData = deviceGrid_.getData();
            }
        }
    }
}
