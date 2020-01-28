#include <iostream>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/mirror_buffer.h>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/standard.h>

namespace std
{
    std::ostream& operator<<(std::ostream& out, sudoku::cuda::Result result)
    {
        out << sudoku::cuda::toString(result);
        return out;
    }
}

BOOST_AUTO_TEST_CASE(DeviceKernelParams_4threads)
{
    unsigned threadCount = 4;
    sudoku::standard::Dimensions dims;
    std::vector<sudoku::Grid> grids(threadCount, dims);
    sudoku::cuda::DeviceKernelParams params(dims, grids, threadCount);
    sudoku::cuda::kernelWrapper(1, threadCount, params.getKernelParams());
    BOOST_REQUIRE_EQUAL(params.getThreadResult(0), sudoku::cuda::Result::OK_TIMED_OUT);
}
