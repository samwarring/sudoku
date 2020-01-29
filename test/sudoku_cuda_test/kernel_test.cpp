#include <iostream>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/standard.h>

BOOST_AUTO_TEST_CASE(DeviceKernelParams_4threads)
{
    unsigned threadCount = 4;
    sudoku::standard::Dimensions dims;
    std::vector<sudoku::Grid> grids(threadCount, dims);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    sudoku::cuda::kernel::DeviceData deviceData(hostData);
    sudoku::cuda::kernel::launch(1, threadCount, deviceData);
    BOOST_REQUIRE_EQUAL(deviceData.getResult(0), sudoku::cuda::Result::OK_TIMED_OUT);
    BOOST_REQUIRE_EQUAL(deviceData.getResult(1), sudoku::cuda::Result::OK_FOUND_SOLUTION);
    BOOST_REQUIRE_EQUAL(deviceData.getResult(2), sudoku::cuda::Result::OK_TIMED_OUT);
    BOOST_REQUIRE_EQUAL(deviceData.getResult(3), sudoku::cuda::Result::OK_FOUND_SOLUTION);
}
