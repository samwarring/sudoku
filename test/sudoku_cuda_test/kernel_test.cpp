#include <iostream>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/mirror_buffer.h>

namespace std
{
    std::ostream& operator<<(std::ostream& out, sudoku::cuda::Result result)
    {
        out << sudoku::cuda::toString(result);
        return out;
    }
}

BOOST_AUTO_TEST_CASE(kernel_standardDims)
{
    sudoku::cuda::MirrorBuffer<sudoku::cuda::Result> results(1);
    results.getHostData()[0] = sudoku::cuda::Result::ERROR_NOT_SET;
    results.copyToDevice();
    sudoku::cuda::computeNextSolutionKernelWrapper(1, 1, results.getDeviceData());
    results.copyToHost();
    BOOST_REQUIRE_EQUAL(results.getHostData()[0], sudoku::cuda::Result::OK_TIMED_OUT);
}
