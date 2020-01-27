#include <vector>
#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/grid.h>
#include <sudoku/standard.h>
#include <sudoku/fork.h>
#include "util.h"

/// State for each thread
struct Thread
{
    sudoku::cuda::Dimensions dims;
    sudoku::cuda::Grid grid;

    Thread(sudoku::cuda::KernelParams kernelParams, size_t threadNum)
        : dims(kernelParams)
        , grid(dims, kernelParams, threadNum) {}
};

BOOST_AUTO_TEST_CASE(grid_test)
{
    // Make params
    const size_t threadCount = 4;
    sudoku::standard::Dimensions cpuDims;
    sudoku::cuda::DimensionParams dimParams(cpuDims);
    sudoku::cuda::GridParams gridParams(sudoku::fork(cpuDims, threadCount));
    sudoku::cuda::KernelParams kernelParams = makeHostParams(dimParams, gridParams);

    // Make in-kernel objects for each thread
    std::vector<Thread> threads;
    for (size_t tn = 0; tn < threadCount; ++tn) {
        threads.emplace_back(kernelParams, tn);
    }
}
