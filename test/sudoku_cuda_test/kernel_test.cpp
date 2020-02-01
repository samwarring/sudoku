#include <algorithm>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/fork.h>
#include <sudoku/standard.h>

BOOST_AUTO_TEST_CASE(Kernel_standardDims_emptyFork_4threads)
{
    unsigned threadCount = 10;
    sudoku::square::Dimensions dims(3);
    auto grids = sudoku::fork(dims, threadCount);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    sudoku::cuda::kernel::DeviceData deviceData(hostData);
    sudoku::cuda::kernel::launch(1, threadCount, deviceData);
    std::vector<sudoku::cuda::Result> expectedResults, actualResults;
    std::vector<size_t> expectedEmptyCellCounts, actualEmptyCellCounts;
    for (size_t threadNum = 0; threadNum < threadCount; ++threadNum) {
        actualResults.push_back(deviceData.getResult(threadNum));
        expectedResults.push_back(sudoku::cuda::Result::OK_FOUND_SOLUTION);
        auto solution = deviceData.getCellValues(threadNum);
        actualEmptyCellCounts.push_back(std::count(solution.begin(), solution.end(), 0));
        expectedEmptyCellCounts.push_back(0);
    }
    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        expectedResults.begin(), expectedResults.end(),
        actualResults.begin(), actualResults.end()
    );
    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        expectedEmptyCellCounts.begin(), expectedEmptyCellCounts.end(),
        actualEmptyCellCounts.begin(), actualEmptyCellCounts.end()
    );
}
