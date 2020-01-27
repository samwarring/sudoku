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

BOOST_AUTO_TEST_CASE(kernel_standardDims_2threads)
{
    const size_t threadCount = 2;

    // Set up dimensions.
    sudoku::standard::Dimensions dims;
    sudoku::cuda::compute_next_solution_kernel::DimensionParams dimParams(dims);
    sudoku::cuda::DeviceBuffer<size_t> groupValues(dimParams.groupValues);
    sudoku::cuda::DeviceBuffer<size_t> groupOffsets(dimParams.groupOffsets);
    sudoku::cuda::DeviceBuffer<size_t> groupsForCellValues(dimParams.groupsForCellValues);
    sudoku::cuda::DeviceBuffer<size_t> groupsForCellOffsets(dimParams.groupsForCellOffsets);

    // Set up grids.
    std::vector<sudoku::Grid> grids;
    for (size_t tn = 0; tn < threadCount; ++tn) {
        grids.emplace_back(dims);
    }
    sudoku::cuda::compute_next_solution_kernel::GridParams gridParams(grids);
    sudoku::cuda::DeviceBuffer<size_t> cellValues(gridParams.cellValues);
    sudoku::cuda::DeviceBuffer<size_t> blockCounts(gridParams.blockCounts);
    sudoku::cuda::DeviceBuffer<size_t> restrictions(gridParams.restrictions);
    sudoku::cuda::DeviceBuffer<size_t> restrictionsOffsets(gridParams.restrictionsOffsets);

    // Set up results.
    sudoku::cuda::MirrorBuffer<sudoku::cuda::Result> results(threadCount);
    results.copyToDevice();

    // Set up params.
    sudoku::cuda::compute_next_solution_kernel::Params params;
    params.cellCount = dimParams.cellCount;
    params.maxCellValue = dimParams.maxCellValue;
    params.groupCount = dimParams.groupCount;
    params.groupValues = groupValues.begin();
    params.groupOffsets = groupOffsets.begin();
    params.groupsForCellValues = groupsForCellValues.begin();
    params.groupsForCellOffsets = groupsForCellOffsets.begin();
    params.cellValues = cellValues.begin();
    params.restrictions = restrictions.begin();
    params.restrictionsOffsets = restrictionsOffsets.begin();
    params.blockCounts = blockCounts.begin();
    params.results = results.getDeviceData();
    
    // Execute kernel.
    sudoku::cuda::compute_next_solution_kernel::kernelWrapper(1, threadCount, params);

    // Copy results back to host.
    results.copyToHost();

    // Compare results to expected value.
    BOOST_REQUIRE_EQUAL(results.getHostData()[0], sudoku::cuda::Result::OK_TIMED_OUT);
}
