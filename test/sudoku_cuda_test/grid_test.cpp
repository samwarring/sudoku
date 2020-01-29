#include <vector>
#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/grid.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/standard.h>
#include <sudoku/fork.h>

BOOST_AUTO_TEST_CASE(Grid_singleEmptyGrid)
{
    sudoku::standard::Dimensions dims;
    sudoku::cuda::kernel::HostData hostData(dims, {dims});
    sudoku::cuda::Dimensions deviceDims(hostData.getData().dimsData);
    sudoku::cuda::Grid deviceGrid(deviceDims, hostData.getData().gridData, 0);
    for (size_t i = 0; i < dims.getCellCount(); ++i) {
        BOOST_REQUIRE_EQUAL(deviceGrid.getCellValue(i), 0);
    }

    deviceGrid.setCellValue(3, 3);
    BOOST_REQUIRE_EQUAL(deviceGrid.getCellValue(3), 3);
}
