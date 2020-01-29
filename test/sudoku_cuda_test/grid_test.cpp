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
}

BOOST_AUTO_TEST_CASE(Grid_setAndGetValue_4threads)
{
    sudoku::square::Dimensions dims(2);
    sudoku::cuda::kernel::HostData hostData(dims, {4, dims});
    sudoku::cuda::Dimensions deviceDims(hostData.getData().dimsData);
    std::vector<sudoku::cuda::Grid> deviceGrids;
    for (size_t i = 0; i < 4; ++i) {
        deviceGrids.emplace_back(deviceDims, hostData.getData().gridData, i);
        deviceGrids[i].setCellValue(i, i);
        BOOST_REQUIRE_EQUAL(deviceGrids[i].getCellValue(i), i);
        deviceGrids[i].clearCellValue(i);
        BOOST_REQUIRE_EQUAL(deviceGrids[i].getCellValue(i), 0);
    }
}

BOOST_AUTO_TEST_CASE(Grid_getNextAvailableValue)
{
    sudoku::square::Dimensions dims(2);
    sudoku::cuda::kernel::HostData hostData(dims, {dims});
    sudoku::cuda::Grid grid(hostData.getData().dimsData, hostData.getData().gridData, 0);
    
    // 0 0 | 0 0 ...
    BOOST_REQUIRE_EQUAL(grid.getNextAvailableValue(0, 0), 1);

    // 0 1 | 0 0 ...
    grid.setCellValue(1, 1);
    BOOST_REQUIRE_EQUAL(grid.getNextAvailableValue(0, 0), 2);

    // 0 1 | 0 0
    // 2 0 | 0 0 ...
    grid.setCellValue(4, 2);
    BOOST_REQUIRE_EQUAL(grid.getNextAvailableValue(0, 0), 3);

    // 0 1 | 0 0
    // 2 3 | 0 0 ...
    grid.setCellValue(5, 3);
    BOOST_REQUIRE_EQUAL(grid.getNextAvailableValue(0, 0), 4);

    // 0 1 | 4 0
    // 2 3 | 0 0 ...
    grid.setCellValue(2, 4);
    BOOST_REQUIRE_EQUAL(grid.getNextAvailableValue(0, 0), 0);
}

BOOST_AUTO_TEST_CASE(Grid_getNextAvailableValue_withInitialValues)
{
    // 0 0 | 0 1
    // 2 0 | 0 0 ...
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::Grid> grids(1, dims);
    grids[0].setCellValue(3, 1);
    grids[0].setCellValue(4, 2);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    sudoku::cuda::Grid grid(hostData.getData().dimsData, hostData.getData().gridData, 0);
    BOOST_REQUIRE_EQUAL(grid.getNextAvailableValue(0, 0), 3);
}

BOOST_AUTO_TEST_CASE(Grid_getMaxBlockEmptyCell_withInitialValues)
{
    // 0 1 | 0 2  <- pos=2 has greatest block count
    // 0 0 | 3 0
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::Grid> grids(1, dims);
    grids[0].setCellValue(1, 1);
    grids[0].setCellValue(3, 2);
    grids[0].setCellValue(6, 3);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    sudoku::cuda::Grid grid(hostData.getData().dimsData, hostData.getData().gridData, 0);
    BOOST_REQUIRE_EQUAL(grid.getMaxBlockEmptyCell(), 2);
}

BOOST_AUTO_TEST_CASE(Grid_getMaxBlockCount_withInitialRestrictions)
{
    // 0 0 | 0 0
    // 0 0 | x 0  <- restrict pos=6 from values 1 and 2
    // ----+----
    // 0 0 | 0 0
    // 0 0 | 0 0
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::Grid> grids(1, dims);
    grids[0].restrictCellValue(6, 1);
    grids[0].restrictCellValue(6, 2);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    sudoku::cuda::Grid grid(hostData.getData().dimsData, hostData.getData().gridData, 0);
    BOOST_REQUIRE_EQUAL(grid.getMaxBlockEmptyCell(), 6);
    BOOST_REQUIRE_EQUAL(grid.getNextAvailableValue(6, 0), 3);
}
