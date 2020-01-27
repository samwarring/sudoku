#include <memory>
#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/standard.h>
#include "util.h"

BOOST_AUTO_TEST_CASE(Dimensions_standard)
{
    sudoku::standard::Dimensions cpuDims;
    sudoku::cuda::DimensionParams dimParams{ cpuDims };
    sudoku::cuda::KernelParams kernelParams{ makeHostParams(dimParams) };
    sudoku::cuda::Dimensions dims{ kernelParams };
    
    BOOST_REQUIRE_EQUAL(dims.getCellCount(), 81);
    BOOST_REQUIRE_EQUAL(dims.getMaxCellValue(), 9);
    BOOST_REQUIRE_EQUAL(dims.getGroupCount(), 27);

    std::vector<size_t> expectedGroup3{ 27, 28, 29, 30, 31, 32, 33, 34, 35 };
    std::vector<size_t> expectedGroupsForCell3{ 0, 12, 19 };
    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        expectedGroup3.begin(), expectedGroup3.end(),
        dims.getCellsInGroup(3), dims.getCellsInGroup(3) + dims.getCellsInGroupCount(3)
    );
    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        expectedGroupsForCell3.begin(), expectedGroupsForCell3.end(),
        dims.getGroupsForCell(3), dims.getGroupsForCell(3) + dims.getGroupsForCellCount(3)
    );
}
