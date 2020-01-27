#include <memory>
#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/standard.h>

struct StandardDimensionsTestFixture
{
    sudoku::standard::Dimensions cpuDims;
    sudoku::cuda::DimensionParams dimParams;
    sudoku::cuda::KernelParams kernelParams;
    std::unique_ptr<sudoku::cuda::Dimensions> dims;

    StandardDimensionsTestFixture() : dimParams(cpuDims)
    {
        kernelParams.cellCount = dimParams.cellCount;
        kernelParams.maxCellValue = dimParams.maxCellValue;
        kernelParams.groupCount = dimParams.groupCount;
        kernelParams.groupValues = dimParams.groupValues.data();
        kernelParams.groupOffsets = dimParams.groupOffsets.data();
        kernelParams.groupsForCellValues = dimParams.groupsForCellValues.data();
        kernelParams.groupsForCellOffsets = dimParams.groupsForCellOffsets.data();
        dims = std::make_unique<sudoku::cuda::Dimensions>(kernelParams);
    }
};

BOOST_FIXTURE_TEST_SUITE(StandardDimensionsTestSuite, StandardDimensionsTestFixture)

    BOOST_AUTO_TEST_CASE(scalar_values)
    {
        BOOST_CHECK_EQUAL(dims->getCellCount(), 81);
        BOOST_CHECK_EQUAL(dims->getMaxCellValue(), 9);
        BOOST_CHECK_EQUAL(dims->getGroupCount(), 27);
    }

    BOOST_AUTO_TEST_CASE(list_values)
    {
        std::vector<size_t> expectedGroup3{ 27, 28, 29, 30, 31, 32, 33, 34, 35 };
        std::vector<size_t> expectedGroupsForCell3{ 0, 12, 19 };
        BOOST_REQUIRE_EQUAL_COLLECTIONS(
            expectedGroup3.begin(), expectedGroup3.end(),
            dims->getCellsInGroup(3), dims->getCellsInGroup(3) + dims->getCellsInGroupCount(3)
        );
        BOOST_REQUIRE_EQUAL_COLLECTIONS(
            expectedGroupsForCell3.begin(), expectedGroupsForCell3.end(),
            dims->getGroupsForCell(3), dims->getGroupsForCell(3) + dims->getGroupsForCellCount(3)
        );
    }

BOOST_AUTO_TEST_SUITE_END()