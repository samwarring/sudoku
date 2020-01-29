#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/standard.h>

BOOST_AUTO_TEST_CASE(Dimensions_standard)
{
    sudoku::standard::Dimensions cpuDims;
    sudoku::cuda::Dimensions::HostData hostData(cpuDims);
    sudoku::cuda::Dimensions dims(hostData.getData());
    
    BOOST_REQUIRE_EQUAL(dims.getCellCount(), 81);
    BOOST_REQUIRE_EQUAL(dims.getMaxCellValue(), 9);
    BOOST_REQUIRE_EQUAL(dims.getGroupCount(), 27);

    std::vector<size_t> expectedGroup3{ 27, 28, 29, 30, 31, 32, 33, 34, 35 };
    std::vector<size_t> expectedGroupsForCell3{ 0, 12, 19 };
    std::vector<size_t> actualGroup3;
    std::vector<size_t> actualGroupsForCell3;
    for (size_t itemNum = 0; itemNum < dims.getCellsInGroupCount(3); ++itemNum) {
        actualGroup3.push_back(dims.getCellInGroup(3, itemNum));
    }
    for (size_t itemNum = 0; itemNum < dims.getGroupsForCellCount(3); ++itemNum) {
        actualGroupsForCell3.push_back(dims.getGroupForCell(3, itemNum));
    }

    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        expectedGroup3.begin(), expectedGroup3.end(),
        actualGroup3.begin(), actualGroup3.end()
    );
    BOOST_REQUIRE_EQUAL_COLLECTIONS(
        expectedGroupsForCell3.begin(), expectedGroupsForCell3.end(),
        actualGroupsForCell3.begin(), actualGroupsForCell3.end()
    );
}
