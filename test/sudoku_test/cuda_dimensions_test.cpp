#include <boost/test/unit_test.hpp>
#include <sudoku/standard.h>
#include <sudoku/cuda/dimensions.h>

BOOST_AUTO_TEST_CASE(CudaDimensions_serializeAndDeserialize, * boost::unit_test::disabled())
{
    // Serialize the standard dimensions.
    sudoku::standard::Dimensions dims;
    auto serializedDims = sudoku::cuda::Dimensions::serialize(dims);
    BOOST_REQUIRE_GT(serializedDims.size(), 81);

    // De-serialize the buffer (typically from gpu, but testing on cpu).
    sudoku::cuda::Dimensions cudaDims(serializedDims.data(), serializedDims.size());
    BOOST_REQUIRE(cudaDims.isValid());
    BOOST_REQUIRE_EQUAL(cudaDims.getCellCount(), 81);
    BOOST_REQUIRE_EQUAL(cudaDims.getMaxCellValue(), 9);
    BOOST_REQUIRE_EQUAL(cudaDims.getGroupCount(), 27);
}
