#include <boost/test/unit_test.hpp>
#include <sudoku/groupwise_block_counter.h>
#include <sudoku/standard.h>

BOOST_AUTO_TEST_CASE(GroupwiseBlockCounter_setCellValue_getNextAvailableValue)
{
    sudoku::standard::Dimensions dims;
    sudoku::GroupwiseBlockCounter counter(dims);
    
    // 1 x 0 ... <-- 1 in same row
    // 0 0 2 ... <-- 2 in same subsquare
    // 0 0 0
    // 0 3 0 ... <-- 3 in same column
    // ...
    counter.setCellValue(0, 1);
    counter.setCellValue(11, 2);
    counter.setCellValue(28, 3);

    BOOST_REQUIRE_EQUAL(counter.getNextAvailableValue(1, 0), 4);
    BOOST_REQUIRE_EQUAL(counter.getNextAvailableValue(1, 1), 4);
    BOOST_REQUIRE_EQUAL(counter.getNextAvailableValue(1, 2), 4);
    BOOST_REQUIRE_EQUAL(counter.getNextAvailableValue(1, 3), 4);
    BOOST_REQUIRE_EQUAL(counter.getNextAvailableValue(1, 4), 5);
}

BOOST_AUTO_TEST_CASE(GroupwiseBlockCounter_getNextAvailableValue_noMoreValues)
{
    sudoku::square::Dimensions dims(2);
    sudoku::GroupwiseBlockCounter counter(dims);
    // 1 2 3 x <-- x has one remaining available value = 4
    // 0 0 0 0
    // 0 0 0 0
    // 0 0 0 0 
    counter.setCellValue(0, 1);
    counter.setCellValue(1, 2);
    counter.setCellValue(2, 3);
    BOOST_REQUIRE_EQUAL(counter.getNextAvailableValue(3, 0), 4);

    // 1 2 3 x <-- x has no more available values
    // 0 0 0 4
    // 0 0 0 0
    // 0 0 0 0
    counter.setCellValue(7, 4);
    BOOST_REQUIRE_EQUAL(counter.getNextAvailableValue(3, 0), 0);
}
