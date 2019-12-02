#include <boost/test/unit_test.hpp>
#include <sudoku/potential.h>

BOOST_AUTO_TEST_CASE(Potential_getMaxCellValue)
{
    sudoku::Potential p(9);
    BOOST_REQUIRE_EQUAL(p.getMaxCellValue(), 9);
}

BOOST_AUTO_TEST_CASE(Potential_1_blockAndUnblock)
{
    sudoku::Potential p(1);
    BOOST_REQUIRE(!p.isBlocked(1));
    BOOST_REQUIRE_EQUAL(p.getAmountBlocked(), 0);
    BOOST_REQUIRE_EQUAL(p.getNextAvailableValue(0), 1);
    
    p.block(1);

    BOOST_REQUIRE(p.isBlocked(1));
    BOOST_REQUIRE_EQUAL(p.getAmountBlocked(), 1);
    BOOST_REQUIRE_EQUAL(p.getNextAvailableValue(0), 0);

    p.unblock(1);

    BOOST_REQUIRE(!p.isBlocked(1));
    BOOST_REQUIRE_EQUAL(p.getAmountBlocked(), 0);
    BOOST_REQUIRE_EQUAL(p.getNextAvailableValue(0), 1);
}

BOOST_AUTO_TEST_CASE(Potential_3_getNextAvailableValue)
{
    sudoku::Potential p(3);  // [0 0 0]

    auto requireNextAvailableValue = [&p] (size_t v0, size_t v1, size_t v2, size_t v3, size_t amtBlocked) {
        BOOST_REQUIRE_EQUAL(p.getNextAvailableValue(0), v0);
        BOOST_REQUIRE_EQUAL(p.getNextAvailableValue(1), v1);
        BOOST_REQUIRE_EQUAL(p.getNextAvailableValue(2), v2);
        BOOST_REQUIRE_EQUAL(p.getNextAvailableValue(3), v3);
        BOOST_REQUIRE_EQUAL(p.getAmountBlocked(), amtBlocked);
    };

    requireNextAvailableValue(1, 2, 3, 0, 0);
    p.block(1);              // [1 0 0]
    requireNextAvailableValue(2, 2, 3, 0, 1);
    p.block(2);              // [1 1 0]
    requireNextAvailableValue(3, 3, 3, 0, 2);
    p.block(3);              // [1 1 1]
    requireNextAvailableValue(0, 0, 0, 0, 3);
    p.unblock(2);            // [1 0 1]
    requireNextAvailableValue(2, 2, 0, 0, 2);
    p.unblock(1);            // [0 0 1]
    requireNextAvailableValue(1, 2, 0, 0, 1);
    p.block(3);              // [0 0 2]
    requireNextAvailableValue(1, 2, 0, 0, 1);
    p.unblock(3);            // [0 0 1]
    requireNextAvailableValue(1, 2, 0, 0, 1);
}