#include <boost/test/unit_test.hpp>
#include "guess_stack_kernels.h"

BOOST_AUTO_TEST_CASE(GuessStack_max16_pushAndPop)
{
    GuessStackKernels gs(16);
    
    BOOST_REQUIRE_EQUAL(gs.getSize(), 0);

    gs.push(1);
    gs.push(2);
    gs.push(3);

    BOOST_REQUIRE_EQUAL(gs.getSize(), 3);
    BOOST_REQUIRE_EQUAL(gs.pop(), 3);
    BOOST_REQUIRE_EQUAL(gs.getSize(), 2);
}

BOOST_AUTO_TEST_CASE(GuessStack_max81_pushAndPop)
{
    GuessStackKernels gs(81);

    gs.push(9);
    gs.push(18);
    gs.push(27);
    gs.push(42);

    BOOST_REQUIRE_EQUAL(gs.getSize(), 4);
    BOOST_REQUIRE_EQUAL(gs.pop(), 42);
    BOOST_REQUIRE_EQUAL(gs.getSize(), 3);
    
    gs.push(33);

    BOOST_REQUIRE_EQUAL(gs.getSize(), 4);
    BOOST_REQUIRE_EQUAL(gs.pop(), 33);
    BOOST_REQUIRE_EQUAL(gs.pop(), 27);
    BOOST_REQUIRE_EQUAL(gs.pop(), 18);
    BOOST_REQUIRE_EQUAL(gs.pop(), 9);
    BOOST_REQUIRE_EQUAL(gs.getSize(), 0);
}

BOOST_AUTO_TEST_CASE(GuessStack_max16_fillAndClear)
{
    GuessStackKernels gs(16);
    for (sudoku::cuda::CellCount cellPos = 0; cellPos < 16; ++cellPos) {
        gs.push(cellPos);
        BOOST_REQUIRE_EQUAL(gs.getSize(), cellPos + 1);
    }
    for (sudoku::cuda::CellCount popCount = 0; popCount < 16; ++popCount) {
        BOOST_REQUIRE_EQUAL(gs.pop(), 15 - popCount);
        BOOST_REQUIRE_EQUAL(gs.getSize(), 15 - popCount);
    }
}
