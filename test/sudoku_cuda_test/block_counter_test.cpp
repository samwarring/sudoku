#include <boost/test/unit_test.hpp>
#include "block_counter_kernels.h"

BOOST_AUTO_TEST_CASE(BlockCounter_16cells_4values_blockAndUnblock)
{
    BlockCounterKernels bc(16, 4);
    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(0), 0);
    
    bc.block(0, 1);
    bc.block(0, 2);
    bc.block(0, 2);
    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(0), 2);
    BOOST_REQUIRE_EQUAL(bc.getValueBlockCount(0, 1), 1);
    BOOST_REQUIRE_EQUAL(bc.getValueBlockCount(0, 2), 2);

    bc.unblock(0, 2);
    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(0), 2);

    bc.unblock(0, 2);
    bc.unblock(0, 1);
    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(0), 0);
}

BOOST_AUTO_TEST_CASE(BlockCounter_16cells_4values_markOccupiedAndFree)
{
    BlockCounterKernels bc(16, 4);
    bc.markOccupied(3);
    BOOST_REQUIRE_LT(bc.getCellBlockCount(3), 0);

    bc.block(3, 3);
    bc.block(3, 4);
    BOOST_REQUIRE_LT(bc.getCellBlockCount(3), 0);

    bc.markFree(3);
    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(3), 2);
}

BOOST_AUTO_TEST_CASE(BlockCounter_81cells)
{
    BlockCounterKernels bc(81, 9);
    bc.block(15, 1);
    bc.block(15, 2);
    bc.block(14, 2);
    bc.markOccupied(13);

    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(15), 2);
    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(14), 1);
    BOOST_REQUIRE_LT(bc.getCellBlockCount(13), 0);

    bc.markFree(13);
    bc.unblock(15, 2);

    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(13), 0);
    BOOST_REQUIRE_EQUAL(bc.getCellBlockCount(15), 1);
}

BOOST_AUTO_TEST_CASE(BlockCounter_16cells_getMaxCellBlockCountPair)
{
    BlockCounterKernels bc(16, 4);
    bc.block(13, 1);
    auto pair = bc.getMaxBlockCountPair();
    BOOST_REQUIRE_EQUAL(pair.cellPos, 13);
    BOOST_REQUIRE_EQUAL(pair.cellBlockCount, 1);
}

BOOST_AUTO_TEST_CASE(BlockCounter_81cells_getMaxCellBlockCountPair)
{
    BlockCounterKernels bc(81, 9);
    bc.block(42, 1);
    bc.block(42, 2);
    bc.block(42, 3); // cell block count at 3

    bc.block(33, 1);
    bc.block(33, 2);
    bc.block(33, 2); // cell block count remains at 2

    auto pair = bc.getMaxBlockCountPair();
    BOOST_REQUIRE_EQUAL(pair.cellPos, 42);
    BOOST_REQUIRE_EQUAL(pair.cellBlockCount, 3);
}
