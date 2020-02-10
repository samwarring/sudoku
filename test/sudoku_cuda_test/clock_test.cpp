#include <boost/test/unit_test.hpp>
#include "clock_kernels.h"

BOOST_AUTO_TEST_CASE(Clock)
{
    ClockKernels ck;
    BOOST_REQUIRE_EQUAL(ck.getTickCount(), 0);
    BOOST_REQUIRE_EQUAL(ck.getMs().count(), 0);

    ck.doWork(20000);
    auto tickCount1 = ck.getTickCount();
    auto ms1 = ck.getMs();
    BOOST_REQUIRE_GT(tickCount1, 20000); // Assume each iteration >= 1 tick
    BOOST_REQUIRE_GT(ms1.count(), 0);

    ck.doWork(10000); // fewer iterations
    auto tickCount2 = ck.getTickCount();
    auto ms2 = ck.getMs();
    BOOST_REQUIRE_GT(tickCount2, tickCount1);
    BOOST_REQUIRE_GE(ms2.count(), ms1.count());
}
