#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/guess_stack.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/square.h>

BOOST_AUTO_TEST_CASE(GuessStack)
{
    const size_t threadCount = 2;
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::Grid> grids(threadCount, dims);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    std::vector<sudoku::cuda::GuessStack> stacks;
    for (size_t threadNum = 0; threadNum < threadCount; ++threadNum) {
        stacks.emplace_back(hostData.getData().guessStackData, threadNum);
        BOOST_REQUIRE_EQUAL(stacks[threadNum].getSize(), 0);
    }

    // Fill thread 0's stack.
    for (size_t stackPos = 0; stackPos < dims.getCellCount(); ++stackPos) {
        stacks[0].push(stackPos);
        BOOST_REQUIRE_EQUAL(stacks[0].getSize(), stackPos + 1);
    }

    // Fill thread 1's stack.
    for (size_t stackPos = 0; stackPos < dims.getCellCount(); ++stackPos) {
        stacks[1].push(100 + stackPos);
        BOOST_REQUIRE_EQUAL(stacks[1].getSize(), stackPos + 1);
        
        // Size of thread 0's stack remains unaffected
        BOOST_REQUIRE_EQUAL(stacks[0].getSize(), dims.getCellCount());
    }

    // Pop values from thread 0
    // (Must use signed type for stackPos or else stackPos >= 0 is never false)
    for (int stackPos = (int)dims.getCellCount() - 1; stackPos >= 0; --stackPos) {
        BOOST_REQUIRE_EQUAL(stacks[0].pop(), stackPos);
        BOOST_REQUIRE_EQUAL(stacks[0].getSize(), stackPos);

        // Size of thread 1's stack remains unaffected
        BOOST_REQUIRE_EQUAL(stacks[1].getSize(), dims.getCellCount());
    }

    // Pop values from thread 1
    for (int stackPos = (int)dims.getCellCount() - 1; stackPos >= 0; --stackPos) {
        BOOST_REQUIRE_EQUAL(stacks[1].pop(), 100 + stackPos);
        BOOST_REQUIRE_EQUAL(stacks[1].getSize(), stackPos);
    }
}
