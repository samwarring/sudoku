#include <algorithm>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/solver.h>
#include <sudoku/fork.h>
#include <sudoku/square.h>

struct ThreadState
{
    sudoku::cuda::Dimensions dims;
    sudoku::cuda::Grid grid;
    sudoku::cuda::GuessStack guessStack;
    sudoku::cuda::Solver solver;

    ThreadState(sudoku::cuda::kernel::Data data, size_t threadNum)
        : dims(data.dimsData)
        , grid(dims, data.gridData, threadNum)
        , guessStack(data.guessStackData, threadNum)
        , solver(dims, grid, guessStack)
    {}
};

struct SolverFixture
{
    sudoku::square::Dimensions dims;
    std::vector<sudoku::Grid> grids;
    sudoku::cuda::kernel::HostData hostData;
    std::vector<ThreadState> threads;
    
    SolverFixture(size_t squareDimRoot, size_t threadCount)
        : dims(squareDimRoot)
        , grids(sudoku::fork(dims, threadCount))
        , hostData(dims, grids)
    {
        for (size_t threadNum = 0; threadNum < threadCount; ++threadNum) {
            threads.emplace_back(hostData.getData(), threadNum);
        }
    }
};

BOOST_AUTO_TEST_CASE(Solver_4x4_1thread_findsAnySolution)
{
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::Grid> grids(1, dims);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    sudoku::cuda::Dimensions cudaDims(hostData.getData().dimsData);
    sudoku::cuda::Grid cudaGrid(cudaDims, hostData.getData().gridData, 0);
    sudoku::cuda::GuessStack cudaGuessStack(hostData.getData().guessStackData, 0);
    sudoku::cuda::Solver solver(cudaDims, cudaGrid, cudaGuessStack);
    // Should find solution.
    BOOST_REQUIRE_EQUAL(solver.computeNextSolution(100000), sudoku::cuda::Result::OK_FOUND_SOLUTION);
    // Solution should not contain any empty cells
    auto solution = hostData.getCellValues(0);
    BOOST_REQUIRE_EQUAL(std::count(solution.cbegin(), solution.cend(), 0), 0);
}

BOOST_AUTO_TEST_CASE(Solver_9x9_fork_4threads_allFindSolution)
{
    size_t threadCount = 4;
    sudoku::square::Dimensions dims(3);
    auto grids = sudoku::fork(dims, threadCount);
    sudoku::cuda::kernel::HostData hostData(dims, grids);

    for (size_t threadNum = 0; threadNum < threadCount; ++threadNum) {
        sudoku::cuda::Dimensions dims(hostData.getData().dimsData);
        sudoku::cuda::Grid grid(dims, hostData.getData().gridData, threadNum);
        sudoku::cuda::GuessStack guessStack(hostData.getData().guessStackData, threadNum);
        sudoku::cuda::Solver solver(dims, grid, guessStack);
        sudoku::cuda::Result result = solver.computeNextSolution(1000);
        BOOST_REQUIRE_EQUAL(result, sudoku::cuda::Result::OK_FOUND_SOLUTION);
        auto solution = hostData.getCellValues(threadNum);
        BOOST_REQUIRE_EQUAL(std::count(solution.cbegin(), solution.cend(), 0), 0);
    }
}

BOOST_AUTO_TEST_CASE(Solver_9x9_1thread_timeoutUntilSolved)
{
    SolverFixture fixture(3, 1);
    // Make 10 guesses before timing out. Continue guessing until
    // timeouts stop. (WARNING: A bug could cause this to loop forever)
    sudoku::cuda::Result result;
    int timeoutCount = -1;
    do {
        timeoutCount++;
        result = fixture.threads[0].solver.computeNextSolution(10);
    } while (result == sudoku::cuda::Result::OK_TIMED_OUT);

    // After the loop, sudoku should be solved.
    BOOST_REQUIRE_EQUAL(result, sudoku::cuda::Result::OK_FOUND_SOLUTION);

    // If we never timed out, then we aren't testing the timeout logic!
    BOOST_REQUIRE_GT(timeoutCount, 0);
}
