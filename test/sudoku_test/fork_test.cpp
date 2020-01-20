#include <string>
#include <unordered_map>
#include <vector>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>
#include <sudoku/dimensions.h>
#include <sudoku/fork.h>
#include <sudoku/formatter.h>
#include <sudoku/grid.h>
#include <sudoku/solver.h>
#include <sudoku/square.h>
#include "util.h"

// Each test case represents the number of peers to request.
std::vector<size_t> testCases_fork9x9{ 1, 2, 3, 6, 8, 16 };

BOOST_DATA_TEST_CASE(fork9x9, testCases_fork9x9)
{
    // Initialize a solver and fork it!
    const size_t numPeers = sample;
    sudoku::square::Dimensions dims(3);
    std::vector<size_t> cellValues(dims.getCellCount(), 0);
    sudoku::Grid grid(dims, std::move(cellValues));
    auto peers = sudoku::fork(std::move(grid), numPeers);
    
    // Test that we obtained the requested number of peers.
    // For some sudokus, this may not be possible, but an empty
    // 9x9 sudoku should not have a problem.
    BOOST_REQUIRE_EQUAL(peers.size(), numPeers);

    // Use a hashmap of (solution) -> (# occurances) to determine
    // if all solutions are unique.
    std::unordered_map<std::string, size_t> solutions;
    std::string formatString(dims.getCellCount(), '0');
    sudoku::Formatter fmt(dims, formatString);

    // Each peer should all be solvable.
    for (auto& peer : peers) {
        sudoku::Solver solver(std::move(peer));
        BOOST_REQUIRE(solver.computeNextSolution());
        solutions[fmt.format(solver.getCellValues())]++;
    }

    // Each peer's solution should be unique from all the others.
    // Verify number of solution occurances is 1 for all solutions.
    for (auto it = solutions.begin(); it != solutions.end(); ++it) {
        size_t solutionOcurrances = it->second;
        BOOST_CHECK(solutionOcurrances == 1);
    }
}


BOOST_AUTO_TEST_CASE(Solver_oneCellRemaining_fork)
{
    sudoku::square::Dimensions dims(2);
    std::vector<size_t> cellValues{
        1, 2, 3, 4,
        3, 4, 1, 2,
        2, 1, 4, 3,
        4, 3, 2, 0 // last remaining cell. Should be set to 1.
    };
    sudoku::Grid grid(dims, cellValues);
    auto peers = sudoku::fork(grid, 3);

    // Even though we requested 3 peers, there was only one
    // remaining value. This means we expect 1 peer.
    BOOST_REQUIRE_EQUAL(peers.size(), 1);

    // We want computeNextSolution() to produce the solution
    // obtained during fork() instead of popping the last
    // guess and continuing (which would yield no more
    // solutions in this case).
    sudoku::Solver solver(peers[0]);
    BOOST_REQUIRE(solver.computeNextSolution());
    std::vector<size_t> expectedSolution = {
        1, 2, 3, 4,
        3, 4, 1, 2,
        2, 1, 4, 3,
        4, 3, 2, 1
    };
    BOOST_REQUIRE_EQUAL_VECTORS(expectedSolution, solver.getCellValues());

    // The next attempt to compute a solution should find no
    // more solutions.
    BOOST_REQUIRE(!solver.computeNextSolution());
}

BOOST_AUTO_TEST_CASE(Solver_fork_peerInitializedWithSolution)
{
    // In this example, the last remaining cell has multiple
    // available values (since there are no groups). This will
    // cause fork() to produce 4 peers, where each is initialized
    // with a complete solution.
    sudoku::Dimensions dims(4, 4, {});
    std::vector<size_t> cellValues{1, 2, 3, 0};
    sudoku::Grid grid(dims, cellValues);
    //sudoku::Solver solver(grid);
    auto peers = sudoku::fork(grid, 4);
    BOOST_REQUIRE_EQUAL(peers.size(), 4);
    for (const auto& peer : peers) {
        sudoku::Solver solver(peer);
        BOOST_CHECK(solver.computeNextSolution());
    }
}