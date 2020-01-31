#include <algorithm>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/solver.h>
#include <sudoku/square.h>

BOOST_AUTO_TEST_CASE(Solver)
{
    sudoku::square::Dimensions dims(2);
    std::vector<sudoku::Grid> grids(1, dims);
    sudoku::cuda::kernel::HostData hostData(dims, grids);
    sudoku::cuda::Dimensions cudaDims(hostData.getData().dimsData);
    sudoku::cuda::Grid cudaGrid(cudaDims, hostData.getData().gridData, 0);
    sudoku::cuda::GuessStack cudaGuessStack(hostData.getData().guessStackData, 0);
    sudoku::cuda::Solver solver(cudaDims, cudaGrid, cudaGuessStack);
    // Should find solution.
    BOOST_REQUIRE(solver.computeNextSolution(100000));
    // Solution should not contain any empty cells
    auto solution = hostData.getCellValues(0);
    BOOST_REQUIRE_EQUAL(std::count(solution.cbegin(), solution.cend(), 0), 0);
}
