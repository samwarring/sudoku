#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <sudoku/cuda/solver.h>
#include <sudoku/square.h>

BOOST_AUTO_TEST_CASE(Solver_4x4_empty)
{
    sudoku::square::Dimensions dims(2);
    sudoku::cuda::Solver solver(dims);
    BOOST_REQUIRE(solver.computeNextSolution());
    auto cellValues = solver.getCellValues();
    BOOST_REQUIRE_EQUAL(std::count(cellValues.begin(), cellValues.end(), 0), 0);
}

BOOST_AUTO_TEST_CASE(Solver_9x9_empty)
{
    sudoku::square::Dimensions dims(3);
    sudoku::cuda::Solver solver(dims);
    BOOST_REQUIRE(solver.computeNextSolution());
    auto cellValues = solver.getCellValues();
    BOOST_REQUIRE_EQUAL(std::count(cellValues.begin(), cellValues.end(), 0), 0);
}