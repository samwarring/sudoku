#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <sudoku/cuda/solver.h>
#include <sudoku/square.h>

size_t roots[] = { 2, 3, 4 };
BOOST_DATA_TEST_CASE(Solver_square_empty, roots)
{
    sudoku::square::Dimensions dims(sample);
    sudoku::cuda::Solver solver(dims);
    BOOST_REQUIRE(solver.computeNextSolution());
    auto cellValues = solver.getCellValues();
    BOOST_REQUIRE_EQUAL(std::count(cellValues.begin(), cellValues.end(), 0), 0);
    sudoku::Grid verifyGrid(dims, sudoku::cuda::Solver::castCellValues(cellValues));
}
