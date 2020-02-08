#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <sudoku/cell_value_parser.h>
#include <sudoku/cuda/solver.h>
#include <sudoku/square.h>

size_t roots[] = { 2, 3, 4, 5 };
BOOST_DATA_TEST_CASE(Solver_square_empty, roots)
{
    sudoku::square::Dimensions dims(sample);
    sudoku::cuda::Solver solver(dims);
    BOOST_REQUIRE(solver.computeNextSolution());
    auto cellValues = solver.getCellValues();
    BOOST_REQUIRE_EQUAL(std::count(cellValues.begin(), cellValues.end(), 0), 0);
    sudoku::Grid verifyGrid(dims, sudoku::cuda::Solver::castCellValues(cellValues));
}

BOOST_AUTO_TEST_CASE(Solver_multipleKernelInvocations_sameState)
{
    constexpr unsigned BATCH_SIZE = 40;
    constexpr unsigned BATCH_COUNT = 2;

    sudoku::square::Dimensions dims(5);
    sudoku::cuda::Solver solver1(dims);
    auto values1 = solver1.getCellValues();
    for (auto i = 1; i <= BATCH_COUNT; ++i) {
        solver1.computeNextSolution(BATCH_SIZE);
        
        // Count the occupied cells after this batch. We should have filled
        // BATCH_SIZE cells.
        values1 = solver1.getCellValues();
        BOOST_REQUIRE_EQUAL(std::count(values1.begin(), values1.end(), 0),
                            dims.getCellCount() - (BATCH_SIZE * i));
    }

    sudoku::cuda::Solver solver2(dims);
    solver2.computeNextSolution(BATCH_COUNT * BATCH_SIZE);
    auto values2 = solver2.getCellValues();
    BOOST_REQUIRE_EQUAL(std::count(values2.begin(), values2.end(), 0),
                        dims.getCellCount() - (BATCH_SIZE * BATCH_COUNT));

    BOOST_REQUIRE_EQUAL_COLLECTIONS(values1.begin(), values1.end(),
                                    values2.begin(), values2.end());
}

BOOST_AUTO_TEST_CASE(Solver_9x9_initialValues)
{
    sudoku::square::Dimensions dims(3);
    auto initialCellValues = sudoku::parseCellValues(dims, "9 5 0 2");
    sudoku::Grid grid(dims, initialCellValues);
    sudoku::cuda::Solver solver(grid);
    BOOST_REQUIRE(solver.computeNextSolution());
    auto cellValues = solver.getCellValues();
    BOOST_CHECK_EQUAL(cellValues[0], 9);
    BOOST_CHECK_EQUAL(cellValues[1], 5);
    BOOST_CHECK_EQUAL(cellValues[3], 2);
}
