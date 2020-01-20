#include <string>
#include <unordered_map>
#include <boost/test/unit_test.hpp>
#include <sudoku/parallel_solver.h>
#include <sudoku/standard.h>
#include <sudoku/cell_value_parser.h>
#include <sudoku/formatter.h>
#include "util.h"

BOOST_AUTO_TEST_CASE(ParallelSolver_example9x9_4threads)
{
    sudoku::standard::Dimensions dims;
    auto cellValues = sudoku::parseCellValues(
        dims,
        "020000000 000600003 074080000"
        "000003002 080040010 600500000"
        "000010780 500009000 000000040"
    );
    auto expectedSolution = sudoku::parseCellValues(
        dims,
        "126437958 895621473 374985126"
        "457193862 983246517 612578394"
        "269314785 548769231 731852649"
    );

    sudoku::Grid grid(dims, cellValues);
    sudoku::ParallelSolver solver(grid, 4, 10);
    BOOST_REQUIRE(solver.computeNextSolution());
    const auto& actualSolution = solver.getCellValues();
    BOOST_REQUIRE_EQUAL_VECTORS(expectedSolution, actualSolution);
}

BOOST_AUTO_TEST_CASE(ParallelSolver_empty9x9_16Threads)
{
    sudoku::standard::Dimensions dims;
    std::vector<size_t> cellValues(dims.getCellCount(), 0);
    sudoku::Grid grid(dims, cellValues);
    sudoku::ParallelSolver solver(grid, 16, 8);

    // Compute 100 solutions. Each solution should be unique; use a
    // hash map to be sure.
    const size_t solutionCount = 100;
    std::string formatString(dims.getCellCount(), '0');
    sudoku::Formatter fmt(dims, formatString);
    std::unordered_map<std::string, size_t> solutions;
    for (size_t solutionNum = 0; solutionNum < solutionCount; ++solutionNum) {
        BOOST_REQUIRE(solver.computeNextSolution());
        solutions[fmt.format(solver.getCellValues())]++;
    }
    
    // Check that all solutions are unique.
    for (auto it = solutions.begin(); it != solutions.end(); ++it) {
        size_t solutionOcurrences = it->second;
        BOOST_CHECK(solutionOcurrences == 1);
    }
}

BOOST_AUTO_TEST_CASE(ParallelSolver_25x25_2threads)
{
    // I encountered this situation by coincidence. One thread quickly
    // finds a solution, but the others take forever. If the user only
    // wants a single solution, the ParallelSolver destructor hangs
    // because the worker threads are still working and haven't "checked in"
    // with the condition variable.

    sudoku::square::Dimensions dims(5);
    std::vector<size_t> cellValues(dims.getCellCount(), 0);
    sudoku::Grid grid(dims, std::move(cellValues));
    sudoku::ParallelSolver solver(std::move(grid), 2, 1);
    BOOST_REQUIRE(solver.computeNextSolution());
}

BOOST_AUTO_TEST_CASE(ParalellSolver_9x9_2threads_noSolution)
{
    sudoku::standard::Dimensions dims;
    auto cellValues = sudoku::parseCellValues(dims, 
        "1 2 3 4 5 6 7 8 0 " // note last cell in row is blocked for all values.
        "0 0 0 0 0 0 0 0 9 "
    );
    sudoku::Grid grid(dims, std::move(cellValues));
    sudoku::ParallelSolver solver(std::move(grid), 2, 1);
    BOOST_REQUIRE(!solver.computeNextSolution());
}

BOOST_AUTO_TEST_CASE(ParallelSolver_findsAllSolutions)
{
    // Using the single-threaded solver, I found 288 total solutions with
    // a 4x4 square sudoku. This test verifies that none of the solutions
    // are lost when using a parallel solver.
    sudoku::square::Dimensions dims(2);
    std::vector<size_t> cellValues(dims.getCellCount(), 0);
    sudoku::Grid grid(dims, cellValues);
    sudoku::ParallelSolver solver(grid, 4, 10);
    auto solutionCount = 0;
    while (solver.computeNextSolution()) {
        solutionCount++;
    }
    BOOST_REQUIRE_EQUAL(solutionCount, 288);
}
