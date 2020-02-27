#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <sudoku/dimensions.h>
#include <sudoku/groupwise_empty_solver.h>
#include <sudoku/parallel_solver.h>
#include <sudoku/solver.h>
#include <sudoku/solver_interface.h>
#include <sudoku/square.h>
#include <sudoku/cell_value_parser.h>

enum class SolverType
{
    SEQUENTIAL,
    PARALLEL,
    GROUPWISE_EMPTY
};

namespace std
{
    ostream& operator<<(ostream& out, const SolverType& solverType)
    {
        switch (solverType) {
            case SolverType::SEQUENTIAL: return out << "sudoku::Solver";
            case SolverType::PARALLEL: return out << "sudoku::ParallelSolver";
            case SolverType::GROUPWISE_EMPTY: return out << "sudoku::GroupwiseEmptySolver";
            default: return out << "<Unknown Solver>";
        }
    }
}

std::unique_ptr<sudoku::SolverInterface> solverFactory(const sudoku::Dimensions& dims, SolverType solverType)
{
    switch (solverType) {
        case SolverType::SEQUENTIAL:
            return std::make_unique<sudoku::Solver>(dims);
        case SolverType::PARALLEL:
            return std::make_unique<sudoku::ParallelSolver>(dims, 4, 8);
        case SolverType::GROUPWISE_EMPTY:
            return std::make_unique<sudoku::GroupwiseEmptySolver>(dims);
        default:
            throw std::runtime_error("unrecognised SolverType");
    }
}

std::unique_ptr<sudoku::SolverInterface> solverFactory(const sudoku::Dimensions& dims,
                                                       std::vector<sudoku::CellValue> initialValues,
                                                       SolverType solverType)
{
    sudoku::Grid grid(dims, initialValues);
    switch (solverType) {
        case SolverType::SEQUENTIAL:
            return std::make_unique<sudoku::Solver>(grid);
        case SolverType::PARALLEL:
            return std::make_unique<sudoku::ParallelSolver>(grid, 4, 8);
        default:
            throw std::runtime_error("unrecognised SolverType");
    }
}

SolverType emptySolverTypes[] = {
    SolverType::SEQUENTIAL,
    SolverType::PARALLEL,
    SolverType::GROUPWISE_EMPTY
};

BOOST_DATA_TEST_CASE(SolverInterface_empty9x9, emptySolverTypes)
{
    sudoku::square::Dimensions dims(3);
    auto solver = solverFactory(dims, sample);
    BOOST_REQUIRE(solver->computeNextSolution());
    auto cellValues = solver->getCellValues();
    BOOST_REQUIRE_EQUAL(0, std::count(cellValues.begin(), cellValues.end(), 0));

    auto metrics = solver->getMetrics();
    BOOST_REQUIRE_GE(metrics.totalGuesses, dims.getCellCount());
    BOOST_REQUIRE_EQUAL(metrics.totalGuesses - metrics.totalBacktracks, dims.getCellCount());
    BOOST_REQUIRE_GT(std::chrono::duration_cast<std::chrono::nanoseconds>(metrics.duration).count(), 0);

    // Empty sudokus have more than 1 solution.
    BOOST_REQUIRE(solver->computeNextSolution());
    auto cellValues2 = solver->getCellValues();
    BOOST_REQUIRE_EQUAL(0, std::count(cellValues2.begin(), cellValues2.end(), 0));
    BOOST_REQUIRE(cellValues != cellValues2);
}

SolverType solverTypes[] = {
    SolverType::SEQUENTIAL,
    SolverType::PARALLEL
};

BOOST_DATA_TEST_CASE(SolverInterface_initialValues, solverTypes)
{
    sudoku::square::Dimensions dims(3);
    auto inputValueString =
        "000600400 700003600 000091080"
        "000000000 050180003 000306045"
        "040200060 903000000 020000100";
    auto outputValueString = 
        "581672439 792843651 364591782"
        "438957216 256184973 179326845"
        "845219367 913768524 627435198";
    auto inputValues = sudoku::parseCellValues(dims, inputValueString);
    auto expectedSolution = sudoku::parseCellValues(dims, outputValueString);
    auto solver = solverFactory(dims, inputValues, sample);
    BOOST_REQUIRE(solver->computeNextSolution());
    auto foundSolution = solver->getCellValues();
    BOOST_REQUIRE_EQUAL_COLLECTIONS(expectedSolution.begin(), expectedSolution.end(),
                                    foundSolution.begin(), foundSolution.end());
}
