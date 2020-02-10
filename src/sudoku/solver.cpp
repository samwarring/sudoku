#include <sudoku/solver.h>

namespace sudoku
{
    Solver::Solver(Grid grid)
        : grid_(std::move(grid))
        , haltEvent_(false)
    {
        // Check if already solved. If so, the first call to computeNextSolution
        // should return true.
        unreportedSolution_ = grid_.isFull();
    }

    bool Solver::computeNextSolution()
    {
        auto startTime = Metrics::now();
        bool result = sequentialSolve();
        auto stopTime = Metrics::now();
        metrics_.duration += (stopTime - startTime);
        return result;
    }

    void Solver::pushGuess(CellCount cellPos, CellValue cellValue)
    {
        grid_.setCellValue(cellPos, cellValue);
        guesses_.push(cellPos);
        metrics_.totalGuesses++;
    }

    std::pair<CellCount, CellValue> Solver::popGuess()
    {
        auto cellPos = guesses_.top();
        guesses_.pop();
        auto cellValue = grid_.getCellValue(cellPos);
        grid_.clearCellValue(cellPos);
        metrics_.totalBacktracks++;
        return {cellPos, cellValue};
    }

    bool Solver::sequentialSolve()
    {
        if (unreportedSolution_) {
            // We previously found a solution without sequentialSolve().
            // Future calls to sequentialSolve() will continue searching,
            // but not this one.
            unreportedSolution_ = false;
            return true;
        }

        const Dimensions& dims = grid_.getDimensions();
        auto cellPos = grid_.getMaxBlockEmptyCell();
        CellValue minCellValue = 0;

        // If already solved, pop the last guess.
        if (cellPos == dims.getCellCount()) {
            if (guesses_.size() == 0) {
                // Nothing to pop. No more solutions.
                return false;
            }
            auto prevGuess = popGuess();
            cellPos = prevGuess.first;
            minCellValue = prevGuess.second;
        }

        while (cellPos != dims.getCellCount()) {
            
            // Check if the solver should stop searching.
            // TODO: If performance becomes an issue, a couple ideas:
            //       1. examine other memory orders (e.g. std::memory_order_relaxed)
            //       2. don't check the event for _every_ iteration of the loop.
            if (haltEvent_.load()) {
                return false;
            }

            // Does this cell have any remaining potential values?
            auto cellValue = grid_.getCellPotential(cellPos).getNextAvailableValue(minCellValue);
            if (cellValue == 0) {
                // Backtrack
                if (guesses_.size() == 0) {
                    // Can't backtrack any further. No solution.
                    return false;
                }
                auto prevGuess = popGuess();
                cellPos = prevGuess.first;
                minCellValue = prevGuess.second;
                continue;
            }

            // We have an available value. Try it, and continue guessing.
            pushGuess(cellPos, cellValue);
            cellPos = grid_.getMaxBlockEmptyCell();
            minCellValue = 0;
        }
        return true;
    }
}
