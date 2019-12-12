#include <sudoku/solver.h>

namespace sudoku
{
    Solver::Solver(const Dimensions& dims, std::vector<size_t> cellValues)
        : dims_(dims)
        , cellValues_(std::move(cellValues))
    {
        if (cellValues_.size() != dims_.getCellCount()) {
            throw SolverException("Number of initial values does not match dimensions");
        }
        initializeCellPotentials();
        validateCellValues();
    }

    bool Solver::computeNextSolution()
    {
        return sequentialSolve();
    }

    const std::vector<size_t>& Solver::getCellValues() const
    {
        return cellValues_;
    }

    void Solver::initializeCellPotentials()
    {
        cellPotentials_.reserve(dims_.getCellCount());
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            cellPotentials_.emplace_back(dims_.getMaxCellValue());
        }

        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            if (cellValues_[cellPos] > 0) {
                for (size_t groupNum : dims_.getGroupsForCell(cellPos)) {
                    for (size_t relatedCellPos : dims_.getCellsInGroup(groupNum)) {
                        cellPotentials_[relatedCellPos].block(cellValues_[cellPos]);
                    }
                }
            }
        }
    }

    void Solver::validateCellValues() const
    {
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            if (dims_.getMaxCellValue() < cellValues_[cellPos]) {
                throw SolverException("A cell value is out of range");
            }
        }

        for (size_t groupNum = 0; groupNum < dims_.getNumGroups(); ++groupNum) {
            std::vector<int> occurs(dims_.getMaxCellValue());
            for (size_t cellPos : dims_.getCellsInGroup(groupNum)) {
                size_t cellValue = cellValues_[cellPos];
                if (cellValue != 0) {
                    if (0 < occurs[cellValue - 1]) {
                        throw SolverException("A group contained a repeated value");
                    }
                    occurs[cellValue - 1]++;
                }
            }
        }
    }

    void Solver::pushGuess(size_t cellPos, size_t cellValue)
    {
        cellValues_[cellPos] = cellValue;
        for (size_t groupNum : dims_.getGroupsForCell(cellPos)) {
            for (size_t relatedCellPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedCellPos].block(cellValue);
            }
        }
        guesses_.push({ cellPos, cellValue });
        totalGuesses_++;
    }

    std::pair<size_t, size_t> Solver::popGuess()
    {
        auto prevGuess = guesses_.top();
        guesses_.pop();
        size_t cellPos = prevGuess.first;
        size_t cellValue = prevGuess.second;
        cellValues_[cellPos] = 0;
        for (size_t groupNum : dims_.getGroupsForCell(cellPos)) {
            for (size_t relatedCellPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedCellPos].unblock(cellValue);
            }
        }
        return prevGuess;
    }

    size_t Solver::selectNextCell() const
    {
        size_t leadingPos = dims_.getCellCount();
        size_t leadingNumBlocked = 0;
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            if (cellValues_[cellPos] == 0) {
                size_t amountBlocked = cellPotentials_[cellPos].getAmountBlocked();
                if (leadingNumBlocked <= amountBlocked) {
                    leadingPos = cellPos;
                    leadingNumBlocked = amountBlocked;
                }
            }
        }
        return leadingPos;
    }

    bool Solver::sequentialSolve()
    {
        size_t cellPos = selectNextCell();
        size_t minCellValue = 0;

        // If already solved, pop the last guess.
        if (cellPos == dims_.getCellCount()) {
            if (guesses_.size() == 0) {
                // Nothing to pop. No solution.
                return false;
            }
            auto prevGuess = popGuess();
            cellPos = prevGuess.first;
            minCellValue = prevGuess.second;
        }

        while (cellPos != dims_.getCellCount()) {

            // Does this cell have any remaining potential values?
            size_t cellValue = cellPotentials_[cellPos].getNextAvailableValue(minCellValue);
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
            cellPos = selectNextCell();
            minCellValue = 0;
        }
        return true;
    }
}