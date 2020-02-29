#include <sudoku/groupwise_solver.h>

namespace sudoku
{
    GroupwiseSolver::GroupwiseSolver(const Dimensions& dims, std::vector<CellValue> cellValues)
        : dims_(&dims)
        , guessStack_(dims.getCellCount())
        , guessStackPos_(0)
        , blockCounter_(dims)
    {
        if (cellValues.size() == 0) {
            cellValues_.resize(dims.getCellCount());
        }
        else {
            cellValues_ = std::move(cellValues);
        }
        dims.validateCellValues(cellValues_);
        for (CellCount cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
            auto cellValue = cellValues_[cellPos];
            if (cellValue) {
                blockCounter_.setCellValue(cellPos, cellValue);
            }
        }
    }

    bool GroupwiseSolver::computeNextSolution()
    {
        Timer timer(metrics_.duration);
        CellCount cellPos = getNextEmptyCell(0);
        CellValue minCellValue = 0;

        if (cellPos >= dims_->getCellCount()) {
            // Already solved. Pop last guess before searching.
            cellPos = guessStack_[--guessStackPos_];
            minCellValue = cellValues_[cellPos];
            blockCounter_.clearCellValue(cellPos, minCellValue);
            cellValues_[cellPos] = 0;
            metrics_.totalBacktracks++;
        }

        while (cellPos < dims_->getCellCount()) {
            auto cellValue = blockCounter_.getNextAvailableValue(cellPos, minCellValue);
            if (cellValue) {
                // Push the guess
                cellValues_[cellPos] = cellValue;
                blockCounter_.setCellValue(cellPos, cellValue);
                metrics_.totalGuesses++;
                guessStack_[guessStackPos_++] = cellPos;

                // Prepare for next guess
                cellPos = getNextEmptyCell(cellPos);
                minCellValue = 0;
            }
            else {
                if (guessStackPos_ == 0) {
                    // No more guesses to backtrack. No solution.
                    return false;
                }

                // Pop the last guess
                cellPos = guessStack_[--guessStackPos_];
                metrics_.totalBacktracks++;
                blockCounter_.clearCellValue(cellPos, cellValues_[cellPos]);
                minCellValue = cellValues_[cellPos];
                cellValues_[cellPos] = 0;
            }
        }

        return true;
    }

    CellCount GroupwiseSolver::getNextEmptyCell(CellCount cellPos) const
    {
        while (cellPos < dims_->getCellCount() && cellValues_[cellPos]) {
            cellPos++;
        }
        return cellPos;
    }
}
