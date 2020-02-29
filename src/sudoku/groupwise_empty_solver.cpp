#include <sudoku/groupwise_empty_solver.h>

namespace sudoku
{
    GroupwiseEmptySolver::GroupwiseEmptySolver(const Dimensions& dims)
        : dims_(&dims)
        , cellValues_(dims.getCellCount())
        , blockCounter_(dims)
    {}

    bool GroupwiseEmptySolver::computeNextSolution()
    {
        Timer timer(metrics_.duration);
        CellCount cellPos = 0;
        CellValue minCellValue = 0;

        // If already solved, pop the last guess before continuing to search
        if (cellValues_[dims_->getCellCount() - 1]) {
            cellPos = dims_->getCellCount() - 1;
            minCellValue = cellValues_[cellPos];
            blockCounter_.clearCellValue(cellPos, minCellValue);
            metrics_.totalBacktracks++;
            cellValues_[cellPos] = 0;
        }

        while (cellPos < dims_->getCellCount()) {
            auto cellValue = blockCounter_.getNextAvailableValue(cellPos, minCellValue);
            if (cellValue) {
                metrics_.totalGuesses++;
                cellValues_[cellPos] = cellValue;
                blockCounter_.setCellValue(cellPos, cellValue);
                cellPos++;
                minCellValue = 0;
            }
            else {
                metrics_.totalBacktracks++;
                if (cellPos == 0) {
                    return false;
                }
                cellPos--;
                minCellValue = cellValues_[cellPos];
                blockCounter_.clearCellValue(cellPos, cellValues_[cellPos]);
                cellValues_[cellPos] = 0;
            }
        }
        return true;
    }
}
