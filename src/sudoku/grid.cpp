#include <algorithm>
#include <sudoku/grid.h>

namespace sudoku
{
    Grid::Grid(
        const Dimensions& dims,
        std::vector<CellValue> cellValues,
        std::vector<Restriction> restrictions)
        : dims_(&dims)
        , blockCountTracker_(dims)
        , cellValues_(dims.getCellCount())
    {
        // Initialize cell potentials.
        cellPotentials_.reserve(dims_->getCellCount());
        for (CellCount cellPos = 0; cellPos < dims_->getCellCount(); ++cellPos) {
            cellPotentials_.emplace_back(dims_->getMaxCellValue());
        }

        // Verify cell values and set initial potentials. If given empty vector
        // of cell values, treat as an empty grid.
        if (cellValues.size() != 0 && cellValues.size() != dims.getCellCount()) {
            throw GridException("Incorrect number of initial values");
        }
        for (CellCount cellPos = 0; cellPos < cellValues.size(); ++cellPos) {
            auto cellValue = cellValues[cellPos];
            if (cellValue != 0) {
                if (cellValue > dims.getMaxCellValue()) {
                    throw GridException("Initial cell value out of range");
                }
                if (cellPotentials_[cellPos].isBlocked(cellValue)) {
                    throw GridException("Initial cell values contain repeated value in a group");
                }
                setCellValue(cellPos, cellValue);
            }
        }

        // Verify and set initial restrictions.
        restrictions_.reserve(restrictions.size());
        for (auto restr : restrictions) {
            if (restr.first >= dims.getCellCount()) {
                throw GridException("Initial restriction position out of range");
            }
            if (restr.second == 0 || restr.second > dims.getMaxCellValue()) {
                throw GridException("Initial restriction value out of range");
            }
            restrictCellValue(restr.first, restr.second);
        }
    }

    void Grid::setCellValue(CellCount cellPos, CellValue cellValue)
    {
        cellValues_[cellPos] = cellValue;
        blockCountTracker_.markCellOccupied(cellPos);
        for (auto groupNum : dims_->getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_->getCellsInGroup(groupNum)) {
                if (cellPotentials_[relatedPos].block(cellValue)) {
                    blockCountTracker_.incrementBlockCount(relatedPos);
                }
            }
        }
    }

    void Grid::clearCellValue(CellCount cellPos)
    {
        auto cellValue = cellValues_[cellPos];
        cellValues_[cellPos] = 0;
        blockCountTracker_.markCellEmpty(cellPos);
        for (auto groupNum : dims_->getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_->getCellsInGroup(groupNum)) {
                if (cellPotentials_[relatedPos].unblock(cellValue)) {
                    blockCountTracker_.derementBlockCount(relatedPos);
                }
            }
        }
    }

    void Grid::restrictCellValue(CellCount cellPos, CellValue cellValue)
    {
        if (cellPotentials_[cellPos].block(cellValue)) {
            blockCountTracker_.incrementBlockCount(cellPos);
        }
        restrictions_.emplace_back(cellPos, cellValue);
    }

    CellCount Grid::getMaxBlockEmptyCell() const
    {
        auto candidate = blockCountTracker_.getMaxBlockEmptyCell();
        if (cellValues_[candidate]) {
            // No more empty cells
            return dims_->getCellCount();
        }
        return candidate;
    }
}
