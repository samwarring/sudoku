#include <algorithm>
#include <sudoku/grid.h>

namespace sudoku
{
    Grid::Grid(
        const Dimensions& dims,
        std::vector<size_t> cellValues,
        std::vector<std::pair<size_t, size_t>> restrictions)
        : dims_(&dims)
        , cellValues_(std::move(cellValues))
    {
        // If given empty vector of cell values, treat as an empty grid.
        if (cellValues_.size() == 0) {
            cellValues_.resize(dims.getCellCount());
        }

        // Initialize cell potentials.
        cellPotentials_.reserve(dims_->getCellCount());
        for (size_t cellPos = 0; cellPos < dims_->getCellCount(); ++cellPos) {
            cellPotentials_.emplace_back(dims_->getMaxCellValue());
        }

        // Verify cell values and set initial potentials.
        if (cellValues_.size() != dims.getCellCount()) {
            throw GridException("Incorrect number of initial values");
        }
        for (size_t cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
            auto cellValue = cellValues_[cellPos];
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

    void Grid::setCellValue(size_t cellPos, size_t cellValue)
    {
        cellValues_[cellPos] = cellValue;
        for (auto groupNum : dims_->getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_->getCellsInGroup(groupNum)) {
                cellPotentials_[relatedPos].block(cellValue);
            }
        }
    }

    void Grid::clearCellValue(size_t cellPos)
    {
        size_t cellValue = cellValues_[cellPos];
        cellValues_[cellPos] = 0;
        for (auto groupNum : dims_->getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_->getCellsInGroup(groupNum)) {
                cellPotentials_[relatedPos].unblock(cellValue);
            }
        }
    }

    void Grid::restrictCellValue(size_t cellPos, size_t cellValue)
    {
        cellPotentials_[cellPos].block(cellValue);
        restrictions_.emplace_back(cellPos, cellValue);
    }

    size_t Grid::getMaxBlockEmptyCell() const
    {
        const size_t cellCount = dims_->getCellCount();
        const size_t maxCellValue = dims_->getMaxCellValue();
        int maxBlock = -1;
        size_t maxBlockPos = cellCount;

        for (size_t cellPos = 0; cellPos < cellCount; ++cellPos) {
            if (cellValues_[cellPos] == 0) {
                // TODO: Consider changing block count type from size_t to int. This
                //       lets us easily compare the block cout to -1.
                const size_t blockCountUnsigned = cellPotentials_[cellPos].getAmountBlocked();
                const int blockCount = static_cast<int>(blockCountUnsigned);
                if (maxBlock < blockCount) {
                    if (blockCount == maxCellValue) {
                        // If an empty cell is completely blocked, there is no greater
                        // block count to search for.
                        return cellPos;
                    }
                    maxBlock = blockCount;
                    maxBlockPos = cellPos;
                }
            }
        }

        return maxBlockPos;
    }
}
