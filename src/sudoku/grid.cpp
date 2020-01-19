#include <sudoku/grid.h>

namespace sudoku
{
    Grid::Grid(const Dimensions& dims) : dims_(dims)
    {
        cellPotentials_.reserve(dims_.getCellCount());
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            cellPotentials_.emplace_back(dims_.getMaxCellValue());
        }
    }

    void Grid::setCellValue(size_t cellPos, size_t cellValue)
    {
        for (auto groupNum : dims_.getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedPos].block(cellValue);
            }
        }
    }

    void Grid::clearCellValue(size_t cellPos, size_t cellValue)
    {
        for (auto groupNum : dims_.getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedPos].unblock(cellValue);
            }
        }
    }

    void Grid::restrictCellValue(size_t cellPos, size_t cellValue)
    {
        cellPotentials_[cellPos].block(cellValue);
    }
}
