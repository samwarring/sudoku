#include <sudoku/grid_potential.h>

namespace sudoku
{
    GridPotential::GridPotential(const Dimensions& dims) : dims_(dims)
    {
        cellPotentials_.reserve(dims_.getCellCount());
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            cellPotentials_.emplace_back(dims_.getMaxCellValue());
        }
    }

    void GridPotential::setCellValue(size_t cellPos, size_t cellValue)
    {
        for (auto groupNum : dims_.getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedPos].block(cellValue);
            }
        }
    }

    void GridPotential::clearCellValue(size_t cellPos, size_t cellValue)
    {
        for (auto groupNum : dims_.getGroupsForCell(cellPos)) {
            for (auto relatedPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedPos].unblock(cellValue);
            }
        }
    }

    void GridPotential::restrictCellValue(size_t cellPos, size_t cellValue)
    {
        cellPotentials_[cellPos].block(cellValue);
    }
}
