#ifndef INCLUDED_SUDOKU_GRID_POTENTIAL_H
#define INCLUDED_SUDOKU_GRID_POTENTIAL_H

#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/potential.h>

namespace sudoku
{
    /**
     * A Grid tracks the cell values and potentials for a sudoku.
     * 
     * \warning Methods do not validate cellPos and cellValue arugments.
     *          Take care that cellPos < cellCount and 1 <= cellValue <= maxCellValue
     *          when calling these methods.
     */
    class Grid
    {
        public:

            Grid(const Dimensions& dims);

            /**
             * Get read-only access to a cell potential.
             */
            const Potential& getCellPotential(size_t cellPos) const { return cellPotentials_[cellPos]; }

            /**
             * Get value at a cell.
             */
            size_t getCellValue(size_t cellPos) const { return cellValues_[cellPos]; }

            /**
             * Sets a cell value, and blocks the given cell value for all cells
             * that share a group with the given cell position.
             * 
             * \warning If the cell already contains a value > 0, it will be overwritten
             *          without unblocking the value from related potentials.
             */
            void setCellValue(size_t cellPos, size_t cellValue);

            /**
             * Unblocks the given cell value for all cells that share a group
             * with the given cell position.
             */
            void clearCellValue(size_t cellPos);

            /**
             * Block an individual cell from a certain value without affecting
             * any cells that share a group with it.
             */
            void restrictCellValue(size_t cellPos, size_t cellValue);

        private:
            const Dimensions& dims_;
            std::vector<Potential> cellPotentials_;
            std::vector<size_t> cellValues_;
    };
}

#endif
