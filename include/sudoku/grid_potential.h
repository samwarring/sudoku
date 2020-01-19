#ifndef INCLUDED_SUDOKU_GRID_POTENTIAL_H
#define INCLUDED_SUDOKU_GRID_POTENTIAL_H

#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/potential.h>

namespace sudoku
{
    /**
     * A GridPotential tracks the potential values of all cells in a
     * sudoku grid.
     */
    class GridPotential
    {
        public:

            /**
             * Creates an internal array of cell potentials for each
             * cell in the dimensions
             */
            GridPotential(const Dimensions& dims);

            /**
             * Get read-only access to a cell potential
             */
            const Potential& getCellPotential(size_t cellPos) { return cellPotentials_[cellPos]; }

            /**
             * Blocks the given cell value for all cells that share a group
             * with the given cell position.
             */
            void setCellValue(size_t cellPos, size_t cellValue);

            /**
             * Unblocks the given cell value for all cells that share a group
             * with the given cell position.
             */
            void clearCellValue(size_t cellPos, size_t cellValue);

            /**
             * Block an individual cell from a certain value without affecting
             * any cells that share a group with it.
             */
            void restrictCellValue(size_t cellPos, size_t cellValue);

        private:
            const Dimensions& dims_;
            std::vector<Potential> cellPotentials_;
    };
}

#endif
