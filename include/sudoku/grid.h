#ifndef INCLUDED_SUDOKU_GRID_POTENTIAL_H
#define INCLUDED_SUDOKU_GRID_POTENTIAL_H

#include <stdexcept>
#include <vector>
#include <sudoku/block_count_tracker.h>
#include <sudoku/dimensions.h>
#include <sudoku/potential.h>
#include <sudoku/types.h>

namespace sudoku
{
    /**
     * Thrown when constructing an invalid \ref Grid
     */
    class GridException : public std::logic_error { using std::logic_error::logic_error; };

    /**
     * A Grid tracks the cell values and potentials for a sudoku.
     * 
     * \warning Methods do not validate cellPos and cellValue arguments.
     *          Take care that cellPos < cellCount and 1 <= cellValue <= maxCellValue
     *          when calling these methods.
     */
    class Grid
    {
        public:

            /**
             * Cell value restrictions represented by pairs of <pos, value>
             * indicate that cellValues[pos] should never be assigned 'value'.
             * 
             * \todo Re-define this as a struct in sudoku/types.h
             */
            using Restriction = std::pair<CellCount, CellValue>;

            /**
             * Construct a new grid with the given initial values and cell restrictions.
             * 
             * \param cellValues initial cell values. If empty vector, it is treated
             *                   as an empty grid. If not empty, its size must equal
             *                   dimensions' cell count. Each cell value must be
             *                   less than or equal to dimensions' max cell value.
             * 
             * \param restrictions pos-value pairs to be restricted. Each position
             *                     must be less than dimensions' cell count. Each
             *                     value must be less than or equal to the dimensions'
             *                     max cell value.
             * 
             * \warning Since Grid only holds a pointer to the dimensions,
             *          the dimensions must out-live the grid.
             * 
             * \throw GridException if cellValues or restrictions are invalid.
             */
            Grid(
                const Dimensions& dims,
                std::vector<CellValue> cellValues = {},
                std::vector<Restriction> restrictions = {}
            );

            /**
             * Get grid dimensions
             */
            const Dimensions& getDimensions() const { return *dims_; }

            /**
             * Get read-only access to a cell potential.
             */
            const Potential& getCellPotential(CellCount cellPos) const { return cellPotentials_[cellPos]; }

            /**
             * Get value at a cell.
             */
            CellValue getCellValue(CellCount cellPos) const { return cellValues_[cellPos]; }

            /**
             * Get read-only access to all cell values.
             */
            const std::vector<CellValue>& getCellValues() const { return cellValues_; }

            /**
             * Get read-only access to the restrictions.
             */
            const std::vector<Restriction>& getRestrictions() const { return restrictions_; }

            /**
             * Sets a cell value, and blocks the given cell value for all cells
             * that share a group with the given cell position.
             * 
             * \warning If the cell already contains a value > 0, it will be overwritten
             *          without unblocking the value from related potentials.
             */
            void setCellValue(CellCount cellPos, CellValue cellValue);

            /**
             * Unblocks the given cell value for all cells that share a group
             * with the given cell position.
             */
            void clearCellValue(CellCount cellPos);

            /**
             * Block an individual cell from a certain value without affecting
             * any cells that share a group with it.
             */
            void restrictCellValue(CellCount cellPos, CellValue cellValue);

            /**
             * Get positon of an empty cell with the highest block count.
             * 
             * \return pos < cellCount if an empty cell exists;
             *         pos == cellCount if no empty cell exists.
             */
            CellCount getMaxBlockEmptyCell() const;

            /**
             * Checks if the grid is full (no empty cells)
             */
            bool isFull() const { return getMaxBlockEmptyCell() == dims_->getCellCount(); }

            /**
             * Get the number of values that are blocked for a cell.
             */
            CellBlockCount getBlockCount(CellCount cellPos) const { return blockCountTracker_.getBlockCount(cellPos); }

        private:
            const Dimensions* dims_;
            BlockCountTracker blockCountTracker_;
            std::vector<Potential> cellPotentials_;
            std::vector<CellValue> cellValues_;
            std::vector<Restriction> restrictions_;
    };
}

#endif
