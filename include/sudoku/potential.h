#ifndef INCLUDED_SUDOKU_POTENTIAL_H
#define INCLUDED_SUDOKU_POTENTIAL_H

#include <vector>
#include <sudoku/types.h>

namespace sudoku
{
    /**
     * A Potential is an object that tracks the potential values
     * of a sudoku cell. The object tracks which values are available,
     * which values are blocked, how many times a value has been blocked,
     * as well as how many cell values remain unblocked.
     * 
     */
    class Potential
    {
        public:
            /**
             * Create a potential for the given number of cell values.
             */
            Potential(CellValue maxCellValue) : maxCellValue_(maxCellValue), block_counts_(maxCellValue) {}

            /**
             * Indicate that a related cell position claimed `cellValue`.
             * If previously available, this cellValue is now blocked.
             * 
             * \return true if the value was previously unblocked.
             */
            bool block(CellValue cellValue);

            /**
             * Indicate that a related cell position previously set to
             * `cellValue` has reset. If the related cell position was
             * the only position blocking this value, then the value is
             * now available.
             * 
             * \return true if the value was previously blocked.
             */
            bool unblock(CellValue cellValue);

            /**
             * Checks if at least one related cell is blocking the given
             * `cellValue`.
             */
            bool isBlocked(CellValue cellValue) const { return block_counts_[cellValue - 1] > 0; }

            /**
             * Gets the next available cell value greater than `minValue`.
             * If no such value exists, return 0.
             */
            CellValue getNextAvailableValue(CellValue minValue) const;

            /**
             * Gets a vector containing all available values.
             */
            std::vector<CellValue> getAvailableValues() const;

        private:
            CellValue maxCellValue_;
            std::vector<ValueBlockCount> block_counts_;
    };
}

#endif
