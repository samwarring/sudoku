#ifndef INCLUDED_SUDOKU_DIMENSIONS_H
#define INCLUDED_SUDOKU_DIMENSIONS_H

#include <vector>
#include <stdexcept>
#include <string>

namespace sudoku
{
    /**
     * Error raised when initializing an invalid \ref Dimensions object
     */
    class DimensionsException : public std::logic_error
    {
        using logic_error::logic_error;
    };

    /**
     * Describes the size of the sudoku as well as the relationships between cells.
     */
    class Dimensions
    {
        public:

            /**
             * Initializes the dimensions.
             * 
             * \param cellGroups A collection of groups, where each group contains at most `maxCellValue`
             *                   cells. A solution to the sudoku must not assign the same value to any 
             *                   two cells if they share a group. Values for this parameter are typically
             *                   formed by joining the results of the \ref computeRowGroups,
             *                   \ref computeColumnGroups, and \ref computeGroupsFromMap functions via the
             *                   \ref joinGroups function.
             * 
             * \throws DimensionsError 
             *         1. if any cell group contains a position beyond cellCount.
             *         2. if any cell group size exceeds maxCellValue.
             *         3. if cellCount <= 0.
             *         4. if maxCellValue <= 0.
             * 
             * \note Dimensions objects treat the cells as a 1-dimensional list - each with a unique
             *       position. The topography of related cells are encoded in the `cellGroups` parameter.
             *       To specify that a sudoku is rendered as a 2-dimensional grid, see the \ref Formatter
             *       class.
             * 
             * \note Undefined behavior
             *       1. if any cell group contains duplicate values.
             *       2. if any cell groups contain the excact same values.
             *       These situations can be addressed in the future by passing groups as
             *       std::set<size_t>, but for now, this is not essential.
             */
            Dimensions(size_t cellCount, size_t maxCellValue, std::vector<std::vector<size_t>> cellGroups)
                : cellCount_(cellCount)
                , maxCellValue_(maxCellValue)
                , cellsForEachGroup_(std::move(cellGroups))
                , groupsForEachCell_(computeGroupsForEachCell())
            {}

            /**
             * Gets the number of cells in the sudoku.
             */
            size_t getCellCount() const { return cellCount_; }

            /**
             * Gets the maximum value allowed for any cell in the sudoku.
             */
            size_t getMaxCellValue() const { return maxCellValue_; }

            /**
             * Gets a list of cell positions for the given group num.
             */
            const std::vector<size_t>& getCellsInGroup(size_t groupNum) const { return cellsForEachGroup_[groupNum]; }

            /**
             * Gets a list of group numbers containing the given cell position.
             */
            const std::vector<size_t>& getGroupsForCell(size_t cellPos) const { return groupsForEachCell_[cellPos]; }

        private:

            /**
             * Computes the initial value of groupsForEachCell_
             */
            std::vector<std::vector<size_t>> computeGroupsForEachCell();

            /**
             * Throws an exception if the object is invalid.
             * \see Dimensions::Dimensions
             */
            void validate() const;

            const size_t cellCount_;
            const size_t maxCellValue_;
            const std::vector<std::vector<size_t>> cellsForEachGroup_;
            const std::vector<std::vector<size_t>> groupsForEachCell_;
    };
}

#endif
