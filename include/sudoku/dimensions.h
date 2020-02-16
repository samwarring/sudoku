#ifndef INCLUDED_SUDOKU_DIMENSIONS_H
#define INCLUDED_SUDOKU_DIMENSIONS_H

#include <vector>
#include <stdexcept>
#include <string>
#include <sudoku/types.h>

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
             *       std::set<CellCount>, but for now, this is not essential.
             */
            Dimensions(CellCount cellCount, CellValue maxCellValue, std::vector<std::vector<CellCount>> cellGroups)
                : cellCount_(cellCount)
                , maxCellValue_(maxCellValue)
                , cellsForEachGroup_(std::move(cellGroups))
                , groupsForEachCell_(computeGroupsForEachCell())
                , relatedCells_(computeRelatedCells())
            {}

            /**
             * Gets the number of cells in the sudoku.
             */
            CellCount getCellCount() const { return cellCount_; }

            /**
             * Gets the maximum value allowed for any cell in the sudoku.
             */
            CellValue getMaxCellValue() const { return maxCellValue_; }

            /**
             * Gets the number of groups in the dimensions.
             */
            GroupCount getNumGroups() const { return cellsForEachGroup_.size(); }

            /**
             * Gets a list of cell positions for the given group num.
             */
            const std::vector<CellCount>& getCellsInGroup(GroupCount groupNum) const { return cellsForEachGroup_[groupNum]; }

            /**
             * Gets a list of group numbers containing the given cell position.
             */
            const std::vector<GroupCount>& getGroupsForCell(CellCount cellPos) const { return groupsForEachCell_[cellPos]; }

            /**
             * Gets a list of unique cell positions related to the given cell position.
             */
            const std::vector<CellCount>& getRelatedCells(CellCount cellPos) const { return relatedCells_[cellPos]; }


        private:

            /**
             * Computes the initial value of groupsForEachCell_
             */
            std::vector<std::vector<CellCount>> computeGroupsForEachCell();

            /**
             * Computes the related cell vectors.
             */
            std::vector<std::vector<CellCount>> computeRelatedCells();

            /**
             * Throws an exception if the object is invalid.
             * \see Dimensions::Dimensions
             */
            void validate() const;

            const CellCount cellCount_;
            const CellValue maxCellValue_;
            const std::vector<std::vector<CellCount>> cellsForEachGroup_;
            const std::vector<std::vector<GroupCount>> groupsForEachCell_;
            const std::vector<std::vector<CellCount>> relatedCells_;
    };
}

#endif
