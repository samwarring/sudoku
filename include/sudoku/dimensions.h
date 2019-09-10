#ifndef INCLUDED_SUDOKU_DIMENSIONS_H
#define INCLUDED_SUDOKU_DIMENSIONS_H

#include <vector>
#include <string>

namespace sudoku
{
    /**
     * Computes a vector of cell groups for the given dimensions. Each group is a row
     * of the grid.
     * 
     * \return A vector of vectors, each containing the cell positions in thier respective row.
     */
    std::vector<std::vector<size_t>> computeRowGroups(size_t rowCount, size_t columnCount);

    /**
     * Computes a vector of cell groups for the given dimensions. Each group is a column
     * of the grid.
     * 
     * \return A vector of vectors, each containing the cell positions in their respective column.
     */
    std::vector<std::vector<size_t>> computeColumnGroups(size_t rowCount, size_t columnCount);

    /**
     * Computes a vector of cell groups specified by a string which "maps" each cell position
     * to a unique group.
     * 
     * \return A vector of vectors, each contianing the cell positions in their respective group.
     * 
     * For example, if you wanted to break a 4x4 sudoku into 4 groups where each group is a
     * quadrant, you would use the function like this:
     * 
     *     std::string groupMap = "0 0 1 1 " +
     *                            "0 0 1 1 " +
     *                            "2 2 3 3 " +
     *                            "2 2 3 3 " ;
     *     auto groups = computeGroupsFromMap(groupMap);
     * 
     * This return a vector of 4 groups. The first group (0) contains {0, 1, 4, 5}; the next
     * group (1) contains {2, 3, 6, 7}; etc.
     * 
     */
    std::vector<std::vector<size_t>> computeGroupsFromMap(const std::string& groupMap);

    /**
     * Joins mulitple group vectors into a single group vector.
     * 
     * \param groupOfGroups This is a vector where each element is the result of a
     *                      "computeGroups" function. That is: each element is itself 
     *                      vector of groups.
     * 
     * \return A vector of groups (much like the result of a "computeGroups" function). This
     *         group vector is the union of all groups provided in groupOfGroups.
     * 
     * For example, to obtain the complete set of groups for a 4x4 sudoku, use the function
     * like this:
     * 
     *     auto rowGroups = sudoku::computeRowGroups(4, 4);        // 4 groups
     *     auto columnGroups = sudoku::computeColumnGroups(4, 4);  // 4 groups
     *     auto squareGroups = sudoku::computeGroupsFromMap(       // 4 groups
     *         "0 0 1 1 "
     *         "0 0 1 1 "
     *         "2 2 3 3 "
     *         "2 2 3 3 "
     *     );
     *     auto totalGroups = sudoku::joinGroups({rowGroups, columnGroups, squareGroups});  // 12 groups
     */
    std::vector<std::vector<size_t>> joinGroups(std::vector<std::vector<std::vector<size_t>>> groupOfGroups);

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
             *                   \ref computeColumnGroups, and \ref computeGroupsFromMap functions via th
             *                   \ref joinGroups function.
             * 
             * \note Dimensions objects do not specify the *positions* of the cells. They treat the cells
             *       as a 1-dimensional list - each with a unique position. The topography of related cells
             *       are encoded in the `cellGroups` parameter. To specify that a sudoku is rendered as a 
             *       2-dimensional grid, see the \ref Formatter class.
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

            const size_t cellCount_;
            const size_t maxCellValue_;
            const std::vector<std::vector<size_t>> cellsForEachGroup_;
            const std::vector<std::vector<size_t>> groupsForEachCell_;
    };
}

#endif
