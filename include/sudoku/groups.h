#ifndef INCLUDED_SUDOKU_GROUPS_H
#define INCLUDED_SUDOKU_GROUPS_H

#include <string>
#include <vector>

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
     * \todo
     * -# Specify behavior when non-integers are parsed
     * -# Specify behavior when group numbers are skipped (e.g. "0 1 2 99")
     * -# Include ability to 'ignore' positions so they don't belong to any group
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

}

#endif