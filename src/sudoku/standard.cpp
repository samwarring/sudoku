#include <sudoku/standard.h>
#include <sudoku/dimensions.h>

namespace sudoku
{
    const static auto STANDARD_ROWS = computeRowGroups(9, 9);
    const static auto STANDARD_COLUMNS = computeColumnGroups(9, 9);
    const static auto STANDARD_SQUARES = computeGroupsFromMap(
        " 0 0 0  1 1 1  2 2 2 "
        " 0 0 0  1 1 1  2 2 2 "
        " 0 0 0  1 1 1  2 2 2 "
        //---------------------
        " 3 3 3  4 4 4  5 5 5 "
        " 3 3 3  4 4 4  5 5 5 "
        " 3 3 3  4 4 4  5 5 5 "
        //---------------------
        " 6 6 6  7 7 7  8 8 8 "
        " 6 6 6  7 7 7  8 8 8 "
        " 6 6 6  7 7 7  8 8 8 "
    );

    const std::vector<std::vector<size_t>> STANDARD_GROUPS = joinGroups({
        STANDARD_ROWS, STANDARD_COLUMNS, STANDARD_SQUARES
    });

    const Dimensions STANDARD_DIMENSIONS(
        STANDARD_CELL_COUNT,
        STANDARD_MAX_CELL_VALUE,
        STANDARD_GROUPS
    );
}
