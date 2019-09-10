#include <sudoku/standard.h>
#include <sudoku/dimensions.h>

namespace sudoku
{
    namespace standard
    {
        std::vector<std::vector<size_t>> computeStandardGroups()
        {
            auto rowGroups = computeRowGroups(9, 9);
            auto columnGroups = computeColumnGroups(9, 9);
            auto squareGroups = computeGroupsFromMap(
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
            return joinGroups({rowGroups, columnGroups, squareGroups});
        }
    }
}
