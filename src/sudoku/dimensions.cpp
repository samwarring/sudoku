#include <sudoku/dimensions.h>

namespace sudoku
{
    std::vector<std::vector<size_t>> Dimensions::computeGroupsForEachCell()
    {
        std::vector<std::vector<size_t>> result(cellCount_);
        for(size_t groupNum = 0; groupNum < cellsForEachGroup_.size(); ++groupNum) {
            for (size_t cellPos : cellsForEachGroup_[groupNum]) {
                result[cellPos].push_back(groupNum);
            }
        }
        return result;
    }
}
