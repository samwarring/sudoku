#include <sudoku/dimensions.h>

namespace sudoku
{
    std::vector<std::vector<size_t>> Dimensions::computeGroupsForEachCell()
    {
        // Need to validate the state first or else the rest of this
        // function might raise a vector-index-out-of-bounds error.
        validate();

        std::vector<std::vector<size_t>> result(cellCount_);
        for(size_t groupNum = 0; groupNum < cellsForEachGroup_.size(); ++groupNum) {
            for (size_t cellPos : cellsForEachGroup_[groupNum]) {
                result[cellPos].push_back(groupNum);
            }
        }
        return result;
    }

    void Dimensions::validate() const
    {
        if (cellCount_ == 0) {
            throw DimensionsException("Dimensions cellCount is 0");
        }
        if (maxCellValue_ == 0) {
            throw DimensionsException("Dimensions maxCellValue is 0");
        }
        for (const auto& group : cellsForEachGroup_) {
            if (maxCellValue_ < group.size()) {
                throw DimensionsException("Dimensions cellGroup size exceeds maxCellValue");
            }
            for (size_t cellPos : group) {
                if (cellCount_ <= cellPos) {
                    throw DimensionsException("Dimensions cellGroup contains position beyond cellCount");
                }
            }
        }
    }
}
