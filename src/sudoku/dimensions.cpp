#include <unordered_set>
#include <sudoku/dimensions.h>

namespace sudoku
{
    std::vector<std::vector<GroupCount>> Dimensions::computeGroupsForEachCell()
    {
        // Need to validate the state first or else the rest of this
        // function might raise a vector-index-out-of-bounds error.
        validate();

        std::vector<std::vector<GroupCount>> result(cellCount_);
        for(GroupCount groupNum = 0; groupNum < cellsForEachGroup_.size(); ++groupNum) {
            for (CellCount cellPos : cellsForEachGroup_[groupNum]) {
                result[cellPos].push_back(groupNum);
            }
        }
        return result;
    }

    std::vector<std::vector<CellCount>> Dimensions::computeRelatedCells()
    {
        std::vector<std::vector<CellCount>> result;
        result.reserve(cellCount_);

        for (CellCount cellPos = 0; cellPos < cellCount_; ++cellPos) {
            std::unordered_set<CellCount> relatedCellSet;
            for (auto groupNum : groupsForEachCell_[cellPos]) {
                for (auto relatedPos : cellsForEachGroup_[groupNum]) {
                    if (relatedPos != cellPos) {
                        relatedCellSet.insert(relatedPos);
                    }
                }
            }
            result.emplace_back(relatedCellSet.begin(), relatedCellSet.end());
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
            for (auto cellPos : group) {
                if (cellCount_ <= cellPos) {
                    throw DimensionsException("Dimensions cellGroup contains position beyond cellCount");
                }
            }
        }
    }
}
