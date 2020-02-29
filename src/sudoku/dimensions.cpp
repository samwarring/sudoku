#include <sstream>
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

    void Dimensions::validateCellValues(const std::vector<CellValue>& cellValues) const
    {
        if (cellValues.size() > cellCount_) {
            throw std::runtime_error("Too many cell values for dimensions");
        }
        if (cellValues.size() < cellCount_) {
            throw std::runtime_error("Not enough cell values for dimensions");
        }

        // contains: [{values in group 0}, {values in group 1}, ... {values in group G}]
        std::vector<std::unordered_set<CellValue>> groupValues(getNumGroups());

        for (CellCount cellPos = 0; cellPos < cellCount_; ++cellPos) {
            auto cellValue = cellValues[cellPos];
            if (cellValue == 0) {
                continue;
            }
            if (cellValue > maxCellValue_) {
                throw std::runtime_error("Cell value exceeds max for dimensions");
            }

            // Non-empty cells must not repeat a value within a group.
            for (auto groupNum : getGroupsForCell(cellPos)) {
                if (groupValues[groupNum].find(cellValue) != groupValues[groupNum].end()) {
                    std::ostringstream oss;
                    oss << "Group " << groupNum << " contains repeated cell value " << cellValue;
                    throw std::runtime_error(oss.str());
                }
                groupValues[groupNum].insert(cellValue);
            }
        }
    }
}
