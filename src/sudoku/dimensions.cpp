#include <sstream>
#include <string>
#include <vector>
#include <sudoku/dimensions.h>

namespace sudoku
{
    std::vector<std::vector<size_t>> computeRowGroups(size_t rowCount, size_t columnCount)
    {
        const bool isEmpty = (rowCount == 0 || columnCount == 0);
        std::vector<std::vector<size_t>> result(isEmpty ? 0 : rowCount);
        for (size_t row = 0; row < rowCount; ++row) {
            for (size_t col = 0; col < columnCount; ++col) {
                size_t cellPos = (row * columnCount) + col;
                result[row].push_back(cellPos);
            }
        }
        return result;
    }

    std::vector<std::vector<size_t>> computeColumnGroups(size_t rowCount, size_t columnCount)
    {
        const bool isEmpty = (rowCount == 0 || columnCount == 0);
        std::vector<std::vector<size_t>> result(isEmpty ? 0 : columnCount);
        for (size_t column = 0; column < columnCount; ++column) {
            for (size_t row = 0; row < rowCount; ++row) {
                size_t cellPos = (row * columnCount) + column;
                result[column].push_back(cellPos);
            }
        }
        return result;
    }

    std::vector<std::vector<size_t>> computeGroupsFromMap(const std::string& groupMap)
    {
        std::istringstream iss(groupMap);
        size_t cellPos = 0;
        size_t groupNum = 0;
        iss >> groupNum;
        std::vector<std::vector<size_t>> result;
        while (iss.good()) {
            if (groupNum + 1 > result.size()) {
                result.resize(groupNum + 1);
            }
            result[groupNum].push_back(cellPos);

            // Read the next parsed group number for the next cell position
            ++cellPos;
            iss >> groupNum;
        }
        return result;
    }

    std::vector<std::vector<size_t>> joinGroups(std::vector<std::vector<std::vector<size_t>>> groupOfGroups)
    {
        std::vector<std::vector<size_t>> result;
        for (const auto& groups : groupOfGroups) {
            result.insert(end(result), begin(groups), end(groups));
        }
        return result;
    }

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
