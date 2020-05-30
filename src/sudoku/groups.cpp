#include <sstream>
#include <sudoku/groups.h>

namespace sudoku
{
    std::vector<std::vector<CellCount>> computeRowGroups(size_t rowCount, size_t columnCount)
    {
        const bool isEmpty = (rowCount == 0 || columnCount == 0);
        std::vector<std::vector<CellCount>> result(isEmpty ? 0 : rowCount);
        for (size_t row = 0; row < rowCount; ++row) {
            for (size_t col = 0; col < columnCount; ++col) {
                CellCount cellPos = (row * columnCount) + col;
                result[row].push_back(cellPos);
            }
        }
        return result;
    }

    std::vector<std::vector<CellCount>> computeColumnGroups(size_t rowCount, size_t columnCount)
    {
        const bool isEmpty = (rowCount == 0 || columnCount == 0);
        std::vector<std::vector<CellCount>> result(isEmpty ? 0 : columnCount);
        for (size_t column = 0; column < columnCount; ++column) {
            for (size_t row = 0; row < rowCount; ++row) {
                CellCount cellPos = (row * columnCount) + column;
                result[column].push_back(cellPos);
            }
        }
        return result;
    }

    std::vector<std::vector<CellCount>> computeSquareGroups(size_t root)
    {
        // Each square has its own "square-row" and "square-column" separate from
        // the "cell-row" and "cell-column". For example, 9x9 sudokus' first three
        // "cell-rows" are all within the first "square-row".
        const CellCount cellCount = root * root * root * root;
        const CellValue maxCellValue = castCellValue(root * root);
        std::vector<std::vector<CellCount>> result(root * root);
        for (CellCount cellPos = 0; cellPos < cellCount; ++cellPos) {
            size_t row = cellPos / maxCellValue;
            size_t column = cellPos % maxCellValue;
            size_t squareRow = row / root;
            size_t squareColumn = column / root;
            size_t squareIndex = (squareRow * root) + squareColumn;
            result[squareIndex].push_back(cellPos);
        }
        return result;
    }

    std::vector<std::vector<CellCount>> computeInnerRectangularGroups(size_t innerRowCount,
                                                                      size_t innerColumnCount)
    {
        size_t totalCols = innerRowCount * innerColumnCount;
        size_t cellCount = totalCols * totalCols;
        size_t horizontalBandSize = innerRowCount * innerColumnCount * innerRowCount;
        std::vector<std::vector<CellCount>> groups(innerRowCount * innerColumnCount);
        GroupCount groupNum = 0;
        for (CellCount cellPos = 0; cellPos < cellCount; ++cellPos) {
            if (cellPos == 0) {
                // First cell.
                groupNum = 0;
            }
            else if ((cellPos % totalCols == 0) && (cellPos % horizontalBandSize == 0)) {
                // Starting a new horzontal band of groups
                groupNum++;
            }
            else if (cellPos % totalCols == 0) {
                // Starting a new row (same horizontal band)
                groupNum -= (innerRowCount - 1);
            }
            else if (cellPos % innerColumnCount == 0) {
                // Starting a new group in the current row
                groupNum++;
            }
            groups[groupNum].push_back(cellPos);
        }
        return groups;
    }

    std::vector<std::vector<CellCount>> computeGroupsFromMap(const std::string& groupMap)
    {
        std::istringstream iss(groupMap);
        CellCount cellPos = 0;
        GroupCount groupNum = 0;
        iss >> groupNum;
        std::vector<std::vector<CellCount>> result;
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

    std::vector<std::vector<CellCount>> joinGroups(std::vector<std::vector<std::vector<CellCount>>> groupOfGroups)
    {
        std::vector<std::vector<CellCount>> result;
        for (const auto& groups : groupOfGroups) {
            result.insert(end(result), begin(groups), end(groups));
        }
        return result;
    }
}