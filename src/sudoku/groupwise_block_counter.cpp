#include <sudoku/groupwise_block_counter.h>

namespace sudoku
{
    GroupwiseBlockCounter::GroupwiseBlockCounter(const sudoku::Dimensions& dims)
        : dims_(&dims)
        , groupMasks_(dims.getNumGroups())
    {
        if (dims.getMaxCellValue() > (sizeof(CellValueMask) * 8)) {
            throw std::runtime_error("Too many cell values for GroupwiseBlockCounter");
        }
    }

    void GroupwiseBlockCounter::setCellValue(CellCount cellPos, CellValue cellValue)
    {
        CellValueMask mask = getMask(cellValue);
        for (auto groupNum : dims_->getGroupsForCell(cellPos)) {
            groupMasks_[groupNum] |= mask;
        }
    }

    void GroupwiseBlockCounter::clearCellValue(CellCount cellPos, CellValue cellValue)
    {
        CellValueMask mask = ~getMask(cellValue);
        for (auto groupNum : dims_->getGroupsForCell(cellPos)) {
            groupMasks_[groupNum] &= mask;
        }
    }

    CellValue GroupwiseBlockCounter::getNextAvailableValue(CellCount cellPos, CellValue minCellValue) const
    {
        CellValueMask mask = 0;
        for (auto groupNum : dims_->getGroupsForCell(cellPos)) {
            mask |= groupMasks_[groupNum];
        }
        mask >>= minCellValue;
        // TODO: Can the following loop be replaced with a more efficient intrinsic?
        for (CellValue cellValue = minCellValue + 1; cellValue <= dims_->getMaxCellValue(); ++cellValue) {
            if (mask & 1) {
                mask >>= 1;
            }
            else {
                return cellValue;
            }
        }
        return 0;
    }
}