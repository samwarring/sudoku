#ifndef INCLUDED_SUDOKU_GROUPWISE_BLOCK_COUNTER_H
#define INCLUDED_SUDOKU_GROUPWISE_BLOCK_COUNTER_H

#include <sudoku/dimensions.h>

namespace sudoku
{
    /**
     * For each group, this class tracks the values that appear in that group.
     */
    class GroupwiseBlockCounter
    {
        public:
            GroupwiseBlockCounter(const Dimensions& dims);

            void setCellValue(CellCount cellPos, CellValue cellValue);

            void clearCellValue(CellCount cellPos, CellValue cellValue);

            CellValue getNextAvailableValue(CellCount cellPos, CellValue minCellValue) const;

        private:
            using CellValueMask = unsigned long long;

            static CellValueMask getMask(CellValue cellValue) { return static_cast<CellValueMask>(1) << (cellValue - 1); }

            const Dimensions* dims_;
            std::vector<CellValueMask> groupMasks_;
    };
}

#endif