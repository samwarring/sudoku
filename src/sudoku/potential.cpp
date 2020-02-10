#include <sudoku/potential.h>

namespace sudoku
{
    bool Potential::block(CellValue cellValue)
    {
        return (++block_counts_[cellValue - 1] == 1);
    }

    bool Potential::unblock(CellValue cellValue)
    {
        return (--block_counts_[cellValue - 1] == 0);
    }

    CellValue Potential::getNextAvailableValue(CellValue minValue) const
    {
        const CellValue maxCellValue = block_counts_.size();
        for (CellValue cellValue = minValue + 1; cellValue <= maxCellValue; ++cellValue) {
            if (block_counts_[cellValue - 1] == 0) {
                return cellValue;
            }
        }
        return 0;
    }

    std::vector<CellValue> Potential::getAvailableValues() const
    {
        const CellValue maxCellValue = block_counts_.size();
        std::vector<CellValue> result;
        for (CellValue cellValue = 1; cellValue <= maxCellValue; ++cellValue) {
            if (block_counts_[cellValue - 1] == 0) {
                result.push_back(cellValue);
            }
        }
        return result;
    }
}