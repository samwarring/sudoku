#include <sudoku/potential.h>

namespace sudoku
{
    bool Potential::block(size_t cellValue)
    {
        return (++block_counts_[cellValue - 1] == 1);
    }

    bool Potential::unblock(size_t cellValue)
    {
        return (--block_counts_[cellValue - 1] == 0);
    }

    size_t Potential::getNextAvailableValue(size_t minValue) const
    {
        const size_t maxCellValue = block_counts_.size();
        for (size_t cellValue = minValue + 1; cellValue <= maxCellValue; ++cellValue) {
            if (block_counts_[cellValue - 1] == 0) {
                return cellValue;
            }
        }
        return 0;
    }

    std::vector<size_t> Potential::getAvailableValues() const
    {
        const size_t maxCellValue = block_counts_.size();
        std::vector<size_t> result;
        for (size_t cellValue = 1; cellValue <= maxCellValue; ++cellValue) {
            if (block_counts_[cellValue - 1] == 0) {
                result.push_back(cellValue);
            }
        }
        return result;
    }
}