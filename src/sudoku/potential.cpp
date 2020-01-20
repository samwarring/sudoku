#include <sudoku/potential.h>

namespace sudoku
{
    void Potential::block(size_t cellValue)
    {
        if (block_counts_[cellValue - 1] == 0) {
            numCellValuesBlocked_++;
        }
        block_counts_[cellValue - 1]++;
    }

    void Potential::unblock(size_t cellValue)
    {
        if (block_counts_[cellValue - 1] == 1) {
            numCellValuesBlocked_--;
        }
        block_counts_[cellValue - 1]--;
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
        result.reserve(maxCellValue - numCellValuesBlocked_);
        for (size_t cellValue = 1; cellValue <= maxCellValue; ++cellValue) {
            if (block_counts_[cellValue - 1] == 0) {
                result.push_back(cellValue);
            }
        }
        return result;
    }
}