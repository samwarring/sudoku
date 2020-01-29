#include <algorithm>
#include <iterator>
#include <sudoku/cuda/guess_stack.h>

namespace sudoku
{
    namespace cuda
    {
        GuessStack::HostData::HostData(const std::vector<sudoku::Grid>& grids)
        {
            // Get maximum empty cell count for all grids
            size_t maxEmptyCellCount = 0;
            for (const auto& grid : grids) {
                const size_t emptyCellCount = grid.getEmptyCellCount();
                if (emptyCellCount > maxEmptyCellCount) {
                    maxEmptyCellCount = emptyCellCount;
                }
            }

            // Allocate enough space for each grid to guess all empty cells.
            values_.resize(maxEmptyCellCount * grids.size());

            // Each thread gets its own "current size"
            sizes_.resize(grids.size());

            data_.values = values_.data();
            data_.sizes = sizes_.data();
            data_.threadCount = grids.size();
        }

        GuessStack::DeviceData::DeviceData(const HostData& hostData)
            : values_(hostData.values_)
            , sizes_(hostData.sizes_)
            , data_(hostData.data_)
        {
            data_.values = values_.begin();
            data_.sizes = sizes_.begin();
        }

        void GuessStack::push(size_t cellPos)
        {
            *top() = cellPos;
            (*size())++;
        }

        size_t GuessStack::pop()
        {
            CUDA_HOST_ASSERT(*size() > 0);
            (*size())--;
            return *top();
        }

        size_t GuessStack::getSize()
        {
            return *size();
        }

        size_t* GuessStack::size()
        {
            return data_.sizes + threadNum_;
        }

        size_t* GuessStack::top()
        {
            return data_.values + (*size() * data_.threadCount) + threadNum_;
        }
    }
}
