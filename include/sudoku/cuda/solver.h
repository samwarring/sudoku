#ifndef INCLUDED_SUDOKU_CUDA_SOLVER_H
#define INCLUDED_SUDOKU_CUDA_SOLVER_H

#include <vector>
#include <sudoku/cuda/device_buffer.h>
#include <sudoku/cuda/types.h>
#include <sudoku/grid.h>
#include <sudoku/metrics.h>

namespace sudoku
{
    namespace cuda
    {
        class Solver
        {
            public:
                Solver(const sudoku::Grid& grid);

                bool computeNextSolution();

                Result computeNextSolution(unsigned guessCount);

                const std::vector<sudoku::cuda::CellValue>& getCellValues() const;

                Metrics getMetrics() const { return metrics_; }

                /**
                 * Convert between sudoku::* classes' cell values (size_t)
                 * and  sudoku::cuda::* classes' cell values (sudoku::cuda::CellValue)
                 * \todo Unify data types across both namespaces.
                 */
                static std::vector<CellValue> castCellValues(const std::vector<size_t>& cellValues);
                static std::vector<size_t> castCellValues(const std::vector<CellValue>& cellValues);

            private:
                // Common
                CellCount cellCount_;
                CellValue maxCellValue_;
                CellCount cellCountPow2_;
                unsigned sharedMemSize_;
                // Related groups
                GroupCount totalGroupCount_;
                DeviceBuffer<GroupCount> groupCounts_;
                DeviceBuffer<GroupCount> groupIds_;
                // Block counter
                DeviceBuffer<CellBlockCount> cellBlockCounts_;
                DeviceBuffer<ValueBlockCount> valueBlockCounts_;
                // Guess stack
                DeviceBuffer<Guess> guessStackValues_;
                DeviceBuffer<CellCount> guessStackSize_;
                // Grid
                std::vector<CellValue> hostCellValues_;
                DeviceBuffer<CellValue> deviceCellValues_;

                // Misc
                Metrics metrics_;
                bool initCellValuesDone_ = false;

                // Run a single batch of guesses. Do not copy cell values back to host.
                Result computeNextSolutionOneBatch(unsigned guessCount);
        };
    }
}

#endif
