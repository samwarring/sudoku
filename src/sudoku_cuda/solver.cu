#include <sudoku/cuda/solver.h>

namespace sudoku
{
    namespace cuda
    {
        CUDA_HOST_AND_DEVICE
        Solver::Solver(Dimensions dims, Grid grid, GuessStack guessStack)
            : dims_(dims)
            , grid_(grid)
            , guessStack_(guessStack)
        {}

        CUDA_HOST_AND_DEVICE
        Result Solver::computeNextSolution(size_t maxGuessCount)
        {
            size_t cellPos = grid_.getMaxBlockEmptyCell();
            size_t minCellValue = 0;
            while (cellPos != dims_.getCellCount()) {
                
                if (maxGuessCount == 0) {
                    // Out of guesses
                    return Result::OK_TIMED_OUT;
                }

                size_t cellValue = grid_.getNextAvailableValue(cellPos, minCellValue);
                if (cellValue == 0) {
                    // No more available values. Backtrack.
                    if (guessStack_.getSize() == 0) {
                        // No solution.
                        return Result::OK_NO_SOLUTION;
                    }
                    // Pop last guess; clear guess from grid; continue.
                    size_t prevGuessPos = guessStack_.pop();
                    minCellValue = grid_.getCellValue(prevGuessPos);
                    grid_.clearCellValue(prevGuessPos);
                    cellPos = prevGuessPos;
                    continue;
                }

                // We have an available value. Try it, and continue guessing.
                guessStack_.push(cellPos);
                grid_.setCellValue(cellPos, cellValue);
                cellPos = grid_.getMaxBlockEmptyCell();
                minCellValue = 0;
                maxGuessCount--;
            }

            // No more empty cells. Sudoku is solved!
            return Result::OK_FOUND_SOLUTION;
        }
    }
}
