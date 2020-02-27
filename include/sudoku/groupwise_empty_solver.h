#ifndef INCLUDED_SUDOKU_GROUPWISE_EMPTY_SOLVER_H
#define INCLUDED_SUDOKU_GROUPWISE_EMPTY_SOLVER_H

#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/groupwise_block_counter.h>
#include <sudoku/solver_interface.h>

namespace sudoku
{
    /**
     * Solves empty sudokus by guessing cell values from "left to right". The next cell
     * is selected as the next empty cell in the sudoku. This class uses the
     * GroupwiseBlockCounter to track which values are available for each cell - which
     * makes this solver very memory-efficient for large square sudokus.
     */
    class GroupwiseEmptySolver : public SolverInterface
    {
        public:
            GroupwiseEmptySolver(const Dimensions& dims);

            bool computeNextSolution();

            const std::vector<CellValue>& getCellValues() const { return cellValues_; }

            Metrics getMetrics() const { return metrics_; }

        private:
            const Dimensions* dims_;
            std::vector<CellValue> cellValues_;
            GroupwiseBlockCounter blockCounter_;
            Metrics metrics_;
    };
}

#endif
