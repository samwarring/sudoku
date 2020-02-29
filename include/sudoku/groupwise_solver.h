#ifndef INCLUDED_SUDOKU_GROUPWISE_SOLVER_H
#define INCLUDED_SUDOKU_GROUPWISE_SOLVER_H

#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/groupwise_block_counter.h>
#include <sudoku/solver_interface.h>

namespace sudoku
{
    class GroupwiseSolver : public SolverInterface
    {
        public:
            GroupwiseSolver(const Dimensions& dims, std::vector<CellValue> cellValues = {});

            bool computeNextSolution();

            const std::vector<CellValue>& getCellValues() const { return cellValues_; }

            Metrics getMetrics() const { return metrics_; }

        private:
            CellCount getNextEmptyCell(CellCount cellPos) const;

            const Dimensions* dims_;
            std::vector<CellValue> cellValues_;
            std::vector<CellCount> guessStack_;
            size_t guessStackPos_;
            GroupwiseBlockCounter blockCounter_;
            Metrics metrics_;
    };
}

#endif
