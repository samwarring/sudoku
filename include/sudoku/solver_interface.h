#ifndef INCLUDED_SUDOKU_SOLVER_INTERFACE_H
#define INCLUDED_SUDOKU_SOLVER_INTERFACE_H

#include <vector>
#include <sudoku/metrics.h>
#include <sudoku/types.h>

namespace sudoku
{
    class SolverInterface
    {
        public:
            virtual ~SolverInterface() {}
            virtual auto computeNextSolution() -> bool = 0;
            virtual auto getCellValues() const -> const std::vector<CellValue>& = 0;
            virtual auto getMetrics() const -> Metrics = 0;
    };
}

#endif
