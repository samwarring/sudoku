#ifndef INCLUDED_SUDOKU_PARALLEL_SOLVER_H
#define INCLUDED_SUDOKU_PARALLEL_SOLVER_H

#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/grid.h>
#include <sudoku/metrics.h>
#include <sudoku/solver.h>
#include <sudoku/solution_queue.h>
#include <sudoku/types.h>
#include <sudoku/solver_interface.h>

namespace sudoku
{
    class ParallelSolverException : public std::logic_error { using std::logic_error::logic_error; };

    class ParallelSolver : public SolverInterface
    {
        public:
            ParallelSolver(
                sudoku::Grid grid,
                size_t threadCount,
                size_t queueSize
            );

            ~ParallelSolver();

            bool computeNextSolution();

            const std::vector<CellValue>& getCellValues() const;

            Metrics getMetrics() const { return metrics_; }

        private:
            void startThreads();

            sudoku::Grid grid_;
            std::vector<CellValue> solution_;
            std::vector<std::thread> threads_;
            std::vector<std::unique_ptr<Solver>> solvers_;
            SolutionQueue queue_;
            std::unique_ptr<SolutionQueue::Consumer> consumer_;
            size_t threadCount_;
            Metrics metrics_;
    };
}

#endif
