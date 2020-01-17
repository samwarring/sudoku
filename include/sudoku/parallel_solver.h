#ifndef INCLUDED_SUDOKU_PARALLEL_SOLVER_H
#define INCLUDED_SUDOKU_PARALLEL_SOLVER_H

#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/solver.h>
#include <sudoku/solution_queue.h>

namespace sudoku
{
    class ParallelSolverException : public std::logic_error { using std::logic_error::logic_error; };

    class ParallelSolver
    {
        public:
            ParallelSolver(
                const Dimensions& dims,
                std::vector<size_t> cellValues,
                size_t threadCount,
                size_t queueSize
            );

            ~ParallelSolver();

            bool computeNextSolution();

            const std::vector<size_t>& getCellValues() const { return cellValues_; }

        private:
            void startThreads();

            const Dimensions& dims_;
            std::vector<size_t> cellValues_;
            std::vector<std::thread> threads_;
            std::vector<std::unique_ptr<Solver>> solvers_;
            SolutionQueue queue_;
            std::unique_ptr<SolutionQueue::Consumer> consumer_;
            size_t threadCount_;
    };
}

#endif
