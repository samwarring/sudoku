#ifndef INCLUDED_SUDOKU_SOLVER_H
#define INCLUDED_SUDOKU_SOLVER_H

#include <atomic>
#include <chrono>
#include <stack>
#include <stdexcept>
#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/grid.h>
#include <sudoku/metrics.h>
#include <sudoku/types.h>
#include <sudoku/solver_interface.h>

namespace sudoku
{
    /**
     * Thrown when constructing invalid Solver objects.
     */
    class SolverException : public std::logic_error { using logic_error::logic_error; };

    /**
     * Solver objects solve sudokus. The initial cell values are fixed at
     * object construction and cannot be changed. After constructing, users
     * can iterate through all possible solutions of the sudoku for the given
     * initial values.
     */
    class Solver : public SolverInterface
    {
        public:

            /**
             * \param grid the initial grid to be solved.
             */
            Solver(Grid grid);

            /**
             * Compute the next solution for the sudoku.
             * 
             * \return true if a new solution could be found, or false
             *         if there are no more solutions.
             */
            bool computeNextSolution();

            /**
             * Read the current cell values of the sudoku. If \ref computeNextSolution
             * has not been called, this returns the initial cell values.
             */
            const std::vector<CellValue>& getCellValues() const { return grid_.getCellValues(); }

            /**
             * Halt any occurance of \ref computeNextSolution() that may be
             * consuming another thread.
             */
            void halt() { haltEvent_.store(true); }

            /**
             * Get the metrics collected so far while solving.
             */
            Metrics getMetrics() const { return metrics_; }

        private:
            void pushGuess(CellCount cellPos, CellValue cellValue);

            std::pair<CellCount, CellValue> popGuess();

            Grid grid_;
            Metrics metrics_;
            std::stack<CellCount> guesses_;  ///< stack of cell positions.
            std::atomic<bool> haltEvent_;

            // Set this flag to true if the solver contains a solution not yet reported
            // by computeNextSolution. E.g. the solver was initialized with an already-
            // solved sudoku.
            bool unreportedSolution_ = false;
    };
}

#endif