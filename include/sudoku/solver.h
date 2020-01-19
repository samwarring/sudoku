#ifndef INCLUDED_SUDOKU_SOLVER_H
#define INCLUDED_SUDOKU_SOLVER_H

#include <atomic>
#include <chrono>
#include <memory>
#include <stack>
#include <stdexcept>
#include <vector>
#include <sudoku/dimensions.h>
#include <sudoku/grid.h>
#include <sudoku/metrics.h>

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
    class Solver
    {
        public:

            /**
             * \param dims dimensions of the sudoku
             * \param cellValues initial values of the sudoku
             * \throw SolverException if a standard sudoku had two 1's in the
             *        first row, or if a cell value exceeds the max cell value.
             */
            Solver(const Dimensions& dims, std::vector<size_t> cellValues);

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
            const std::vector<size_t>& getCellValues() const { return cellValues_; }

            /**
             * Fork the solver into multiple solver objects, each searching
             * a non-overlapping portion of the solution space.
             * 
             * \param numPeers Number of additional objects to produce (at most).
             * 
             * \return a vector of additional Solver objects. The current
             *         object maintains a unique portion of the solution-space
             *         not covered by the returned objects.
             * 
             * \note If the sudoku was obviously solvable/unsolvable, fork() will
             *       return an empty vector. In this case, caller should check if
             *       the sudoku was solved, or found to be no-solution using the
             *       \ref computeNextSolution() method.
             * 
             * \note The resulting vector will never contain more than maxCellCount
             *       solvers. This can be improved in a later version.
             */
            std::vector<std::unique_ptr<Solver>> fork(size_t numPeers);

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
            void initializeCellPotentials();

            void validateCellValues() const;

            void pushGuess(size_t cellPos, size_t cellValue);

            std::pair<size_t, size_t> popGuess();

            size_t selectNextCell() const;

            bool sequentialSolve();

            size_t selectForkCell();

            std::vector<std::unique_ptr<Solver>> forkOneValuePerPeer(
                size_t forkPos,
                const std::vector<size_t>& availableValues
            );

            std::vector<std::unique_ptr<Solver>> forkManyValuesPerPeer(
                size_t forkPos,
                const std::vector<size_t>& availableValues,
                size_t numPeers
            );

            std::vector<std::unique_ptr<Solver>> forkMorePeersThanValues(
                size_t forkPos,
                const std::vector<size_t>& availableValues,
                size_t numPeers
            );

            const Dimensions& dims_;
            std::vector<size_t> cellValues_;
            std::stack<std::pair<size_t, size_t>> guesses_;  ///< pairs of (position, value)
            Grid grid_;
            std::atomic<bool> haltEvent_;
            Metrics metrics_;

            // Set this flag to true if the solver contains a solution not yet reported
            // by computeNextSolution. E.g. the solver was initialized with an already-
            // solved sudoku; or a solution was reached in a call to fork.
            bool unreportedSolution_ = false;
    };
}

#endif 