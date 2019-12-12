#ifndef INCLUDED_SUDOKU_SOLVER_H
#define INCLUDED_SUDOKU_SOLVER_H

#include <vector>
#include <stack>
#include <stdexcept>
#include <sudoku/dimensions.h>
#include <sudoku/potential.h>

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
            const std::vector<size_t>& getCellValues() const;

            /**
             * Get the number of total guesses made so far.
             */
            size_t getTotalGuesses() const { return totalGuesses_; }

        private:
            void initializeCellPotentials();

            void validateCellValues() const;

            void pushGuess(size_t cellPos, size_t cellValue);

            std::pair<size_t, size_t> popGuess();

            size_t selectNextCell() const;

            bool sequentialSolve();

            const Dimensions& dims_;
            std::vector<size_t> cellValues_;
            std::stack<std::pair<size_t, size_t>> guesses_;  ///< pairs of (position, value)
            std::vector<Potential> cellPotentials_;
            size_t totalGuesses_ = 0;  ///< Number of guesses made. Does not decrease.
    };
}

#endif 