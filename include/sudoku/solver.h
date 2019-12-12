#ifndef INCLUDED_SUDOKU_SOLVER_H
#define INCLUDED_SUDOKU_SOLVER_H

#include <vector>
#include <stack>
#include <sudoku/dimensions.h>
#include <sudoku/potential.h>

namespace sudoku
{
    /**
     * \todo Document this class
     */
    class Solver
    {
        public:
            Solver(const Dimensions& dims, std::vector<size_t> cellValues);

            bool computeNextSolution();

            const std::vector<size_t>& getCellValues() const;

        private:
            void initializeCellPotentials();

            void pushGuess(size_t cellPos, size_t cellValue);

            std::pair<size_t, size_t> popGuess();

            size_t selectNextCell() const;

            bool recursiveSolve();

            bool sequentialSolve();

            const Dimensions& dims_;
            std::vector<size_t> cellValues_;
            std::stack<std::pair<size_t, size_t>> guesses_;  ///< pairs of (position, value)
            std::vector<Potential> cellPotentials_;
    };
}

#endif 