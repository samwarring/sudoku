#ifndef INCLUDED_SUDOKU_FORK_H
#define INCLUDED_SUDOKU_FORK_H

#include <vector>
#include <sudoku/grid.h>

namespace sudoku
{
    /**
     * Fork a grid into multiple 'peer' grids. Each peer can be solved
     * independently to yield a solution to the original grid.
     * 
     * \param grid the original grid to fork.
     * 
     * \param peerCount the number of peer grids to create (at most).
     * 
     * \return a vector of grids. The vector contains at least one grid
     *         and at most 'peerCount' grids.
     */
    std::vector<Grid> fork(Grid grid, size_t peerCount);
}

#endif
