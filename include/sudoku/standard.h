#ifndef INCLUDED_SUDOKU_STANDARD_H
#define INCLUDED_SUDOKU_STANDARD_H

#include <vector>
#include <sudoku/dimensions.h>

/**
 * \file standard.h
 * 
 * This file defines specializations of \ref sudoku classes which aid in solving
 * standard 9x9 sudokus.
 */

namespace sudoku
{
    /**
     * Number of cells in a standard 9x9 sudoku
     */
    const size_t STANDARD_CELL_COUNT = 81;
    
    /**
     * Maximum cell value for a 9x9 sudoku.
     */
    const size_t STANDARD_MAX_CELL_VALUE = 9;

    /**
     * Cell groups for a 9x9 sudoku.
     */
    extern const std::vector<std::vector<size_t>> STANDARD_GROUPS;

    /**
     * Dimensions for solving 9x9 sudokus.
     */
    extern const Dimensions STANDARD_DIMENSIONS;
}

#endif
