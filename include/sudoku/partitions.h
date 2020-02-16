#ifndef INCLUDED_SUDOKU_PARTITIONS_H
#define INCLUDED_SUDOKU_PARTITIONS_H

#include <vector>
#include <sudoku/types.h>

/**
 * \file partitions.h
 * 
 * \brief Delcares algorithms to divide a grid into N partitions.
 * 
 * Each sudoku::partitionXxx function returns an array with cellCount
 * elements. The value of each element denotes the assigned partition
 * for that cell.
 */ 

namespace sudoku
{
    using PartitionCount = CellCount;

    std::vector<PartitionCount> partitionRoundRobin(CellCount cellCount, PartitionCount partitionCount);

    std::vector<PartitionCount> partitionRoundRobinRotate(CellCount cellCount, PartitionCount partitionCount);

    std::vector<PartitionCount> partitionRandom(CellCount cellCount, PartitionCount partitionCount);

    std::vector<PartitionCount> partitionDiagonal(CellCount width, CellCount height, PartitionCount partitionCount);

    std::vector<PartitionCount> partitionRRTrail(CellCount cellCount, PartitionCount partitionCount);
}

#endif
