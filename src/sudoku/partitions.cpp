#include <cstdlib>
#include <sudoku/partitions.h>

namespace sudoku
{
    std::vector<PartitionCount> partitionRoundRobin(CellCount cellCount, PartitionCount partitionCount)
    {
        std::vector<PartitionCount> result(cellCount);
        for (CellCount cellPos = 0; cellPos < cellCount; ++cellPos) {
            result[cellPos] = cellPos % partitionCount;
        }
        return result;
    }

    std::vector<PartitionCount> partitionRoundRobinRotate(CellCount cellCount, PartitionCount partitionCount)
    {
        std::vector<PartitionCount> result(cellCount);
        PartitionCount rotation = 0;
        for (CellCount cellPos = 0; cellPos < cellCount; ++cellPos) {
            result[cellPos] = (cellPos + rotation) % partitionCount;
            if (cellPos % partitionCount == (partitionCount - 1)) {
                rotation = (rotation + 1) % partitionCount;
            }
        }
        return result;
    }

    std::vector<PartitionCount> partitionRandom(CellCount cellCount, PartitionCount partitionCount)
    {
        std::vector<PartitionCount> result(cellCount);
        srand(420);
        for (CellCount cellPos = 0; cellPos < cellCount; ++cellPos) {
            result[cellPos] = rand() % partitionCount;
        }
        return result;
    }

    std::vector<PartitionCount> partitionDiagonal(CellCount width, CellCount height, PartitionCount partitionCount)
    {
        // Example order for 5x3 rectangle (.'s are ignored)
        //     0 2 5 8 b . .
        //   . 1 4 7 a d .
        // . . 3 6 9 c e
        std::vector<PartitionCount> result(width * height);
        int startCol = (int)height * -1;
        PartitionCount curPartition = 0;
        for (auto startCol = (int)(height - 1) * -1; startCol < width; ++startCol) {
            for (CellCount step = 0; step < height; ++step) {
                auto row = height - step - 1;
                auto col = startCol + (int)step;
                if (col < 0) continue;
                CellCount cellPos = (width * row) + col;
                result[cellPos] = curPartition;
                curPartition = (curPartition + 1) % partitionCount;
            }
        }
        return result;
    }

    std::vector<PartitionCount> partitionRRTrail(CellCount cellCount, PartitionCount partitionCount)
    {
        std::vector<PartitionCount> result(cellCount);
        PartitionCount curPartition = 0;
        PartitionCount trailer = 0;
        for (CellCount cellPos = 0; cellPos < cellCount; ++cellPos) {
            if (curPartition < partitionCount) {
                result[cellPos] = curPartition;
                curPartition++;
            }
            else {
                result[cellPos] = trailer;
                trailer = (trailer + 1) % partitionCount;
                curPartition = 0;
            }
        }
        return result;
    }
}
