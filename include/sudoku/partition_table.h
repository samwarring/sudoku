#ifndef INCLUDED_SUDOKU_PARTITION_TABLE_H
#define INCLUDED_SUDOKU_PARTITION_TABLE_H

#include <sudoku/dimensions.h>
#include <sudoku/partitions.h>

namespace sudoku
{
    /**
     * Translates global cell positions to per-partition cell positions and vice-versa.
     */
    class PartitionTable
    {
        public:
            /**
             * Computes a new partition table.
             * 
             * \param dims Dimensions for computing related cell indices.
             * \param partitionCount Max value occurring in partitionIds
             * \param partitionIds A vector where each value denotes a partitionId.
             *                     Consider using a vector computed by one of the
             *                     functions in \ref sudoku/partitions.h.
             */
            PartitionTable(const Dimensions& dims, 
                           PartitionCount partitionCount,
                           const std::vector<PartitionCount>& partitionIds);

            /**
             * Gets the partitionId covering the cell position.
             */
            PartitionCount getPartitionId(CellCount cellPos) const;

            /**
             * Gets unique index for the cell among others with the same partitionId;
             * also referred to as the "per-partition" cell position.
             */
            CellCount getPartitionIndex(CellCount cellPos) const;

            /**
             * Gets the number of cells covered by the partitionId.
             */
            CellCount getPartitionSize(PartitionCount partitionId) const;

            /**
             * Gets the global cell position given the partitionId and partitionIndex.
             */
            CellCount getCellPosition(PartitionCount partitionId, CellCount partitionIndex) const;

            /**
             * Gets a vector of per-partition cell positions related to the global
             * cell position. The returned positions do not cover ALL of cellPos's
             * related positions - only those covered by partitionId.
             */
            const std::vector<CellCount>& getRelatedIndicesForPartition(PartitionCount partitionId, CellCount cellPos) const;

        private:
            const Dimensions* dims_;
            PartitionCount partitionCount_;
            std::vector<std::vector<CellCount>> partitions_;
            std::vector<CellCount> partitionSizes_;
            CellCount maxPartitionSize_;
            std::vector<CellCount> combinedPartitions_;
            std::vector<CellCount> combinedPartitionIndex_;
            std::vector<std::vector<CellCount>> relatedIndices_;

            /**
             * Helpers for initializing private members
             */

            std::vector<std::vector<CellCount>> computePartitions(const std::vector<PartitionCount>& partitionIds);

            std::vector<CellCount> computePartitionSizes();

            CellCount computeMaxPartitionSize();

            std::vector<CellCount> computeCombinedPartitions();

            std::vector<CellCount> computeCombinedPartitionIndex();

            std::vector<std::vector<CellCount>> computeRelatedIndices();

            /**
             * Selects the correct vector of related partitionIndices.
             * Use the returned offset to access relatedIndices_.
             */
            size_t getRelatedIndicesOffset(PartitionCount partitionId, CellCount cellPos) const;
    };

}

#endif