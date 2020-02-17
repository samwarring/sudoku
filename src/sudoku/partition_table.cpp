#include <algorithm>
#include <sudoku/partition_table.h>

namespace sudoku
{
    PartitionTable::PartitionTable(const Dimensions& dims, 
                                   PartitionCount partitionCount,
                                   const std::vector<PartitionCount>& partitionIds)
                                   : dims_(&dims)
                                   , partitionCount_(partitionCount)
                                   , partitions_(computePartitions(partitionIds))
                                   , partitionSizes_(computePartitionSizes())
                                   , maxPartitionSize_(computeMaxPartitionSize())
                                   , combinedPartitions_(computeCombinedPartitions())
                                   , combinedPartitionIndex_(computeCombinedPartitionIndex())
                                   , relatedIndices_(computeRelatedIndices())
    {
        // partitions_ only necessary to initialize other members. No longer needed.
        std::vector<std::vector<CellCount>> emptyVector;
        partitions_.swap(emptyVector);
    }

    PartitionCount PartitionTable::getPartitionId(CellCount cellPos) const
    {
        return combinedPartitionIndex_[cellPos] / maxPartitionSize_;
    }

    CellCount PartitionTable::getPartitionIndex(CellCount cellPos) const
    {
        return combinedPartitionIndex_[cellPos] % maxPartitionSize_;
    }

    CellCount PartitionTable::getPartitionSize(PartitionCount partitionId) const
    {
        return partitionSizes_[partitionId];
    }

    CellCount PartitionTable::getCellPosition(PartitionCount partitionId, CellCount partitionIndex) const
    {
        return combinedPartitions_[(partitionId * maxPartitionSize_) + partitionIndex];
    }

    const std::vector<CellCount>& PartitionTable::getRelatedIndicesForPartition(PartitionCount partitionId,
                                                                                CellCount cellPos) const
    {
        auto offset = getRelatedIndicesOffset(partitionId, cellPos);
        return relatedIndices_[offset];
    }

    size_t PartitionTable::getRelatedIndicesOffset(PartitionCount partitionId, CellCount cellPos) const
    {
        return (dims_->getCellCount() * partitionId) + cellPos;
    }

    std::vector<std::vector<CellCount>> PartitionTable::computePartitions(const std::vector<PartitionCount>& partitionIds)
    {
        // Separate partitions into one vector for each partition.
        std::vector<std::vector<CellCount>> partitions(partitionCount_);
        for (CellCount cellPos = 0; cellPos < dims_->getCellCount(); ++cellPos) {
            partitions[partitionIds[cellPos]].push_back(cellPos);
        }
        return partitions;
    }

    std::vector<CellCount> PartitionTable::computePartitionSizes()
    {
        std::vector<CellCount> result(partitionCount_);
        std::transform(partitions_.begin(), partitions_.end(), result.begin(),
                       [](auto& partition){ return partition.size(); });
        return result;
    }

    CellCount PartitionTable::computeMaxPartitionSize()
    {
        return *std::max_element(partitionSizes_.begin(), partitionSizes_.end());
    }

    std::vector<CellCount> PartitionTable::computeCombinedPartitions()
    {
        // Allocate the combined partitions. Unused values are set to cellCount.
        std::vector<CellCount> result(maxPartitionSize_ * partitionCount_, dims_->getCellCount());

        // Copy each partition aligned to the maxPartitionSize
        for (PartitionCount partitionId = 0; partitionId < partitionCount_; ++partitionId) {
            std::copy(partitions_[partitionId].begin(), partitions_[partitionId].end(),
                      result.data() + (partitionId * maxPartitionSize_));
        }

        return result;
    }

    std::vector<CellCount> PartitionTable::computeCombinedPartitionIndex()
    {
        std::vector<CellCount> result(dims_->getCellCount());

        for (PartitionCount partitionId = 0; partitionId < partitionCount_; ++partitionId) {
            size_t combinedPartitionsOffset = partitionId * maxPartitionSize_;
            for (CellCount partitionIndex = 0; partitionIndex < partitions_[partitionId].size(); ++partitionIndex) {
                CellCount cellPos = combinedPartitions_[combinedPartitionsOffset + partitionIndex];
                result[cellPos] = combinedPartitionsOffset + partitionIndex;
            }
        }

        return result;
    }

    std::vector<std::vector<CellCount>> PartitionTable::computeRelatedIndices()
    {
        std::vector<std::vector<CellCount>> result(partitionCount_ * dims_->getCellCount());

        for (CellCount cellPos = 0; cellPos < dims_->getCellCount(); ++cellPos) {
            for (auto relatedPos : dims_->getRelatedCells(cellPos)) {
                auto partitionId = getPartitionId(relatedPos);
                auto partitionIndex = getPartitionIndex(relatedPos);
                auto relatedIndicesOffset = getRelatedIndicesOffset(partitionId, cellPos);
                result[relatedIndicesOffset].push_back(partitionIndex);
            }
        }

        return result;
    }
}
