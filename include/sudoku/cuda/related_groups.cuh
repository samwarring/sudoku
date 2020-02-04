#ifndef INCLUDED_SUDOKU_CUDA_RELATED_GROUPS_CUH
#define INCLUDED_SUDOKU_CUDA_RELATED_GROUPS_CUH

#include <algorithm>
#include <sudoku/cuda/types.h>
#include <sudoku/dimensions.h>

namespace sudoku
{
    namespace cuda
    {
        template <unsigned MAX_GROUPS_FOR_CELL>
        class RelatedGroups
        {
        private:
            CellCount   cellCount_;
            GroupCount  totalGroupCount_;
            GroupCount  groupCount_;
            GroupCount  groupIds_[MAX_GROUPS_FOR_CELL];
            GroupCount* sharedGroupUpdates_;

            __device__ unsigned getGlobalGroupIdOffset(GroupCount groupIter) const
            {
                return (groupIter * cellCount_) + threadIdx.x;
            }

        public:
            static unsigned getGlobalGroupIdOffset(CellCount cellCount, CellCount cellPos, GroupCount groupIter)
            {
                return (groupIter * cellCount) + cellPos;
            }

            static std::vector<GroupCount> getGroupCounts(const sudoku::Dimensions& dims)
            {
                std::vector<GroupCount> result(dims.getCellCount());
                for (size_t cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
                    result[cellPos] = static_cast<GroupCount>(dims.getGroupsForCell(cellPos).size());
                }
                return result;
            }

            static std::vector<GroupCount> getGroupIds(const sudoku::Dimensions& dims)
            {
                auto groupCounts = getGroupCounts(dims);
                GroupCount maxGroupCount = *std::max_element(groupCounts.begin(), groupCounts.end());
                std::vector<GroupCount> result(dims.getCellCount() * maxGroupCount);
                for (size_t cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
                    const auto& groups = dims.getGroupsForCell(cellPos);
                    for (size_t groupIter = 0; groupIter < groups.size(); ++groupIter) {
                        auto offset = getGlobalGroupIdOffset(
                            static_cast<CellCount>(dims.getCellCount()),
                            static_cast<CellCount>(cellPos),
                            static_cast<GroupCount>(groupIter));
                        result[offset] = static_cast<GroupCount>(groups[groupIter]);
                    }
                }
                return result;
            }

            __device__ RelatedGroups(CellCount cellCount, GroupCount totalGroupCount,
                                     GroupCount* globalGroupCounts, GroupCount* globalGroupIds,
                                     GroupCount* sharedGroupUpdates)
                : cellCount_(cellCount)
                , totalGroupCount_(totalGroupCount)
                , groupCount_(0)
                , sharedGroupUpdates_(sharedGroupUpdates)
            {
                if (threadIdx.x < cellCount) {
                    groupCount_ = globalGroupCounts[threadIdx.x];
                    for (GroupCount groupIter = 0; groupIter < groupCount_; ++groupIter) {
                        auto offset = getGlobalGroupIdOffset(groupIter);
                        groupIds_[groupIter] = globalGroupIds[offset];
                    }
                }
            }

            __device__ GroupCount getGroupCount() const
            {
                return groupCount_;
            }

            /// Thread for given cell position writes the cell value to the shared
            /// group-update buffer for each related group.
            __device__ void broadcast(CellCount cellPos, CellValue cellValue)
            {
                // Clear all previous broadcasts (if any).
                if (threadIdx.x < totalGroupCount_) {
                    sharedGroupUpdates_[threadIdx.x] = 0;
                }
                __syncthreads();

                // Only requested cell writes cell value update to its groups.
                if (threadIdx.x == cellPos) {
                    for (GroupCount groupIter = 0; groupIter < groupCount_; ++groupIter) {
                        sharedGroupUpdates_[groupIds_[groupIter]] = cellValue;
                    }
                }
                __syncthreads();
            }

            /// Get the broadcasted value for my group at the given position.
            /// Call this method once per related group when blocking/unblocking values.
            __device__ CellValue getBroadcast(GroupCount groupIter)
            {
                return groupIter < groupCount_ ? sharedGroupUpdates_[groupIds_[groupIter]] : 0;
            }
        };
    }
}

#endif
