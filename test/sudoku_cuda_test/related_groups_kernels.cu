#include <sudoku/cuda/related_groups.cuh>
#include <sudoku/cuda/util.h>
#include "related_groups_kernels.h"

using namespace sudoku::cuda;

using TestRelatedGroups = RelatedGroups<3>;

__global__ void relatedGroupsBroadcastAndReceiveKernel(RelatedGroupsKernelArgs args,
                                                       CellCount cellPos,
                                                       CellValue cellValue,
                                                       CellValue* valuesReceived)
{
    extern __shared__ CellValue sharedGroupUpdates[];
    TestRelatedGroups relatedGroups(args.cellCount, args.totalGroupCount, args.groupCounts,
                                    args.groupIds, sharedGroupUpdates);
    relatedGroups.broadcast(cellPos, cellValue);
    CellValue valueReceived = 0;
    for (GroupCount groupIter = 0; groupIter < relatedGroups.getGroupCount(); ++groupIter) {
        CellValue curValueReceived = relatedGroups.getBroadcast(groupIter);
        valueReceived = curValueReceived == 0 ? valueReceived : curValueReceived;
    }
    if (threadIdx.x < args.cellCount) {
        valuesReceived[threadIdx.x] = valueReceived;
    }
}

RelatedGroupsKernels::RelatedGroupsKernels(const sudoku::Dimensions& dims)
    : groupCounts_(TestRelatedGroups::getGroupCounts(dims))
    , groupIds_(TestRelatedGroups::getGroupIds(dims))
{
    args_.cellCount = static_cast<CellCount>(dims.getCellCount());
    args_.totalGroupCount = static_cast<GroupCount>(dims.getNumGroups());
    args_.groupCounts = groupCounts_.get();
    args_.groupIds = groupIds_.get();
}

std::vector<CellValue> RelatedGroupsKernels::broadcastAndReceive(CellCount cellPos, CellValue cellValue)
{
    DeviceBuffer<CellValue> valuesReceived(args_.cellCount);
    size_t sharedMemSize = sizeof(CellValue) * args_.totalGroupCount;
    CellCount cellCountPow2 = sudoku::cuda::nearestPowerOf2(args_.cellCount);
    relatedGroupsBroadcastAndReceiveKernel<<<1, cellCountPow2, sharedMemSize>>>(
        args_, cellPos, cellValue, valuesReceived.get()
    );
    ErrorCheck::lastError();
    return valuesReceived.copyToHost();
}
