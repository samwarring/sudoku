#include <stdexcept>
#include <vector>
#include <sudoku/cuda/types.h>
#include <sudoku/cuda/block_counter.cuh>
#include <sudoku/cuda/grid.cuh>
#include <sudoku/cuda/related_groups.cuh>
#include <sudoku/cuda/util.h>
#include <sudoku/grid.h>
#include "block_counter_kernels.h"
#include "grid_kernels.h"
#include "related_groups_kernels.h"

#define MAX_CELL_VALUE 9
#define MAX_GROUPS_FOR_CELL 3

using namespace sudoku::cuda;

using TestRelatedGroups = RelatedGroups<MAX_GROUPS_FOR_CELL>;
using TestBlockCounter = BlockCounter<MAX_CELL_VALUE>;
using TestGrid = Grid<MAX_CELL_VALUE, MAX_GROUPS_FOR_CELL>;

struct GridKernelObjects
{
    TestRelatedGroups relatedGroups;
    TestBlockCounter blockCounter;
    TestGrid grid;

    __device__ GridKernelObjects(RelatedGroupsKernelArgs relatedGroupsArgs,
                                 BlockCounterKernelArgs blockCounterArgs,
                                 GridKernelArgs gridArgs,
                                 CellValue* sharedGroupUpdates,
                                 TestBlockCounter::Pair* sharedBlockCountReductionBuffer)
        : relatedGroups(relatedGroupsArgs.cellCount,
                        relatedGroupsArgs.totalGroupCount,
                        relatedGroupsArgs.groupCounts,
                        relatedGroupsArgs.groupIds,
                        sharedGroupUpdates)
        , blockCounter(blockCounterArgs.cellCount,
                       blockCounterArgs.maxCellValue,
                       blockCounterArgs.cellBlockCounts,
                       blockCounterArgs.valueBlockCounts,
                       sharedBlockCountReductionBuffer)
        , grid(relatedGroups,
               blockCounter,
               gridArgs.cellValues,
               gridArgs.cellCount)
    {}
};

__global__ void gridInitBlockCountsKernel(RelatedGroupsKernelArgs relatedGroupsArgs,
                                          BlockCounterKernelArgs blockCounterArgs,
                                          GridKernelArgs gridArgs)
{
    extern __shared__ CellValue sharedGroupUpdates[];
    GridKernelObjects objs(relatedGroupsArgs, blockCounterArgs, gridArgs, sharedGroupUpdates, nullptr);
    objs.grid.initBlockCounts();
}

__global__ void gridSetCellValueKernel(RelatedGroupsKernelArgs relatedGroupsArgs,
                                       BlockCounterKernelArgs blockCounterArgs,
                                       GridKernelArgs gridArgs,
                                       CellCount cellPos, CellValue cellValue)
{
    extern __shared__ CellValue sharedGroupUpdates[];
    GridKernelObjects objs(relatedGroupsArgs, blockCounterArgs, gridArgs, sharedGroupUpdates, nullptr);
    objs.grid.setCellValue(cellPos, cellValue);
}

__global__ void gridClearCellValueKernel(RelatedGroupsKernelArgs relatedGroupsArgs,
                                         BlockCounterKernelArgs blockCounterArgs,
                                         GridKernelArgs gridArgs, CellCount cellPos)
{
    extern __shared__ CellValue sharedGroupUpdates[];
    GridKernelObjects objs(relatedGroupsArgs, blockCounterArgs, gridArgs, sharedGroupUpdates, nullptr);
    objs.grid.clearCellValue(cellPos);
}

__global__ void gridGetMaxCellBlockCountPosKernel(RelatedGroupsKernelArgs relatedGroupsArgs,
                                                 BlockCounterKernelArgs blockCounterArgs,
                                                 GridKernelArgs gridArgs, CellCount* outPos)
{
    extern __shared__ TestBlockCounter::Pair sharedBlockCounterPairs[];
    GridKernelObjects objs(relatedGroupsArgs, blockCounterArgs, gridArgs, nullptr, sharedBlockCounterPairs);
    auto pair = objs.blockCounter.getMaxCellBlockCountPair();
    if (threadIdx.x == 0) {
        *outPos = pair.cellPos;
    }
}

static std::vector<CellValue> castCellValues(const std::vector<size_t> cellValues)
{
    std::vector<CellValue> result(cellValues.size());
    for (size_t i = 0; i < cellValues.size(); ++i) {
        result[i] = static_cast<CellValue>(cellValues[i]);
    }
    return result;
}

GridKernels::GridKernels(const sudoku::Grid& grid)
    : dims_(grid.getDimensions())
    , hostCellValues_(castCellValues(grid.getCellValues()))
    , groupCounts_(TestRelatedGroups::getGroupCounts(dims_))
    , groupIds_(TestRelatedGroups::getGroupIds(dims_))
    , deviceCellBlockCounts_(std::vector<CellBlockCount>(dims_.getCellCount(), 0))
    , deviceValueBlockCounts_(std::vector<ValueBlockCount>(dims_.getCellCount() * dims_.getMaxCellValue(), 0))
    , deviceCellValues_(hostCellValues_)
{
    if (dims_.getMaxCellValue() > MAX_CELL_VALUE) {
        throw std::runtime_error("Max cell value too large for kernel");
    }
    if (dims_.getMaxGroupsForCellCount() > MAX_GROUPS_FOR_CELL) {
        throw std::runtime_error("Max groups for cell too large for kernel");
    }

    CellCount cellCount = static_cast<CellCount>(dims_.getCellCount());
    GroupCount totalGroupCount = static_cast<CellCount>(dims_.getNumGroups());
    CellValue maxCellValue = static_cast<CellValue>(dims_.getMaxCellValue());
    cellCountPow2_ = nearestPowerOf2(cellCount);
    sharedGroupUpdatesSize_ = sizeof(CellValue) * dims_.getNumGroups();

    relatedGroupsArgs_.cellCount = cellCount;
    relatedGroupsArgs_.totalGroupCount = totalGroupCount;
    relatedGroupsArgs_.groupCounts = groupCounts_.get();
    relatedGroupsArgs_.groupIds = groupIds_.get();

    blockCounterArgs_.cellCount = cellCount;
    blockCounterArgs_.maxCellValue = maxCellValue;
    blockCounterArgs_.cellBlockCounts = deviceCellBlockCounts_.get();
    blockCounterArgs_.valueBlockCounts = deviceValueBlockCounts_.get();

    gridArgs_.cellValues = deviceCellValues_.get();
    gridArgs_.cellCount = cellCount;
}

void GridKernels::copyToHost()
{
    hostCellValues_ = deviceCellValues_.copyToHost();
    hostCellBlockCounts_ = deviceCellBlockCounts_.copyToHost();
    hostValueBlockCounts_ = deviceValueBlockCounts_.copyToHost();
}

void GridKernels::initBlockCounts()
{
    gridInitBlockCountsKernel<<<1, cellCountPow2_, sharedGroupUpdatesSize_>>>(
        relatedGroupsArgs_, blockCounterArgs_, gridArgs_
    );
    ErrorCheck::lastError();
    copyToHost();
}

void GridKernels::setCellValue(CellCount cellPos, CellValue cellValue)
{
    gridSetCellValueKernel<<<1, cellCountPow2_, sharedGroupUpdatesSize_>>>(
        relatedGroupsArgs_, blockCounterArgs_, gridArgs_, cellPos, cellValue
    );
    ErrorCheck::lastError();
    copyToHost();
}

void GridKernels::clearCellValue(CellCount cellPos)
{
    gridClearCellValueKernel<<<1, cellCountPow2_, sharedGroupUpdatesSize_>>>(
        relatedGroupsArgs_, blockCounterArgs_, gridArgs_, cellPos
    );
    ErrorCheck::lastError();
    copyToHost();
}

CellCount GridKernels::getMaxCellBlockCountPos()
{
    size_t sharedBlockCounterPairsSize = sizeof(TestBlockCounter::Pair) * cellCountPow2_;
    DeviceBuffer<CellCount> outPos(1);
    gridGetMaxCellBlockCountPosKernel<<<1, cellCountPow2_, sharedBlockCounterPairsSize>>>(
        relatedGroupsArgs_, blockCounterArgs_, gridArgs_, outPos.get()
    );
    ErrorCheck::lastError();
    return outPos.copyToHost()[0];
}

CellValue GridKernels::getCellValue(CellCount cellPos) const
{
    return hostCellValues_[cellPos];
}

CellBlockCount GridKernels::getCellBlockCount(CellCount cellPos) const
{
    return hostCellBlockCounts_[cellPos];
}

ValueBlockCount GridKernels::getValueBlockCount(CellCount cellPos, CellValue cellValue) const
{
    auto offset = TestBlockCounter::getValueBlockCountOffset(blockCounterArgs_.cellCount, cellPos, cellValue);
    return hostValueBlockCounts_[offset];
}
