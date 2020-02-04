#include <vector>
#include <sudoku/cuda/block_counter.cuh>
#include <sudoku/cuda/error_check.h>
#include <sudoku/cuda/util.h>
#include "block_counter_kernels.h"

using namespace sudoku::cuda;

using TestBlockCounter = BlockCounter<100>;

static __device__ TestBlockCounter construct(BlockCounterKernelArgs args)
{
    return {args.cellCount, args.maxCellValue, args.cellBlockCounts, args.valueBlockCounts};
}

__global__ void blockCounterBlockKernel(BlockCounterKernelArgs args, CellCount blockPos, CellValue blockValue)
{
    auto blockCounter = construct(args);
    if (threadIdx.x == blockPos) {
        blockCounter.block(blockValue);
    }
}

__global__ void blockCounterUnblockKernel(BlockCounterKernelArgs args, CellCount blockPos, CellValue blockValue)
{
    auto blockCounter = construct(args);
    if (threadIdx.x == blockPos) {
        blockCounter.unblock(blockValue);
    }
}

__global__ void blockCounterMarkOccupiedKernel(BlockCounterKernelArgs args, CellCount occupiedPos)
{
    auto blockCounter = construct(args);
    if (threadIdx.x == occupiedPos) {
        blockCounter.markOccupied();
    }
}

__global__ void blockCounterMarkFreeKernel(BlockCounterKernelArgs args, CellCount freePos)
{
    auto blockCounter = construct(args);
    if (threadIdx.x == freePos) {
        blockCounter.markFree();
    }
}

__global__ void blockCounterGetMaxCellBlockCountPairKernel(BlockCounterKernelArgs args, BlockCounterKernels::Pair* out)
{
    extern __shared__ TestBlockCounter::Pair sharedBuffer[];
    auto blockCounter = construct(args);
    TestBlockCounter::Pair pair = blockCounter.getMaxCellBlockCountPair(sharedBuffer);
    BlockCounterKernels::Pair outPair{ pair.cellPos, pair.cellBlockCount };
    *out = outPair;
}

BlockCounterKernels::BlockCounterKernels(sudoku::cuda::CellCount cellCount, sudoku::cuda::CellValue maxCellValue)
    : cellCountPow2_(nearestPowerOf2(cellCount))
    , hostCellBlockCounts_(cellCount, 0)
    , hostValueBlockCounts_(cellCount * maxCellValue, 0)
    , deviceCellBlockCounts_(hostCellBlockCounts_)
    , deviceValueBlockCounts_(hostValueBlockCounts_)
{
    args_.cellCount = cellCount;
    args_.maxCellValue = maxCellValue;
    args_.cellBlockCounts = deviceCellBlockCounts_.get();
    args_.valueBlockCounts = deviceValueBlockCounts_.get();
}

void BlockCounterKernels::copyToHost()
{
    hostCellBlockCounts_ = deviceCellBlockCounts_.copyToHost();
    hostValueBlockCounts_ = deviceValueBlockCounts_.copyToHost();
}

CellBlockCount BlockCounterKernels::getCellBlockCount(CellCount cellPos) const
{
    return hostCellBlockCounts_[cellPos];
}

ValueBlockCount BlockCounterKernels::getValueBlockCount(CellCount cellPos, CellValue cellValue) const
{
    auto offset = TestBlockCounter::getValueBlockCountOffset(args_.cellCount, cellPos, cellValue);
    return hostValueBlockCounts_[offset];
}

void BlockCounterKernels::block(CellCount blockPos, CellValue blockValue)
{
    blockCounterBlockKernel<<<1, cellCountPow2_>>>(args_, blockPos, blockValue);
    ErrorCheck::lastError();
    copyToHost();
}

void BlockCounterKernels::unblock(CellCount unblockPos, CellValue unblockValue)
{
    blockCounterUnblockKernel<<<1, cellCountPow2_>>>(args_, unblockPos, unblockValue);
    ErrorCheck::lastError();
    copyToHost();
}

void BlockCounterKernels::markOccupied(CellCount occupiedPos)
{
    blockCounterMarkOccupiedKernel<<<1, cellCountPow2_>>>(args_, occupiedPos);
    ErrorCheck::lastError();
    copyToHost();
}

void BlockCounterKernels::markFree(CellCount freePos)
{
    blockCounterMarkFreeKernel<<<1, cellCountPow2_>>>(args_, freePos);
    ErrorCheck::lastError();
    copyToHost();
}

BlockCounterKernels::Pair BlockCounterKernels::getMaxBlockCountPair()
{
    unsigned sharedMemSize = sizeof(TestBlockCounter::Pair) * cellCountPow2_;
    sudoku::cuda::DeviceBuffer<Pair> devicePair(1);
    blockCounterGetMaxCellBlockCountPairKernel<<<1, cellCountPow2_, sharedMemSize>>>(args_, devicePair.get());
    ErrorCheck::lastError();
    std::vector<Pair> hostPair = devicePair.copyToHost();
    return hostPair[0];
}
