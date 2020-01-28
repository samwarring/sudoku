#include <algorithm>
#include <cassert>
#include <sudoku/cuda/error.h>
#include <sudoku/cuda/kernel.h>
#include <sudoku/cuda/dimensions.h>
#include <sudoku/cuda/grid.h>

namespace sudoku
{
    namespace cuda
    {
        __global__ static void kernel(KernelParams params)
        {
            size_t threadNum = threadIdx.x;
            Dimensions dims(params);
            Grid grid(dims, params, threadNum);
            Result* result = params.results + threadNum;
            *result = Result::OK_TIMED_OUT;
        }

        void kernelWrapper(unsigned blockCount, unsigned threadsPerBlock, KernelParams params)
        {
            kernel<<<blockCount, threadsPerBlock>>>(params);
            ErrorCheck() << cudaGetLastError();
        }

        DimensionParams::DimensionParams(const sudoku::Dimensions& dims)
        {
            cellCount = dims.getCellCount();
            maxCellValue = dims.getMaxCellValue();
            groupCount = dims.getNumGroups();

            // Concatenate groups.
            for (size_t gn = 0; gn < groupCount; ++gn) {
                groupOffsets.push_back(groupValues.size());
                for (auto cellPos : dims.getCellsInGroup(gn)) {
                    groupValues.push_back(cellPos);
                }
            }
            groupOffsets.push_back(groupValues.size());

            // Concatenate groups for each cell.
            for (size_t cp = 0; cp < cellCount; ++cp) {
                groupsForCellOffsets.push_back(groupsForCellValues.size());
                for (auto gn : dims.getGroupsForCell(cp)) {
                    groupsForCellValues.push_back(gn);
                }
            }
            groupsForCellOffsets.push_back(groupsForCellValues.size());
        }

        GridParams::GridParams(const std::vector<sudoku::Grid> grids)
        {
            assert(grids.size() > 0);

            // Concatenate cell values
            for (const auto& grid : grids) {
                cellValues.insert(
                    cellValues.end(),
                    grid.getCellValues().cbegin(),
                    grid.getCellValues().cend()
                );
            }

            // Concatenate restrictions
            for (const auto& grid : grids) {
                restrictionsOffsets.push_back(restrictions.size());
                for (auto restr : grid.getRestrictions()) {
                    restrictions.push_back(restr.first);
                    restrictions.push_back(restr.second);
                }
            }
            restrictionsOffsets.push_back(restrictions.size());

            // Allocate space for block counts (initialize to 0)
            const auto& dims = grids[0].getDimensions();
            blockCounts.resize(grids.size() * dims.getCellCount() * (1 + dims.getMaxCellValue()));
        }

        DeviceKernelParams::DeviceKernelParams(DimensionParams dimParams, GridParams gridParams, size_t threadCount)
            : groupValues_(dimParams.groupValues)
            , groupOffsets_(dimParams.groupOffsets)
            , groupsForCellValues_(dimParams.groupsForCellValues)
            , groupsForCellOffsets_(dimParams.groupsForCellOffsets)
            , cellValues_(gridParams.cellValues)
            , restrictions_(gridParams.restrictions)
            , restrictionsOffsets_(gridParams.restrictionsOffsets)
            , blockCounts_(gridParams.blockCounts)
            , results_(threadCount)
        {
            kernelParams_.cellCount = dimParams.cellCount;
            kernelParams_.maxCellValue = dimParams.maxCellValue;
            kernelParams_.groupCount = dimParams.groupCount;
            kernelParams_.groupValues = groupValues_.begin();
            kernelParams_.groupOffsets = groupOffsets_.begin();
            kernelParams_.groupsForCellValues = groupsForCellValues_.begin();
            kernelParams_.groupsForCellOffsets = groupsForCellOffsets_.begin();
            kernelParams_.cellValues = cellValues_.begin();
            kernelParams_.restrictions = restrictions_.begin();
            kernelParams_.restrictionsOffsets = restrictionsOffsets_.begin();
            kernelParams_.blockCounts = blockCounts_.begin();
            kernelParams_.results = results_.getDeviceData();
            std::fill(results_.getHostData(), results_.getHostData() + results_.getItemCount(),
                      Result::ERROR_NOT_SET);
            results_.copyToDevice();
        }

        Result DeviceKernelParams::getThreadResult(size_t threadNum)
        {
            results_.copyToHost();
            return results_.getHostData()[threadNum];
        }
    }
}
