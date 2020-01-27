#include <cassert>
#include <sudoku/cuda/error.h>
#include <sudoku/cuda/kernel.h>

namespace sudoku
{
    namespace cuda
    {
        namespace compute_next_solution_kernel
        {
            __global__ static void kernel(Params params)
            {
                params.results[threadIdx.x] = Result::OK_TIMED_OUT;
            }

            void kernelWrapper(unsigned blockCount, unsigned threadsPerBlock, Params params)
            {
                kernel<<<blockCount, threadsPerBlock>>>(params);
                ErrorCheck() << cudaGetLastError();
            }

            DimensionParams::DimensionParams(const sudoku::Dimensions& dims)
            {
                // Concatenate groups.
                for (size_t gn = 0; gn < dims.getNumGroups(); ++gn) {
                    groupOffsets.push_back(groupValues.size());
                    for (auto cellPos : dims.getCellsInGroup(gn)) {
                        groupValues.push_back(cellPos);
                    }
                }
                groupOffsets.push_back(groupValues.size());

                // Concatenate groups for each cell.
                for (size_t cp = 0; cp < dims.getCellCount(); ++cp) {
                    groupsForCellOffsets.push_back(groupsForCellValues.size());
                    for (auto gn : dims.getGroupsForCell(cp)) {
                        groupsForCellValues.push_back(gn);
                    }
                }
                groupsForCellOffsets.push_back(groupsForCellValues.size());
            }

            GridParams::GridParams(const std::vector<Grid> grids)
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
        }
    }
}
