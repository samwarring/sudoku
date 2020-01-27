#include "util.h"

sudoku::cuda::KernelParams makeHostParams(const sudoku::cuda::DimensionParams& dimParams)
{
    sudoku::cuda::KernelParams params;
    params.cellCount = dimParams.cellCount;
    params.maxCellValue = dimParams.maxCellValue;
    params.groupCount = dimParams.groupCount;
    params.groupValues = dimParams.groupValues.data();
    params.groupOffsets = dimParams.groupOffsets.data();
    params.groupsForCellValues = dimParams.groupsForCellValues.data();
    params.groupsForCellOffsets = dimParams.groupsForCellOffsets.data();
    return params;
}

sudoku::cuda::KernelParams makeHostParams(
    const sudoku::cuda::DimensionParams& dimParams,
    sudoku::cuda::GridParams& gridParams)
{
    auto params = makeHostParams(dimParams);
    params.cellValues = gridParams.cellValues.data();
    params.restrictions = gridParams.restrictions.data();
    params.restrictionsOffsets = gridParams.restrictionsOffsets.data();
    return params;
}
