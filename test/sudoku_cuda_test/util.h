#ifndef INCLUDED_SUDOKU_CUDA_TEST_UTIL_H
#define INCLUDED_SUDOKU_CUDA_TEST_UTIL_H

#include <sudoku/cuda/kernel.h>

/**
 * \file util.h
 * 
 * Convenience methods for unit-testing. Set up kernel parameters such that
 * buffers point to host-memory owned by the DimensionParams and GridParams.
 */

sudoku::cuda::KernelParams makeHostParams(const sudoku::cuda::DimensionParams& dimParams);

sudoku::cuda::KernelParams makeHostParams(const sudoku::cuda::DimensionParams& dimParams,
                                          sudoku::cuda::GridParams& gridParams);

#endif
