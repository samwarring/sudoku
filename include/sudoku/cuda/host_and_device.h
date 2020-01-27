#ifndef INCLUDED_SUDOKU_CUDA_HOST_AND_DEVICE_H
#define INCLUDED_SUDOKU_CUDA_HOST_AND_DEVICE_H

#include <cassert>

// https://stackoverflow.com/a/6978720
#ifdef __CUDACC__
#define CUDA_HOST_AND_DEVICE __host__ __device__
#define CUDA_HOST_ASSERT(x)
#else
#define CUDA_HOST_AND_DEVICE
#define CUDA_HOST_ASSERT assert
#endif

#endif
