#ifndef INCLUDED_SUDOKU_CUDA_HOST_AND_DEVICE_H
#define INCLUDED_SUDOKU_CUDA_HOST_AND_DEVICE_H

// https://stackoverflow.com/a/6978720
#ifdef __CUDACC__
#define CUDA_HOST_AND_DEVICE __host__ __device__
#else
#define CUDA_HOST_AND_DEVICE
#endif

#endif
