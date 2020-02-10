#include <sudoku/cuda/clock.cuh>
#include <sudoku/cuda/error_check.h>
#include "clock_kernels.h"

__global__ void doWorkKernel(unsigned iterations, float* data, clock_t* tickCount)
{
    sudoku::cuda::Clock timer(tickCount);
    for (unsigned i = 0; i < iterations; ++i) {
        data[threadIdx.x] *= 3.14;
    }
}

void ClockKernels::doWork(unsigned iterations)
{
    sudoku::cuda::DeviceBuffer<float> deviceData(1024);
    doWorkKernel<<<1, 1024>>>(iterations, deviceData.get(), deviceTickCount_.get());
    sudoku::cuda::ErrorCheck::lastError();
    hostTickCount_ = deviceTickCount_.copyToHost()[0];
}

std::chrono::milliseconds ClockKernels::getMs() const
{
    return sudoku::cuda::Clock::readClock(hostTickCount_);
}
