#ifndef INCLUDED_CLOCK_KERNELS_H
#define INCLUDED_CLOCK_KERNELS_H

#include <chrono>
#include <ctime>
#include <sudoku/cuda/device_buffer.h>

class ClockKernels
{
    private:
        sudoku::cuda::DeviceBuffer<clock_t> deviceTickCount_{1};
        clock_t hostTickCount_{0};

    public:
        void doWork(unsigned iterations);

        std::chrono::milliseconds getMs() const;

        clock_t getTickCount() const { return hostTickCount_; }
};

#endif
