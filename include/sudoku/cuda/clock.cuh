#ifndef INCLUDED_SUDOKU_CUDA_CLOCK_CUH
#define INCLUDED_SUDOKU_CUDA_CLOCK_CUH

#include <chrono>
#include <cuda_runtime.h>
#include <sudoku/cuda/error_check.h>

namespace sudoku
{
    namespace cuda
    {
        class Clock
        {
        private:
            clock_t* globalTickCount_;
            clock_t tickCount_;
            clock_t start_;

        public:
            __device__ Clock(clock_t* globalTickCount)
                : globalTickCount_(globalTickCount)
                , tickCount_(*globalTickCount_)
                , start_(clock())
            {}

            __device__ ~Clock()
            {
                tickCount_ += (clock() - start_);
                if (threadIdx.x == 0) {
                    *globalTickCount_ = tickCount_;
                }
            }

            static std::chrono::milliseconds readClock(unsigned long long tickCount)
            {
                cudaDeviceProp props;
                ErrorCheck(cudaGetDeviceProperties(&props, 0));
                auto khz = static_cast<unsigned long long>(props.clockRate);
                
                // ms = (N ticks) * (1 s / K * 1000 tiks) * (1000 ms / s) = N/K
                return std::chrono::milliseconds{ tickCount / khz };
            }
        };
    }
}

#endif
