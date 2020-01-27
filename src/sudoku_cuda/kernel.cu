#include <sudoku/cuda/kernel.h>

namespace sudoku
{
    namespace cuda
    {
        __global__ void computeNextSolutionKernel(Result* results)
        {
            results[threadIdx.x] = Result::OK_TIMED_OUT;
        }

        void computeNextSolutionKernelWrapper(unsigned blockCount, unsigned threadsPerBlock, Result* results)
        {
            computeNextSolutionKernel<<<blockCount, threadsPerBlock>>>(results);
        }
    }
}
