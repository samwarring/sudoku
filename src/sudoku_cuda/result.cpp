#include <sudoku/cuda/result.h>

namespace sudoku
{
    namespace cuda
    {
        std::string toString(Result result)
        {
            switch (result) {
                case Result::ERROR_INVALID_ARG:
                    return "Invalid argument";
                case Result::ERROR_NOT_SET:
                    return "Not set";
                case Result::OK_FOUND_SOLUTION:
                    return "Found solution";
                case Result::OK_TIMED_OUT:
                    return "Timed out";
                case Result::OK_NO_SOLUTION:
                    return "No solution";
                default:
                    return "Undefined";
            }
        }
    }
}

namespace std
{
    ostream& operator<<(ostream& out, sudoku::cuda::Result result)
    {
        out << sudoku::cuda::toString(result);
        return out;
    }
}
