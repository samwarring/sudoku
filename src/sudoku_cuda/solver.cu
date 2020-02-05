#include <stdexcept>
#include <sudoku/cuda/solver.h>
#include <sudoku/cuda/device_solver.cuh>
#include <sudoku/cuda/util.h>

namespace sudoku
{
    namespace cuda
    {
        constexpr CellValue SOLVER_MAX_CELL_VALUE = 49;
        constexpr GroupCount SOLVER_MAX_GROUPS_FOR_CELL = 3;

        /**
         * Contains common arguments passed to kernels.
         */
        struct KernelParams
        {
            // Common
            CellCount cellCount;
            CellValue maxCellValue;
            // Related groups
            GroupCount totalGroupCount;
            GroupCount* groupCounts;
            GroupCount* groupIds;
            // Block counter
            CellBlockCount* cellBlockCounts;
            ValueBlockCount* valueBlockCounts;
            // Guess stack
            Guess* guessStackValues;
            CellCount* guessStackSize;
            // Grid
            CellValue* cellValues;
        };

        /**
         * Constructs sudoku::cuda::* objects at the start of each kernel
         */
        template <CellValue MAX_CELL_VALUE, GroupCount MAX_GROUPS_FOR_CELL>
        struct KernelObjects
        {
        public:
            using BlockCounter = BlockCounter<MAX_CELL_VALUE>;
            using BlockCounterPair = BlockCounter::Pair;
            using RelatedGroups = RelatedGroups<MAX_GROUPS_FOR_CELL>;
            using Grid = Grid<MAX_CELL_VALUE, MAX_GROUPS_FOR_CELL>;
            using DeviceSolver = DeviceSolver<MAX_CELL_VALUE, MAX_GROUPS_FOR_CELL>;

        private:
            Guess* sharedGuessStack_;
            BlockCounterPair* sharedReductionBuffer_;
            CellValue* sharedGroupUpdates_;
            CellValue* sharedBroadcastValue_;

        public:
            RelatedGroups relatedGroups;
            BlockCounter blockCounter;
            Grid grid;
            GuessStack guessStack;
            DeviceSolver deviceSolver;

            __device__ KernelObjects(KernelParams kp, int* sharedMem)
                : sharedGuessStack_(reinterpret_cast<Guess*>(sharedMem))
                , sharedReductionBuffer_(reinterpret_cast<BlockCounterPair*>(
                                         sharedGuessStack_ + kp.cellCount))
                , sharedGroupUpdates_(reinterpret_cast<CellValue*>(sharedReductionBuffer_ + blockDim.x))
                , sharedBroadcastValue_(reinterpret_cast<CellValue*>(sharedGroupUpdates_ + kp.totalGroupCount))
                , relatedGroups(kp.cellCount, kp.totalGroupCount, kp.groupCounts, kp.groupIds, sharedGroupUpdates_)
                , blockCounter(kp.cellCount, kp.maxCellValue, kp.cellBlockCounts, kp.valueBlockCounts, sharedReductionBuffer_)
                , grid(relatedGroups, blockCounter, kp.cellValues, kp.cellCount)
                , guessStack(kp.guessStackValues, kp.guessStackSize, sharedGuessStack_)
                , deviceSolver(relatedGroups, blockCounter, grid, guessStack, kp.cellCount, kp.maxCellValue, sharedBroadcastValue_)
            {}
        };

        template <CellValue MAX_CELL_VALUE, GroupCount MAX_GROUPS_FOR_CELL>
        __global__ void computeNextSolutionKernel(KernelParams kp, Result* outResult)
        {
            extern __shared__ int sharedMem[];
            KernelObjects<MAX_CELL_VALUE, MAX_GROUPS_FOR_CELL> ko(kp, sharedMem);
            *outResult = ko.deviceSolver.computeNextSolution(1000);
        }

        Solver::Solver(const sudoku::Grid& grid) : cellCount_(static_cast<CellCount>(grid.getDimensions().getCellCount()))
                                                 , maxCellValue_(static_cast<CellValue>(grid.getDimensions().getMaxCellValue()))
                                                 , cellCountPow2_(nearestPowerOf2(cellCount_))
                                                 //, sharedMemSize_ = TODO
                                                 , totalGroupCount_(static_cast<GroupCount>(grid.getDimensions().getNumGroups()))
                                                 , groupCounts_(RelatedGroups<3>::getGroupCounts(grid.getDimensions()))
                                                 , groupIds_(RelatedGroups<3>::getGroupIds(grid.getDimensions()))
                                                 , cellBlockCounts_(cellCount_)
                                                 , valueBlockCounts_(cellCount_ * maxCellValue_)
                                                 , guessStackValues_(cellCount_)
                                                 , guessStackSize_(1)
                                                 , hostCellValues_(castCellValues(grid.getCellValues()))
                                                 , deviceCellValues_(hostCellValues_)
        {
            if (maxCellValue_ > SOLVER_MAX_CELL_VALUE) {
                throw std::runtime_error("Max cell value too large for sudoku::cuda::Solver");
            }
            size_t maxGroupsForCell = groupIds_.size() / cellCount_;
            if (maxGroupsForCell > SOLVER_MAX_GROUPS_FOR_CELL) {
                throw std::runtime_error("Max groups for cell too large for sudoku::cuda::Solver");
            }

            sharedMemSize_ = (
                (sizeof(Guess) * cellCount_) + 
                (sizeof(BlockCounter<SOLVER_MAX_CELL_VALUE>::Pair) * cellCountPow2_) +
                (sizeof(CellValue) * totalGroupCount_) +
                (sizeof(CellValue) * 1)
            );
        }

        bool Solver::computeNextSolution()
        {
            KernelParams kp;
            kp.cellCount = cellCount_;
            kp.maxCellValue = maxCellValue_;
            kp.totalGroupCount = totalGroupCount_;
            kp.groupCounts = groupCounts_.get();
            kp.groupIds = groupIds_.get();
            kp.cellBlockCounts = cellBlockCounts_.get();
            kp.valueBlockCounts = valueBlockCounts_.get();
            kp.guessStackValues = guessStackValues_.get();
            kp.guessStackSize = guessStackSize_.get();
            kp.cellValues = deviceCellValues_.get();
            DeviceBuffer<Result> outResult(1);
            computeNextSolutionKernel<9, 3><<<1, cellCountPow2_, sharedMemSize_>>>(kp, outResult.get());
            ErrorCheck::lastError();
            hostCellValues_ = deviceCellValues_.copyToHost();
            Result kernelResult = outResult.copyToHost()[0];
            return (kernelResult == Result::FOUND_SOLUTION);
        }

        const std::vector<CellValue>& Solver::getCellValues() const
        {
            return hostCellValues_;
        }

        std::vector<CellValue> Solver::castCellValues(const std::vector<size_t>& cellValues)
        {
            std::vector<CellValue> result;
            result.reserve(cellValues.size());
            for (auto cellValue : cellValues) {
                result.push_back(static_cast<CellValue>(cellValue));
            }
            return result;
        }

        std::vector<size_t> Solver::castCellValues(const std::vector<CellValue>& cellValues)
        {
            std::vector<size_t> result;
            result.reserve(cellValues.size());
            for (auto cellValue : cellValues) {
                result.push_back(static_cast<size_t>(cellValue));
            }
            return result;
        }
    }
}