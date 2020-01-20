#include <sudoku/parallel_solver.h>
#include <sudoku/fork.h>

namespace sudoku
{
    ParallelSolver::ParallelSolver(Grid grid, size_t threadCount, size_t queueSize) 
                                   : grid_(std::move(grid)), queue_(queueSize),
                                     consumer_(new SolutionQueue::Consumer(queue_)),
                                     threadCount_(threadCount)
    {
        if (threadCount_ <= 1) {
            throw ParallelSolverException("ParallelSolver must use >1 thread");
        }
    }

    ParallelSolver::~ParallelSolver()
    {
        // Destroy the consumer. Threads waiting to push their solution
        // will shut down.
        consumer_.reset();

        // Halt the solvers. Threads endlessly searching a vast solution
        // space will shut down.
        for (auto& solver : solvers_) {
            solver->halt();
        }

        // Join the threads.
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    bool ParallelSolver::computeNextSolution()
    {
        // Start threads if necessary.
        if (threads_.size() == 0) {
            startThreads();
        }

        // Wait for the next solution in the queue.
        if (!consumer_->pop(solution_)) {
            // No more producers. Likely because all the threads
            // have terminated. No more solutions!
            return false;
        }

        // We successfully popped a solution from the queue! User
        // can read it with getCellValues().
        return true;
    }

    const std::vector<size_t>& ParallelSolver::getCellValues() const
    {
        if (threads_.size() > 0) {
            return solution_;
        }
        else {
            return grid_.getCellValues();
        }
    }

    void ParallelSolver::startThreads()
    {
        // Create the peer grids
        auto grids = sudoku::fork(std::move(grid_), threadCount_);

        // Create the threads
        for (size_t threadNum = 0; threadNum < grids.size(); ++threadNum) {

            // Each thread gets a solver (to compute solutions) and a producer
            // (to send each solution to the queue).
            SolutionQueue::Producer producer(queue_);
            solvers_.emplace_back(std::make_unique<Solver>(std::move(grids[threadNum])));

            // Procedure for each thread: produce solutions until the
            // queue is closed, or until the solver finds no more solutions.
            auto threadProc = [solver = solvers_[threadNum].get(), producer = std::move(producer)]() mutable {
                 while (solver->computeNextSolution()) {
                     if (!producer.push(solver->getCellValues())) {
                         // No more consumers. Likely because the ParallelSolver
                         // is being destroyed.
                         return;
                     }
                 }
            };

            // Start the thread.
            threads_.emplace_back(threadProc);
        }
    }
}
