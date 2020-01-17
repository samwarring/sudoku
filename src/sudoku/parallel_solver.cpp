#include <sudoku/parallel_solver.h>

namespace sudoku
{
    ParallelSolver::ParallelSolver(const Dimensions& dims, std::vector<size_t> cellValues,
                                   size_t threadCount, size_t queueSize) 
                                   : dims_(dims), cellValues_(std::move(cellValues))
                                   , queue_(queueSize), consumer_(new SolutionQueue::Consumer(queue_))
                                   , threadCount_(threadCount)
    {
        if (threadCount_ <= 1) {
            throw ParallelSolverException("ParallelSolver must use >1 thread");
        }
    }

    ParallelSolver::~ParallelSolver()
    {
        consumer_.reset();
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
        if (!consumer_->pop(cellValues_)) {
            // No more producers. Likely because all the threads
            // have terminated. No more solutions!
            return false;
        }

        // We successfully popped a solution from the queue! User
        // can read it with getCellValues().
        return true;
    }

    void ParallelSolver::startThreads()
    {
        // Create the solvers
        auto firstSolver = std::make_unique<Solver>(dims_, std::move(cellValues_));
        solvers_ = firstSolver->fork(threadCount_ - 1);
        solvers_.push_back(std::move(firstSolver));

        // Create the threads
        for (size_t threadNum = 0; threadNum < solvers_.size(); ++threadNum) {

            // Each thread gets a solver (to compute solutions) and a producer
            // (to send each solution to the queue).
            SolutionQueue::Producer producer(queue_);
            Solver* solver = solvers_[threadNum].get();

            // Procedure for each thread: produce solutions until the
            // queue is closed, or until the solver finds no more solutions.
            auto threadProc = [solver, producer = std::move(producer)]() mutable {
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
