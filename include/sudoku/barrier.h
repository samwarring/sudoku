#ifndef INCLUDED_SUDOKU_BARRIER_H
#define INCLUDED_SUDOKU_BARRIER_H

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace sudoku
{
    /// https://stackoverflow.com/a/27118537
    class Barrier
    {
        public:
            Barrier(size_t threadCount) : threadCount_(threadCount), waitCount_(0), generation_(0) {}

            void wait()
            {
                std::unique_lock<std::mutex> lock(mutex_);
                auto prevGeneration = generation_;
                if (++waitCount_ == threadCount_) {
                    generation_++;
                    waitCount_ = 0;
                    cv_.notify_all();
                }
                else {
                    cv_.wait(lock, [this, prevGeneration](){
                        return generation_ != prevGeneration;
                    });
                }
            }

        private:
            size_t threadCount_;
            std::mutex mutex_;
            size_t waitCount_;
            size_t generation_;
            std::condition_variable cv_;
    };

    /// https://stackoverflow.com/a/24777186
    class SpinBarrier
    {
        public:
            SpinBarrier(size_t threadCount) : threadCount_(threadCount), waitCount_(0), generation_(0) {}

            void wait()
            {
                auto prevGeneration = generation_.load();
                auto prevWaitCount = waitCount_.fetch_add(1);
                if (prevWaitCount == (threadCount_ - 1)) {
                    waitCount_.store(0);
                    ++generation_;
                }
                else {
                    while (prevGeneration == generation_.load()) { /* spin */ }
                }
            }

        private:
            size_t threadCount_;
            std::atomic<size_t> waitCount_;
            std::atomic<size_t> generation_;
    };
}

#endif
