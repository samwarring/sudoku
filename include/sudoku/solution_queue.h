#ifndef INCLUDED_SUDOKU_SOLUTION_QUEUE_H
#define INCLUDED_SUDOKU_SOLUTION_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include <sudoku/metrics.h>

namespace sudoku
{
    /**
     * Controls a multi-producer, multi-consumer, thread-safe queue.
     */
    class SolutionQueue
    {
        public:
            /**
             * \param maxSize block producers from pushing to the queue if
             *                it reaches this size.
             */
            SolutionQueue(size_t maxSize);

            /**
             * A "write-handle" to a solution queue.
             */
            class Producer
            {
                public:
                    Producer(SolutionQueue& queue);
                    Producer(const Producer& other);
                    Producer(Producer&& other);
                    ~Producer();
                    Producer& operator=(const Producer& other) = delete;
                    Producer& operator=(Producer&& other) = delete;

                    /**
                     * Add a solution to the queue. If the queue is full, block the
                     * calling thread.
                     * 
                     * \return true if a solution was added to the queue.
                     *         false if the queue has no consumers.
                     */
                    bool push(std::vector<size_t> solution, Metrics metrics);

                private:
                    SolutionQueue& queue_;
                    bool moved_ = false;
            };

            /**
             * A "read-handle" to a solution queue.
             */
            class Consumer
            {
                public:
                    Consumer(SolutionQueue& queue);
                    Consumer(const Consumer& other);
                    Consumer(Consumer&& other);
                    ~Consumer();
                    Consumer& operator=(const Consumer& other) = delete;
                    Consumer& operator=(Consumer&& other) = delete;

                    /**
                     * Read a solution from the queue. If the queue is empty, block
                     * until a solution has been added.
                     * 
                     * \return true if a solution was retrieved.
                     *         false if the queue has no producers.
                     */
                    bool pop(std::vector<size_t>& solution, Metrics& metrics);

                private:
                    SolutionQueue& queue_;
                    bool moved_ = false;
            };

        private:
            size_t maxSize_;
            std::queue<std::vector<size_t>> valuesQueue_;
            std::queue<Metrics> metricsQueue_;
            std::mutex mutex_;
            std::condition_variable condVar_;
            int numProducers_ = 0;
            int numConsumers_ = 0;
    };
}

#endif
