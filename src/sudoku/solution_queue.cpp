#include <sudoku/solution_queue.h>

namespace sudoku
{
    SolutionQueue::SolutionQueue(size_t maxSize) : maxSize_(maxSize)
    {
    }

    SolutionQueue::Producer::Producer(SolutionQueue& queue) : queue_(queue)
    {
        std::lock_guard<std::mutex> lock(queue_.mutex_);
        queue_.numProducers_++;
    }

    SolutionQueue::Producer::Producer(const Producer& other) : queue_(other.queue_)
    {
        std::lock_guard<std::mutex> lock(queue_.mutex_);
        queue_.numProducers_++;
    }

    SolutionQueue::Producer::Producer(Producer&& other) : queue_(other.queue_)
    {
        other.moved_ = true;
    }

    SolutionQueue::Producer::~Producer()
    {
        if (!moved_) {
            std::lock_guard<std::mutex> lock(queue_.mutex_);
            queue_.numProducers_--;

            // If this is the last producer, notify remaining consumers.
            if (queue_.numProducers_ == 0) {
                queue_.condVar_.notify_all();
            }
        }
    }

    bool SolutionQueue::Producer::push(std::vector<CellValue> solution, Metrics metrics)
    {
        // Wait until the queue has free capacity, or until there are
        // no more consumers.
        std::unique_lock<std::mutex> lock(queue_.mutex_);
        queue_.condVar_.wait(lock, [&](){
            return (queue_.valuesQueue_.size() < queue_.maxSize_) || (queue_.numConsumers_ == 0);
        });

        // If there are no consumers, then return false now.
        if (queue_.numConsumers_ == 0) {
            return false;
        }

        // There must be free capacity in the queue. Add the solution to the
        // queue, and notify a consumer thread which may be waiting.
        queue_.valuesQueue_.emplace(std::move(solution));
        queue_.metricsQueue_.push(metrics);
        queue_.condVar_.notify_one();
        return true;
    }

    SolutionQueue::Consumer::Consumer(SolutionQueue& queue) : queue_(queue)
    {
        std::lock_guard<std::mutex> lock(queue_.mutex_);
        queue_.numConsumers_++;
    }

    SolutionQueue::Consumer::Consumer(const Consumer& other) : queue_(other.queue_)
    {
        std::lock_guard<std::mutex> lock(queue_.mutex_);
        queue_.numConsumers_++;
    }

    SolutionQueue::Consumer::Consumer(Consumer&& other) : queue_(other.queue_)
    {
        other.moved_ = true;
    }

    SolutionQueue::Consumer::~Consumer()
    {
        if (!moved_) {
            std::lock_guard<std::mutex> lock(queue_.mutex_);
            queue_.numConsumers_--;

            // If this is the last consumer, notify remaining producers.
            if (queue_.numConsumers_ == 0) {
                queue_.condVar_.notify_all();
            }
        }
    }

    bool SolutionQueue::Consumer::pop(std::vector<CellValue>& solution, Metrics& metrics)
    {
        // Wait until the queue contains at least one element, or until
        // there are no more producers.
        std::unique_lock<std::mutex> lock(queue_.mutex_);
        queue_.condVar_.wait(lock, [&](){
            return (queue_.valuesQueue_.size() > 0) || (queue_.numProducers_ == 0);
        });

        // If there are no more producers, there still may be solutions left
        // to proces in the queue. We only quit if there are no more producers
        // AND the queue is empty.
        if (queue_.numProducers_ == 0 && queue_.valuesQueue_.size() == 0) {
            return false;
        }

        // We know there is at least one producer, and the queue is not empty.
        // Retrieve the front of the queue. Notify a consumer which may be
        // waiting to write the the queue.
        solution = std::move(queue_.valuesQueue_.front());
        metrics = queue_.metricsQueue_.front();
        queue_.valuesQueue_.pop();
        queue_.metricsQueue_.pop();
        queue_.condVar_.notify_one();
        return true;
    }
}
