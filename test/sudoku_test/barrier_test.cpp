#include <vector>
#include <thread>
#include <boost/test/unit_test.hpp>
#include <sudoku/barrier.h>

// Mapping from position to expected result
size_t computeResult(size_t x)
{
    return (x * 777773) % 121;
}

BOOST_AUTO_TEST_CASE(Barrier_SingleUse)
{
    // Each thread will populate their respective result
    size_t threadCount = 20;
    std::vector<size_t> results(threadCount);
    std::vector<std::thread> threads;
    sudoku::Barrier barrier(threadCount);

    // Start threads
    for (size_t i = 1; i < threadCount; ++i) {
        threads.emplace_back([&, i](){
            results[i] = computeResult(i);
            barrier.wait();
        });
    }

    // Compute result 0 on main thread
    results[0] = computeResult(0);
    barrier.wait();

    // Verify results.
    for (size_t i = 0; i < threadCount; ++i) {
        BOOST_CHECK_EQUAL(results[i], computeResult(i));
    }

    // According to barrier, all results are now populated.
    for (auto& t : threads) {
        t.join();
    }
}

BOOST_AUTO_TEST_CASE(Barrier_RepeatedUse)
{
    size_t threadCount = 20;
    size_t iterations = 10;
    std::vector<size_t> result(threadCount * iterations);
    std::vector<std::thread> threads;
    sudoku::Barrier barrier(threadCount);

    // Spawn work on n-1 threads.
    for (size_t i = 1; i < threadCount; ++i) {
        threads.emplace_back([i, threadCount, iterations, &barrier, &result](){
            for (size_t j = 0; j < iterations; ++j) {
                size_t x = i + (j * threadCount);
                result[x] = computeResult(x);
                barrier.wait();
            }
        });
    }

    // Do thread 0's work in the main thread.
    for (size_t j = 0; j < iterations; ++j) {
        size_t x = j * threadCount;
        result[x] = computeResult(x);
        barrier.wait();
    }

    // According to barrier, result fully populated.
    for (size_t x = 0; x < result.size(); ++x) {
        BOOST_CHECK_EQUAL(result[x], computeResult(x));
    }

    // Now join the threads
    for (auto& t : threads) {
        t.join();
    }
}

BOOST_AUTO_TEST_CASE(SpinBarrier)
{
    size_t threadCount = 8;
    std::vector<size_t> result(threadCount);
    std::vector<std::thread> threads;
    sudoku::SpinBarrier barrier(threadCount + 1);
    for(size_t i = 0; i < threadCount; ++i) {
        threads.emplace_back([i, &result, &barrier](){
            result[i] = computeResult(i);
            barrier.wait();
        });
    }
    barrier.wait();
    for (size_t i = 0; i < threadCount; ++i) {
        BOOST_CHECK_EQUAL(result[i], computeResult(i));
    }
    for (auto& t : threads) {
        t.join();
    }
}
