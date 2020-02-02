#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <sudoku/solution_queue.h>
#include "util.h"

BOOST_AUTO_TEST_CASE(SolutionQueue_Constructor)
{
    sudoku::SolutionQueue q(1);
}

BOOST_AUTO_TEST_CASE(SolutionQueue_PushAndPop_SingleThread)
{
    sudoku::SolutionQueue q(2);
    sudoku::SolutionQueue::Producer p(q);
    sudoku::SolutionQueue::Consumer c(q);
    
    // Sample solutions + metrics
    std::vector<size_t> s1{1, 2, 3}, s2{4, 5, 6};
    sudoku::Metrics m1, m2;
    m1.totalGuesses = 123;
    m2.totalGuesses = 456;

    // Push some solutions
    BOOST_REQUIRE(p.push(s1, m1));
    BOOST_REQUIRE(p.push(s2, m2));
    
    // Read the solutions
    std::vector<size_t> solution;
    sudoku::Metrics metrics;
    
    BOOST_REQUIRE(c.pop(solution, metrics));
    BOOST_REQUIRE_EQUAL(solution, s1);
    BOOST_REQUIRE_EQUAL(metrics.totalGuesses, m1.totalGuesses);

    BOOST_REQUIRE(c.pop(solution, metrics));
    BOOST_REQUIRE_EQUAL(solution, s2);
    BOOST_REQUIRE_EQUAL(metrics.totalGuesses, m2.totalGuesses);
}

BOOST_AUTO_TEST_CASE(SolutionQueue_ParallelProducers)
{
    sudoku::SolutionQueue q(16);
    sudoku::SolutionQueue::Consumer c(q);
    std::vector<std::thread> threads;
    const size_t numThreads = 4;

    // Create threads
    for (size_t i = 0; i < numThreads; ++i) {
        sudoku::SolutionQueue::Producer p(q);
        threads.emplace_back([i, p=std::move(p)]() mutable {
            // Each thread computes many solutions.
            // Thread 0: {0, 1, 2} ... {99, 100, 101}
            // ...
            // Thread 3: {300, 301, 302} ... {399, 400, 401}
            constexpr size_t numSolutionsPerThread = 100;
            for (size_t solutionNum = 0; solutionNum < numSolutionsPerThread; ++solutionNum) {
                const size_t base = (numSolutionsPerThread * i) + solutionNum;
                std::vector<size_t> solution = {base, base + 1, base + 2};
                sudoku::Metrics metrics;
                metrics.totalGuesses = i;
                if (!p.push(solution, metrics)) {
                    // Stop generating solutions
                    return;
                }
            }
            return;
        });
    }

    // Count solutions
    size_t numSolutions = 0;
    std::vector<size_t> solution;
    sudoku::Metrics metrics;
    while (c.pop(solution, metrics)) {
        ++numSolutions;
    }
    BOOST_CHECK_EQUAL(numSolutions, 400);

    for (auto& thread : threads) {
        thread.join();
    }
}
