#ifndef INCLUDED_SUDOKU_METRICS_H
#define INCLUDED_SUDOKU_METRICS_H

#include <chrono>

namespace sudoku
{
    struct Metrics
    {
        /**
         * Underlying clock-type for measuring time.
         */
        using Clock = std::chrono::high_resolution_clock;
        
        /**
         * Underlying type for time durations.
         */
        using Duration = Clock::duration;

        /**
         * Underlying type for time pints.
         */
        using TimePoint = Clock::time_point;

        /**
         * Get a time-point with the same clock as the duration
         */
        static TimePoint now() { return Clock::now(); }
        
        /**
         * Total number of guesses pushed.
         */
        size_t totalGuesses = 0;

        /**
         * Total number of backtracks (guesses popped).
         */
        size_t totalBacktracks = 0;

        /**
         * Total amount of time consumed by the solver.
         */
        Duration duration{0};
    };
}

#endif
