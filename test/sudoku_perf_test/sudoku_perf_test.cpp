#include <chrono>
#include <iostream>
#include <sudoku/sudoku.h>

void addMetrics(sudoku::Metrics& target, sudoku::Metrics delta)
{
    target.totalGuesses += delta.totalGuesses;
    target.totalBacktracks += delta.totalBacktracks;
    target.duration += delta.duration;
}

size_t getGBT(sudoku::Metrics metrics)
{
    return metrics.totalGuesses + metrics.totalBacktracks;
}

double getMsec(sudoku::Metrics metrics)
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(metrics.duration).count());
}

double getGBTRate(sudoku::Metrics metrics)
{
    return getGBT(metrics) / getMsec(metrics);
}

int main()
{
    constexpr int SAMPLE_SIZE = 3;
    constexpr sudoku::CellCount root = 6;
    sudoku::square::Dimensions dims(root);
    sudoku::Metrics totalMetrics{0};

    #if SUDOKU_PERF_TEST_IS_DEBUG == 1
    std::cout << "WARNING! This is a Debug build!\n";
    #endif

    std::cout << "Starting performance test: 36x36 sudoku, 3 trials..." << std::endl;
    for (auto i = 0; i < SAMPLE_SIZE; ++i) {
        sudoku::Solver solver(dims);
        if (solver.computeNextSolution()) {
            auto metrics = solver.getMetrics();
            std::cout << "Trial " << i + 1 << '/' << SAMPLE_SIZE << ": ";
            std::cout << getGBT(metrics) << " G+BT, ";
            std::cout << getMsec(metrics) << " ms, ";
            std::cout << getGBTRate(metrics) << " G+BT/ms" << std::endl;
            addMetrics(totalMetrics, metrics);
        }
        else {
            std::cout << "Found no solution. Aborting.\n";
            return 1;
        }
    }

    std::cout << "Average: ";
    std::cout << (getGBT(totalMetrics) / SAMPLE_SIZE) << " G+BT/trial, ";
    std::cout << (getMsec(totalMetrics) / SAMPLE_SIZE) << " ms/trial, ";
    std::cout << getGBTRate(totalMetrics) << " G+BT/ms\n";
    return 0;
}
