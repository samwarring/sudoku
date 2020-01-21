#include <iostream>
#include <string>
#include <optional>
#include <vector>
#include <fstream>
#include <algorithm>
#include <locale>
#include <boost/program_options.hpp>
#include <sudoku/sudoku.h>
#include "program_options.h"

std::string formatMetrics(sudoku::Metrics metrics)
{
    std::ostringstream sout;
    auto durationMilli = std::chrono::duration_cast<std::chrono::milliseconds>(metrics.duration);
    sout << "Total Guesses: " << metrics.totalGuesses << ", ";
    sout << "Total Backtracks: " << metrics.totalBacktracks << ", ";
    sout << "Total Stack Ops: " << metrics.totalGuesses + metrics.totalBacktracks << ", ";
    sout << "Duration: " << durationMilli.count() << " ms, ";
    sout << "Guess Rate: " << (metrics.totalGuesses * 1.0 / durationMilli.count()) << " guesses/ms";
    return sout.str();
}

int handleOptions(const ProgramOptions& options)
{    
    // Print --help if necessary
    if (options.isHelp()) {
        std::cout << options.getDescription();
        return 0;
    }

    // Using square dimensions only
    sudoku::square::Dimensions dims(options.getSquareDimensionRoot());

    // Parse cell values from --input or --input-file
    std::vector<std::vector<size_t>> inputValues;
    if (!options.getInput().empty()) {
        inputValues.push_back(sudoku::parseCellValues(dims, options.getInput().c_str()));
    }
    else if (!options.getInputFile().empty()) {
        std::ifstream fin(options.getInputFile());
        std::string line;
        while (std::getline(fin, line)) {
            if (std::all_of(line.begin(), line.end(), isspace)) {
                continue;
            }
            inputValues.push_back(sudoku::parseCellValues(dims, line.c_str()));
        }
    }

    // Select the correct format
    std::optional<sudoku::Formatter> formatter;
    switch (options.getOutputFormat()) {
        case ProgramOptions::OutputFormat::PRETTY:
            formatter.emplace(sudoku::square::Formatter(dims));
            break;
        case ProgramOptions::OutputFormat::SERIAL:
            formatter.emplace(dims, std::string(dims.getCellCount(), '0'));
            break;
    }

    // Create a hashing object.
    std::hash<std::string> hasher;

    // Solve each set of input values
    for (size_t inputNum = 0; inputNum < inputValues.size(); ++inputNum) {
        const std::vector<size_t>& curInput = inputValues[inputNum];
        sudoku::Grid grid(dims, std::move(curInput));
        
        // Print the input values
        std::cout << "Input " << (inputNum + 1) << ":\n" << formatter->format(grid.getCellValues()) << '\n';

        // If --fork, fork the grid and print each forked grid.
        if (options.getForkCount() > 0) {
            auto grids = sudoku::fork(std::move(grid), options.getForkCount());
            for (size_t gridNum = 0; gridNum < grids.size(); ++gridNum) {
                std::cout << "\nPeer " << (gridNum + 1) << ":\n";
                std::cout << formatter->format(grids[gridNum].getCellValues()) << '\n';
                if (grids[gridNum].getRestrictions().size() > 0) {
                    std::cout << "Restrictions:";
                    for (auto restr : grids[gridNum].getRestrictions()) {
                        std::cout << " (" << restr.first << ',' << restr.second << ')';
                    }
                    std::cout << '\n';
                }
            }
            return 0;
        }

        // Find the first N solutions
        if (options.getThreadCount() == 1) {
            sudoku::Solver solver(std::move(grid));
            size_t numSolutionsFound = 0;
            while (numSolutionsFound < options.getNumSolutions() && solver.computeNextSolution()) {
                numSolutionsFound++;
                std::cout << "\nInput " << (inputNum + 1) << ", ";
                std::cout << "Solution " << numSolutionsFound << ", ";
                std::cout << "Hash: " << hasher(formatter->format(solver.getCellValues())) << ", ";
                std::cout << formatMetrics(solver.getMetrics()) << '\n';
                std::cout << formatter->format(solver.getCellValues()) << '\n';
            }

            if (numSolutionsFound == 0) {
                std::cout << "No solution after " << solver.getMetrics().totalGuesses << " guesses.\n";
            }
        }
        else { // options.getThreadCount > 1
            sudoku::ParallelSolver solver(std::move(grid), options.getThreadCount(), 8);
            size_t solutionCount = 0;
            while (solutionCount < options.getNumSolutions() && solver.computeNextSolution()) {
                solutionCount++;
                std::cout << "\nInput " << (inputNum + 1) << ", ";
                std::cout << "Solution " << solutionCount << ", ";
                std::cout << "Hash: " << hasher(formatter->format(solver.getCellValues())) << '\n';
                std::cout << formatter->format(solver.getCellValues()) << '\n';
            }

            if (solutionCount == 0) {
                std::cout << "No solution.\n";
            }
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    try {
        ProgramOptions options(argc, argv);
        return handleOptions(options);
    }
    catch (const boost::program_options::error& err) {
        std::cerr << "error: " << err.what() << '\n';
        return 1;
    }
    catch (const sudoku::SolverException& err) {
        std::cerr << "error: " << err.what() << '\n';
        return 1;
    }
    catch (const sudoku::CellValueParseException& err) {
        std::cerr << "error: " << err.what() << '\n';
        return 1;
    }
}
