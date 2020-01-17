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

    // Solve each set of input values
    for (size_t inputNum = 0; inputNum < inputValues.size(); ++inputNum) {
        const std::vector<size_t>& curInput = inputValues[inputNum];
        
        // Print the input values
        std::cout << "Input " << (inputNum + 1) << ":\n" << formatter->format(curInput) << '\n';

        // Find the first N solutions
        if (options.getThreadCount() == 1) {
            sudoku::Solver solver(dims, curInput);
            size_t numSolutionsFound = 0;
            while (numSolutionsFound < options.getNumSolutions() && solver.computeNextSolution()) {
                numSolutionsFound++;
                auto duration = solver.getSolutionDuration();
                auto durationMilli = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
                std::cout << "Input " << (inputNum + 1) << ", ";
                std::cout << "Solution " << numSolutionsFound << ", ";
                std::cout << "Total Guesses: " << solver.getTotalGuesses() << ", ";
                std::cout << "Total Backtracks: " << solver.getTotalBacktracks() << ", ";
                std::cout << "Total Stack Ops: " << solver.getTotalGuesses() + solver.getTotalBacktracks() << ", ";
                std::cout << "Duration: " << durationMilli.count() << " ms, ";
                std::cout << "Guess Rate: " << (solver.getTotalGuesses() * 1.0 / durationMilli.count()) << " guesses/ms\n";
                std::cout << formatter->format(solver.getCellValues()) << '\n';
            }

            if (numSolutionsFound == 0) {
                std::cout << "No solution after " << solver.getTotalGuesses() << " guesses.\n";
            }
        }
        else { // options.getThreadCount > 1
            sudoku::ParallelSolver solver(dims, curInput, options.getThreadCount(), 8);
            auto solutionCount = 0;
            while (solutionCount < options.getNumSolutions() && solver.computeNextSolution()) {
                solutionCount++;
                std::cout << "Input " << (inputNum + 1) << ", ";
                std::cout << "Solution " << solutionCount << '\n';
                std::cout << formatter->format(solver.getCellValues()) << '\n';
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
