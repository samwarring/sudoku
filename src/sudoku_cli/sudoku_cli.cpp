#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <locale>
#include <memory>
#include <sstream>
#include <boost/program_options.hpp>
#include <sudoku/sudoku.h>
#include "program_options.h"

std::string computeSerialFormatString(const sudoku::Dimensions& dims)
{
    std::ostringstream ossDigits;
    ossDigits << dims.getMaxCellValue();
    size_t numDigits = ossDigits.str().length();
    std::string placeholder(numDigits, '0');
    
    std::ostringstream ossResult;
    for (sudoku::CellCount pos = 0; pos < dims.getCellCount(); ++pos) {
        ossResult << placeholder << ' ';
    }
    return ossResult.str();
}

std::string formatMetrics(sudoku::Metrics metrics)
{
    std::ostringstream sout;
    auto durationMilli = std::chrono::duration_cast<std::chrono::milliseconds>(metrics.duration);
    auto totalGuessesAndBacktracks = metrics.totalGuesses + metrics.totalBacktracks;
    sout << metrics.totalGuesses << " G, ";
    sout << metrics.totalBacktracks << " BT, ";
    sout << totalGuessesAndBacktracks << " G+BT, ";
    sout << durationMilli.count() << " ms, ";
    sout << (totalGuessesAndBacktracks * 1.0 / durationMilli.count()) << " G+BT/ms";
    return sout.str();
}

std::unique_ptr<sudoku::SolverInterface> solverFactory(const sudoku::Dimensions& dims,
                                                       std::vector<sudoku::CellValue> initialValues,
                                                       const ProgramOptions& options)
{
    auto emptyCount = std::count(initialValues.begin(), initialValues.end(), 0);
    if (options.isGroupwise()) {
        if (options.getThreadCount() > 1) {
            throw std::runtime_error("Groupwise solver does not support multiple threads");
        }
        if (emptyCount == dims.getCellCount()) {
            return std::make_unique<sudoku::GroupwiseEmptySolver>(dims);
        }
        else {
            return std::make_unique<sudoku::GroupwiseSolver>(dims, initialValues);
        }
    }

    sudoku::Grid grid(dims, initialValues);
    if (options.getThreadCount() == 1) {
        return std::make_unique<sudoku::Solver>(grid);
    }
    else if (options.getThreadCount() > 1) {
        return std::make_unique<sudoku::ParallelSolver>(grid, options.getThreadCount(), options.getNumSolutions());
    }
    else {
        throw std::runtime_error("Invalid thread count");
    }
}

int handleOptions(const ProgramOptions& options)
{    
    // Print --help if necessary
    if (options.isHelp()) {
        std::cout << options.getDescription();
        return 0;
    }

    // Using inner-rectangular dimensions only
    auto innerRectangleDims = options.getInnerRectangularSize();
    sudoku::inner_rectangular::Dimensions dims(innerRectangleDims.first,
                                               innerRectangleDims.second);

    // Parse cell values from --input or --input-file
    std::vector<std::vector<sudoku::CellValue>> inputValues;
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
    else {
        // If no --input or --input-file, assume empty sudoku
        inputValues.push_back(sudoku::parseCellValues(dims, "0"));
    }

    // Select the correct format
    std::unique_ptr<sudoku::Formatter> formatter;
    switch (options.getOutputFormat()) {
        case ProgramOptions::OutputFormat::PRETTY:
            formatter.reset(new sudoku::inner_rectangular::Formatter(dims));
            break;
        case ProgramOptions::OutputFormat::SERIAL:
            formatter.reset(new sudoku::Formatter(dims, computeSerialFormatString(dims)));
            break;
    }

    // Create a hashing object.
    std::hash<std::string> hasher;

    // Solve each set of input values
    for (size_t inputNum = 0; inputNum < inputValues.size(); ++inputNum) {
        const std::vector<sudoku::CellValue>& curInput = inputValues[inputNum];
        
        // Print the input values
        if (options.isEchoInput()) {
            std::cout << "Input " << (inputNum + 1) << ":\n" << formatter->format(curInput) << '\n';
        }

        // If --fork, fork the grid and print each forked grid.
        if (options.getForkCount() > 0) {
            sudoku::Grid grid(dims, curInput);
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
        auto solver = solverFactory(dims, curInput, options);
        size_t solutionCount = 0;
        while (solutionCount < options.getNumSolutions() && solver->computeNextSolution()) {
            solutionCount++;
            std::cout << "\nInput " << (inputNum + 1) << ", ";
            std::cout << "Solution " << solutionCount << ", ";
            std::cout << "Hash: " << hasher(formatter->format(solver->getCellValues())) << ", ";
            std::cout << "Metrics: " << formatMetrics(solver->getMetrics()) << '\n';
            std::cout << formatter->format(solver->getCellValues()) << '\n';
        }

        if (solutionCount == 0) {
            std::cout << "No solution\n";
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
    catch (const std::exception& err) {
        std::cerr << "error: " << err.what() << '\n';
    }
}
