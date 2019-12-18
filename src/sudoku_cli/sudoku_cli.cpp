#include <iostream>
#include <string>
#include <optional>
#include <vector>
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

    // Parse cell values from --input
    std::vector<size_t> inputValues;
    sudoku::standard::Dimensions standardDims;
    inputValues = sudoku::parseCellValues(
        standardDims,
        options.getInput().c_str()
    );

    // Select the correct format
    std::optional<sudoku::Formatter> formatter;
    switch (options.getOutputFormat()) {
        case ProgramOptions::OutputFormat::PRETTY:
            formatter.emplace(
                standardDims,
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "------+-------+------\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "------+-------+------\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
                "0 0 0 | 0 0 0 | 0 0 0\n"
            );
            break;
        case ProgramOptions::OutputFormat::SERIAL:
            formatter.emplace(standardDims, std::string(81, '0'));
            break;
    }

    // Print the input values
    std::cout << "Input values:\n" << formatter->format(inputValues) << '\n';

    // Find the first N solutions
    sudoku::Solver solver(standardDims, inputValues);
    size_t numSolutionsFound = 0;
    while (numSolutionsFound < options.getNumSolutions() && solver.computeNextSolution()) {
        numSolutionsFound++;
        auto duration = solver.getSolutionDuration();
        auto durationMilli = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        std::cout << "Solution " << numSolutionsFound << ", ";
        std::cout << "Total Guesses: " << solver.getTotalGuesses() << ", ";
        std::cout << "Duration: " << durationMilli.count() << " ms, ";
        std::cout << "Guess Rate: " << (solver.getTotalGuesses() * 1.0 / durationMilli.count()) << " guesses/ms\n";
        std::cout << formatter->format(solver.getCellValues()) << '\n';
    }

    if (numSolutionsFound == 0) {
        std::cout << "No solution after " << solver.getTotalGuesses() << " guesses.\n";
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
