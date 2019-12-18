#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <sudoku/sudoku.h>

namespace bpo = boost::program_options;

std::ostream& operator<<(std::ostream& ostr, const std::vector<size_t>& vec)
{
    ostr << "[ ";
    for (size_t n : vec) {
        ostr << n << ' ';
    }
    ostr << ']';
    return ostr;
}

bpo::variables_map parseOptions(int argc, char** argv)
{
    bpo::options_description description("Sudoku solver CLI");

    description.add_options()
        ("help,h", "Show usage and exit")
        ("input,i", bpo::value<std::string>(), "Initial cell values")
        ("num-solutions,n", bpo::value<size_t>(), "Compute up to this many solutions")
        ("output-format,f", bpo::value<std::string>(), "Choose from `pretty` (default) or `serial`")
        ; // end of options

    bpo::variables_map optionMap;
    try {
        bpo::store(bpo::parse_command_line(argc, argv, description), optionMap);
        bpo::notify(optionMap);

        // validate --output-format
        if (optionMap.count("output-format")) {
            const auto& outputFormat = optionMap["output-format"].as<std::string>();
            if (outputFormat != "pretty" && outputFormat != "serial") {
                throw bpo::validation_error(bpo::validation_error::invalid_option_value, "output-format");
            }
        }
    }
    catch (const std::exception& err) {
        std::cerr << description << '\n';
        std::cerr << "error: " << err.what() << '\n';
        exit(1);
    }

    if (optionMap.count("help")) {
        std::cout << description;
        exit(0);
    }

    return optionMap;
}

int main(int argc, char** argv)
{
    auto options = parseOptions(argc, argv);
    if (options.count("input")) {
        std::vector<size_t> inputValues;
        sudoku::standard::Dimensions standardDims;
        sudoku::Formatter prettyFormatter(
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
        sudoku::Formatter serialFormatter(standardDims, std::string(81, '0'));

        try {
            inputValues = sudoku::parseCellValues(
                standardDims,
                options["input"].as<std::string>().c_str()
            );
        }
        catch(const sudoku::CellValueParseException& err) {
            std::cerr << "error: " << err.what() << '\n';
            exit(1);
        }

        const sudoku::Formatter* formatter = nullptr;
        if (options.count("output-format")) {
            auto outputFormat = options["output-format"].as<std::string>();
            if (outputFormat == "pretty") {
                formatter = &prettyFormatter;
            }
            else if (outputFormat == "serial") {
                formatter = &serialFormatter;
            }
        }
        else {
            formatter = &prettyFormatter;
        }

        std::cout << "Input values:\n" << formatter->format(inputValues) << '\n';

        try {
            sudoku::Solver solver(standardDims, inputValues);

            size_t numSolutions = 1;
            if (options.count("num-solutions")) {
                numSolutions = options["num-solutions"].as<size_t>();
            }

            size_t numSolutionsFound = 0;
            while (numSolutionsFound < numSolutions && solver.computeNextSolution()) {
                numSolutionsFound++;
                auto durationNano = solver.getSolutionDuration();
                auto durationMilli = std::chrono::duration_cast<std::chrono::milliseconds>(durationNano);
                std::cout << "Solution " << numSolutionsFound << ", ";
                std::cout << "Total Guesses: " << solver.getTotalGuesses() << ", ";
                std::cout << "Duration: " << durationMilli.count() << " ms, ";
                std::cout << "Guess Rate: " << (solver.getTotalGuesses() * 1.0 / durationMilli.count()) << " guesses/ms\n";
                std::cout << formatter->format(solver.getCellValues()) << '\n';
            }

            if (numSolutionsFound == 0) {
                std::cout << "No solution after " << solver.getTotalGuesses() << " guesses.\n";
            }
        }
        catch (const sudoku::SolverException& err) {
            std::cerr << "error: " << err.what() << '\n';
            exit(1);
        }
    }
}
