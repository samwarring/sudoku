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
        ; // end of options

    bpo::variables_map optionMap;
    try {
        bpo::store(bpo::parse_command_line(argc, argv, description), optionMap);
        bpo::notify(optionMap);
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
        sudoku::Formatter formatter(standardDims,
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

        std::cout << "Input values:\n" << formatter.format(inputValues) << '\n';

        sudoku::Solver solver(standardDims, inputValues);

        size_t numSolutions = 1;
        if (options.count("num-solutions")) {
            numSolutions = options["num-solutions"].as<size_t>();
        }

        for (size_t n = 0; n < numSolutions && solver.computeNextSolution(); ++n) {
            std::cout << "Solution " << (n + 1) << ":\n";
            std::cout << formatter.format(solver.getCellValues()) << '\n';
        }
    }
}
