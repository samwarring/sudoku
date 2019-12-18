#include <sstream>
#include "program_options.h"

namespace bpo = boost::program_options;

ProgramOptions::ProgramOptions(int argc, char** argv)
    : description_("Sudoku solver CLI")
{
    description_.add_options()
        ("help,h", "Show usage and exit")
        ("input,i", bpo::value<std::string>(), "Initial cell values")
        ("num-solutions,n", bpo::value<size_t>(), "Compute up to this many solutions")
        ("output-format,f", bpo::value<std::string>(), "Choose from `pretty` (default) or `serial`")
        ; // end of options

    bpo::store(bpo::parse_command_line(argc, argv, description_), optionMap_);
    bpo::notify(optionMap_);

    // validate --output-format
    if (optionMap_.count("output-format")) {
        const auto& outputFormat = optionMap_["output-format"].as<std::string>();
        if (outputFormat != "pretty" && outputFormat != "serial") {
            throw bpo::validation_error(bpo::validation_error::invalid_option_value, "output-format");
        }
    }

    // --input is required
    if (!optionMap_.count("input")) {
        throw bpo::error("--input is required");
    }
}

std::string ProgramOptions::getDescription() const
{
    std::ostringstream sout;
    sout << description_;
    return sout.str();
}

bool ProgramOptions::isHelp() const
{
    return optionMap_.count("help") == 1;
}

std::string ProgramOptions::getInput() const
{
    if (optionMap_.count("input")) {
        return { optionMap_["input"].as<std::string>() };
    }
    else {
        return {};
    }
}

size_t ProgramOptions::getNumSolutions() const
{
    if (optionMap_.count("num-solutions")) {
        return optionMap_["num-solutions"].as<size_t>();
    }
    else {
        return 1;
    }
}

ProgramOptions::OutputFormat ProgramOptions::getOutputFormat() const
{
    if (optionMap_.count("output-format")) {
        std::string outputFormat = optionMap_["output-format"].as<std::string>();
        if (outputFormat == "pretty") {
            return OutputFormat::PRETTY;
        }
        else if (outputFormat == "serial") {
            return OutputFormat::SERIAL;
        }
    }
    else {
        return OutputFormat::PRETTY;
    }
}
