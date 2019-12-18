#include <sstream>
#include "program_options.h"

namespace bpo = boost::program_options;

ProgramOptions::ProgramOptions(int argc, char** argv)
    : description_("Sudoku solver CLI")
    , outputFormat_(OutputFormat::PRETTY)
{
    description_.add_options()
        ("help,h", "Show usage and exit")
        ("input,i", bpo::value<std::string>(), "Initial cell values")
        ("num-solutions,n", bpo::value<size_t>(), "Compute up to this many solutions")
        ("output-format,f", bpo::value<std::string>(), "Choose from `pretty` (default) or `serial`")
        ("input-file,I", bpo::value<std::string>(), "Read multiple initial value strings from input file")
        ; // end of options

    bpo::store(bpo::parse_command_line(argc, argv, description_), optionMap_);
    bpo::notify(optionMap_);

    // validate --output-format
    if (optionMap_.count("output-format")) {
        const auto& outputFormat = optionMap_["output-format"].as<std::string>();
        if (outputFormat == "pretty") {
            outputFormat_ = OutputFormat::PRETTY;
        }
        else if (outputFormat == "serial") {
            outputFormat_ = OutputFormat::SERIAL;
        }
        else {
            throw bpo::validation_error(bpo::validation_error::invalid_option_value, "output-format");
        }
    }

    // --input or --input-file is required ...
    if (!optionMap_.count("input") && !optionMap_.count("input-file")) {
        throw bpo::error("--input or --inputFile is required");
    }
    // ... but they cannot both appear
    if (optionMap_.count("input") && optionMap_.count("input-file")) {
        throw bpo::error("--input and --input-file are mutually exclusive");
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
        return optionMap_["input"].as<std::string>();
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
    return outputFormat_;
}

std::string ProgramOptions::getInputFile() const
{
    if (optionMap_.count("input-file")) {
        return optionMap_["input-file"].as<std::string>();
    }
    else {
        return {};
    }
}
