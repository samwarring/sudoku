#ifndef INCLUDED_SUDOKU_CLI_PROGRAM_OPTIONS_H
#define INCLUDED_SUDOKU_CLI_PROGRAM_OPTIONS_H

#include <string>
#include <boost/program_options.hpp>

class ProgramOptions
{
    public:

        enum class OutputFormat
        {
            PRETTY,
            SERIAL
        };

        ProgramOptions(int argc, char** argv);

        std::string getDescription() const;

        bool isHelp() const;

        std::string getInput() const;

        size_t getNumSolutions() const;

        OutputFormat getOutputFormat() const;

        std::string getInputFile() const;

        size_t getSquareDimensionRoot() const;

        size_t getThreadCount() const;

        size_t getForkCount() const;

        bool isEchoInput() const;

    private:
    
        boost::program_options::options_description description_;
        boost::program_options::variables_map optionMap_;
        OutputFormat outputFormat_;
};

#endif
