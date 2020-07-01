#ifndef INCLUDED_SUDOKU_VERSION_H
#define INCLUDED_SUDOKU_VERSION_H

#include <string>

namespace sudoku
{
    namespace version
    {
        std::string getVersion();
        std::string getDescription();
        std::string getBranch();
        std::string getCommitDate();
        std::string getBuildDate();
    }
}

#endif