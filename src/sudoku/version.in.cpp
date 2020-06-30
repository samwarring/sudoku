#include <sudoku/version.h>

namespace sudoku
{
    namespace version
    {
        std::string getVersion()     { return "@sudoku_VERSION@"; }
        std::string getDescription() { return "@sudoku_VERSION_DESC@"; }
        std::string getBranch()      { return "@sudoku_BRANCH@"; }
        std::string getCommitDate()  { return "@sudoku_COMMIT_DATE@"; }
        std::string getBuildDate()   { return "@sudoku_BUILD_DATE@"; }
    }
}