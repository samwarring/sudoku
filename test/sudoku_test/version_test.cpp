#include <string>
#include <vector>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <sudoku/sudoku.h>

static std::vector<std::string> STRING_VALUES{
    sudoku::version::getVersion(),
    sudoku::version::getDescription(),
    sudoku::version::getBranch(),
    sudoku::version::getCommitDate(),
    sudoku::version::getBuildDate()
};

BOOST_DATA_TEST_CASE(NoUnexpandedCMakeVariables, STRING_VALUES)
{
    BOOST_REQUIRE_EQUAL(sample.find_first_of('@'), std::string::npos);
    BOOST_REQUIRE_EQUAL(sample.find_first_of('$'), std::string::npos);
}