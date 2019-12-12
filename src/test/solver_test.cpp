#include <iostream>
#include <stdexcept>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <sudoku/solver.h>
#include <sudoku/standard.h>
#include <sudoku/cell_value_parser.h>
#include "util.h"

sudoku::standard::Dimensions standardDims;
sudoku::Dimensions simple4{4, 4, { {0, 1, 2, 3} } };

struct TestCase
{
    const char* name = "default";
    const sudoku::Dimensions& dims;
    const char* inputValueString = "";
    const char* expectedValueString = "";
};

std::ostream& operator<<(std::ostream& ostr, const TestCase& tc)
{
    return ostr << tc.name;
}

std::vector<TestCase> testCases{
    {"simple4_1230",      simple4, "1230", "1234"},
    {"simple4_4103",      simple4, "4103", "4123"},
    {"simple4_0000_1234", simple4, "0000", "1234"},
    {"simple4_0000_3142", simple4, "0000", "3142"},
    {"simple4_0000_4321", simple4, "0000", "4321"},

    // From http://elmo.sbs.arizona.edu/sandiway/sudoku/examples.html
    {
        "wildcatjan17", standardDims,
        "000260701 680070090 190004500"
        "820100040 004602900 050003028"
        "009300074 040050036 703018000"
        ,
        "435269781 682571493 197834562"
        "826195347 374682915 951743628"
        "519326874 248957136 763418259"
    },
    {
        "wildcat18", standardDims,
        "100489006 730000040 000001295"
        "007120600 500703008 006095700"
        "914600000 020000037 800512004"
        ,
        "152489376 739256841 468371295"
        "387124659 591763428 246895713"
        "914637582 625948137 873512964"
    },
    {
        "dtfeb19", standardDims,
        "020608000 580009700 000040000"
        "370000500 600000004 008000013"
        "000020000 009800036 000306090"
        ,
        "123678945 584239761 967145328"
        "372461589 691583274 458792613"
        "836924157 219857436 745316892"
    },
    {
        "v2155141", standardDims,
        "000600400 700003600 000091080"
        "000000000 050180003 000306045"
        "040200060 903000000 020000100"
        ,
        "581672439 792843651 364591782"
        "438957216 256184973 179326845"
        "845219367 913768524 627435198"
    },
    {
        "challenge2", standardDims,
        "200300000 804062003 013800200"
        "000020390 507000621 032006000"
        "020009140 601250809 000001002"
        ,
        "276314958 854962713 913875264"
        "468127395 597438621 132596487"
        "325789146 641253879 789641532"
    },
    {
        "challenge1", standardDims,
        "020000000 000600003 074080000"
        "000003002 080040010 600500000"
        "000010780 500009000 000000040"
        ,
        "126437958 895621473 374985126"
        "457193862 983246517 612578394"
        "269314785 548769231 731852649"
    }
};

BOOST_DATA_TEST_CASE(Solver_testCases, testCases)
{
    const TestCase& tc = sample;
    auto inputValues = sudoku::parseCellValues(tc.dims, tc.inputValueString);
    auto expectedValues = sudoku::parseCellValues(tc.dims, tc.expectedValueString);
    sudoku::Solver solver(tc.dims, std::move(inputValues));
    while (solver.computeNextSolution()) {
        // Found a solution, but is it the solution we're looking for?
        if (solver.getCellValues() == expectedValues) {
            return;
        }
    }
    BOOST_REQUIRE(!"Did not find a matching solution");
}
