#include <iostream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <sudoku/solver.h>
#include <sudoku/standard.h>
#include <sudoku/square.h>
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
    },

    // From http://sudopedia.enjoysudoku.com/Valid_Test_Cases.html
    {
        "naked singles", standardDims,
        "305420810487901506029056374850793041613208957074065280241309065508670192096512408",
        "365427819487931526129856374852793641613248957974165283241389765538674192796512438"
    },
    {
        "hidden singles", standardDims,
        "002030008000008000031020000060050270010000050204060031000080605000000013005310400",
        "672435198549178362831629547368951274917243856254867931193784625486592713725316489"
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

// Each test case represents the number of peers to request.
std::vector<size_t> testCases_fork9x9{ 1, 3, 6, 8, 16 };

BOOST_DATA_TEST_CASE(Solver_fork9x9, testCases_fork9x9)
{
    // Initialize a solver and fork it!
    const size_t numPeers = sample;
    sudoku::square::Dimensions dims(3);
    std::vector<size_t> cellValues(dims.getCellCount(), 0);
    sudoku::Solver solver(dims, cellValues);
    auto peers = solver.fork(numPeers);
    
    // Test that we obtained the requested number of peers.
    // For some sudokus, this may not be possible, but an empty
    // 9x9 sudoku should not have a problem.
    BOOST_REQUIRE_EQUAL(peers.size(), numPeers);

    // Each peer (and the original solver) should all be solvable.
    BOOST_REQUIRE(solver.computeNextSolution());
    for (const auto& peer : peers) {
        BOOST_REQUIRE(peer->computeNextSolution());
    }

    // Each peer's solution should be unique from all the others.
    // Use a hashmap of (solution) -> (# occurances) to determine
    // if all solutions are unique.
    std::unordered_map<std::string, size_t> solutions;

    // Add original solver's solution to the hash map.
    std::string formatString(dims.getCellCount(), '0');
    sudoku::Formatter fmt(dims, formatString);
    solutions[fmt.format(solver.getCellValues())]++;

    // Add peers' solutions to the hash map.
    for (const auto& peer : peers) {
        std::string solution = fmt.format(peer->getCellValues());
        solutions[solution]++;
    }

    // Verify number of solution occurances is 1 for all solutions.
    for (auto it = solutions.begin(); it != solutions.end(); ++it) {
        size_t solutionOcurrances = it->second;
        BOOST_CHECK(solutionOcurrances == 1);
    }
}

BOOST_AUTO_TEST_CASE(Solver_halt)
{
    // Prepare a large sudoku
    sudoku::square::Dimensions dims(6);
    std::vector<size_t> cellValues(dims.getCellCount(), 0);
    sudoku::Solver solver(dims, cellValues);

    // Attempt to solve it in another thread. This should
    // take a while...
    std::thread thread([&]() {
        solver.computeNextSolution();
    });

    // Halt the solver on the main thread.
    solver.halt();

    // Now, computeNextSolution on the spawned thread should
    // exit gracefully, letting us join quickly.
    thread.join();
}
