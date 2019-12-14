#include <boost/test/unit_test.hpp>
#include <sudoku/formatter.h>
#include <sudoku/dimensions.h>
#include <vector>

BOOST_AUTO_TEST_CASE(Formatter_2x2)
{
    sudoku::Dimensions dims(4, 4, {});
    const char* formatString = "\n"
                               "0 | 0\n"
                               "--+--\n"
                               "0 | 0\n";
    sudoku::Formatter fmt(dims, formatString);

    std::vector<size_t> cellValues {1, 2, 3, 0};
    const char* expectedString = "\n"
                                 "1 | 2\n"
                                 "--+--\n"
                                 "3 | 0\n";
    std::string outputString = fmt.format(cellValues);
    BOOST_REQUIRE_EQUAL(outputString, expectedString);
}

BOOST_AUTO_TEST_CASE(Formatter_customPlaceholder)
{
    sudoku::Dimensions dims(4, 2, {});
    std::vector<size_t> cellValues{ 0, 1, 2, 1 };
    const char* formatString = "\\..*.*.(:>).*.*../";
    const char* placeholders = "*";
    sudoku::Formatter fmt(dims, formatString, placeholders);
    std::string outputString = fmt.format(cellValues);
    BOOST_REQUIRE_EQUAL(outputString, "\\..0.1.(:>).2.1../");
}
