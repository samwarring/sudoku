#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <sudoku/partitions.h>
#include <sudoku/square.h>

namespace bpo = boost::program_options;
using PartitionList = std::vector<sudoku::PartitionCount>;

namespace std
{
    /**
     * Required to use vector<T> as boost program options parameter
     */
    template <typename T>
    std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec)
    {
        out << "[ ";
        for (auto item : vec) {
            out << item << ' ';
        }
        out << ']';
        return out;
    }
}

/**
 * Prints a column value to the table.
 */
template <typename T>
void printCol(T value, size_t width = 8)
{
    std::cout << std::setw(width) << value;
}

/**
 * Prints single row of header columns.
 */
void printHeader()
{
    printCol("Rank");
    printCol("#Cells");
    printCol("#Parts");
    printCol("RR");
    printCol("RRRot");
    printCol("Rand");
    printCol("Diag");
    printCol("RRTrail");
    printCol("Best");
    printCol("BestVal");
    std::cout << '\n';
}

/**
 * Evaluates the "effectiveness" of a partition for the given dimensions.
 * 
 * A partition is effective if for any given cell, the related cells are
 * evenly partitioned. To calculate the effectiveness of the given partition
 * and dimensions, this function iterates through every cell. For each cell,
 * it finds the related cells and counts how many belong to each partition.
 * The "disparity" at that cell is the size difference between the most
 * and least common partitions.
 * 
 * \return the greatest "disparity" found among all cells. A lower value
 *         means the given partition is more "effective" for the given
 *         dimensions.
 */
sudoku::CellCount computePartitionDisparity(const sudoku::Dimensions& dims,
                                            std::vector<sudoku::PartitionCount> partitions,
                                            sudoku::PartitionCount partitionCount)
{
    sudoku::CellCount maxDisparity = 0;

    for (sudoku::CellCount cellPos = 0; cellPos < dims.getCellCount(); ++cellPos) {
        std::vector<sudoku::CellCount> cellsPerPartition(partitionCount);
        for (auto relatedPos : dims.getRelatedCells(cellPos)) {
            cellsPerPartition[partitions[relatedPos]]++;
        }
        auto minCellCount = *std::min_element(cellsPerPartition.begin(), cellsPerPartition.end());
        auto maxCellCount = *std::max_element(cellsPerPartition.begin(), cellsPerPartition.end());
        sudoku::CellCount disparity = maxCellCount - minCellCount;
        if (disparity > maxDisparity) {
            maxDisparity = disparity;
        }
    }

    return maxDisparity;
}

bpo::variables_map parseCommandLine(int argc, char** argv)
{
    bpo::options_description desc("Tests the effectiveness of sudoku::partition functions");
    desc.add_options()
        ("help,h", "Show usage and exit")
        ("min-rank,m", bpo::value<sudoku::CellCount>()->default_value(3),
                       "Minimum rank of square sudoku")
        ("max-rank,M", bpo::value<sudoku::CellCount>()->default_value(10),
                       "Maximum rank of square sudoku")
        ("partitions,p", bpo::value<PartitionList>()->multitoken()->default_value({2, 4, 8, 16}),
                         "All partition counts to check")
        ; // end of options

    bpo::variables_map result;
    try {
        bpo::store(bpo::parse_command_line(argc, argv, desc), result);
    }
    catch (const bpo::error& err) {
        std::cerr << "error: " << err.what() << '\n';
        exit(1);
    }

    if (result.count("help")) {
        std::cout << desc;
        exit(0);
    }

    bpo::notify(result);
    return result;
}

int main(int argc, char** argv)
{
    auto options = parseCommandLine(argc, argv);

    sudoku::CellCount minRank = options["min-rank"].as<sudoku::CellCount>();
    sudoku::CellCount maxRank = options["max-rank"].as<sudoku::CellCount>();
    PartitionList partitionCounts = options["partitions"].as<PartitionList>();
    std::cout << "Testing effectiveness of sudoku::partition functions...\n";
    std::cout << "Using square sudokus ranked " << minRank << '-' << maxRank << '\n';
    std::cout << "Using partion sizes " << partitionCounts << '\n';
    std::cout << "Lower score is better.\n\n";

    printHeader();
    
    for (auto rank = minRank; rank <= maxRank; ++rank) {
        for (auto partitionCount : partitionCounts) {
            sudoku::square::Dimensions dims(rank);
            std::map<std::string, sudoku::CellCount> disparities;
            printCol(rank);
            printCol(dims.getCellCount());
            printCol(partitionCount);

            auto roundRobinPartitions = sudoku::partitionRoundRobin(dims.getCellCount(), partitionCount);
            auto roundRobinDisparity = computePartitionDisparity(dims, roundRobinPartitions, partitionCount);
            disparities["RR"] = roundRobinDisparity;
            printCol(roundRobinDisparity);

            auto roundRobinRotatePartitions = sudoku::partitionRoundRobinRotate(dims.getCellCount(), partitionCount);
            auto roundRobinRotateDisparity = computePartitionDisparity(dims, roundRobinRotatePartitions, partitionCount);
            disparities["RRRot"] = roundRobinRotateDisparity;
            printCol(roundRobinRotateDisparity);

            auto randomPartitions = sudoku::partitionRandom(dims.getCellCount(), partitionCount);
            auto randomDisparity = computePartitionDisparity(dims, randomPartitions, partitionCount);
            disparities["Rand"] = randomDisparity;
            printCol(randomDisparity);

            auto diagonalPartitions = sudoku::partitionDiagonal(dims.getMaxCellValue(), dims.getMaxCellValue(), partitionCount);
            auto diagonalDisparity = computePartitionDisparity(dims, diagonalPartitions, partitionCount);
            disparities["Diag"] = diagonalDisparity;
            printCol(diagonalDisparity);

            auto rrtPartitions = sudoku::partitionRRTrail(dims.getCellCount(), partitionCount);
            auto rrtDisparity = computePartitionDisparity(dims, rrtPartitions, partitionCount);
            disparities["RRTrail"] = rrtDisparity;
            printCol(rrtDisparity);

            auto bestElem = std::min_element(disparities.begin(), disparities.end(),
                                             [](auto a, auto b) { return a.second < b.second; });
            printCol(bestElem->first);
            printCol(bestElem->second);

            std::cout << '\n';
        }
    }

    return 0;
}
