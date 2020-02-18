#include <iostream>
#include <iomanip>
#include <boost/program_options.hpp>
#include <sudoku/grid.h>
#include <sudoku/partition_grid.h>
#include <sudoku/square.h>
#include <sudoku/metrics.h>

namespace bpo = boost::program_options;

struct GridOperation
{
    sudoku::CellCount cellPos;   // pos being set or cleared
    sudoku::CellValue cellValue; // set value if >0, clear value if ==0
};

std::vector<GridOperation> captureOperations(const sudoku::Dimensions& dims, size_t operationCount)
{
    std::vector<GridOperation> ops;
    sudoku::Grid grid(dims);
    sudoku::CellCount cellPos = 0;
    sudoku::CellValue minCellValue = 0;

    while (cellPos < dims.getCellCount()) {
        if (ops.size() >= operationCount) {
            break;
        }
        auto cellValue = grid.getCellPotential(cellPos).getNextAvailableValue(minCellValue);
        if (cellValue > 0) {
            ops.push_back({cellPos, cellValue});
            grid.setCellValue(cellPos, cellValue);
            cellPos++;
            minCellValue = 0;
            continue;
        }
        else {
            cellPos--;
            minCellValue = grid.getCellValue(cellPos);
            ops.push_back({cellPos, 0});
            grid.clearCellValue(cellPos);
            continue;
        }
    }

    return ops;
}

template <typename GridType>
void replayOperations(const char* name, GridType& grid, const std::vector<GridOperation>& ops)
{
    std::cout << std::setw(32) << std::left << name;

    auto start = sudoku::Metrics::now();
    for (auto op : ops) {
        if (op.cellValue) {
            grid.setCellValue(op.cellPos, op.cellValue);
        }
        else {
            grid.clearCellValue(op.cellPos);
        }
    }
    auto end = sudoku::Metrics::now();
    auto duration = end - start;

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    std::cout << " us\n";
}

template <typename PartitionFuncType>
void replayOnPartitionGrid(const char* name, const sudoku::Dimensions& dims,
                           PartitionFuncType partitionFunc,
                           sudoku::PartitionCount partitionCount,                           
                           const std::vector<GridOperation>& ops)
{
    auto partitionIds = partitionFunc(dims.getCellCount(), partitionCount);
    sudoku::PartitionTable partitionTable(dims, partitionCount, partitionIds);
    sudoku::PartitionGrid partitionGrid(dims, partitionTable);
    replayOperations(name, partitionGrid, ops);
}

bpo::variables_map parseCommandLine(int argc, char** argv)
{
    bpo::options_description desc("Compare performance of various Grid implementations");
    desc.add_options()
        ("help,h", "Show this usage and exit.")
        ("opcount,n", bpo::value<size_t>()->default_value(10000000),
                      "Number of operations to capture and replay")
        ("partitions,p", bpo::value<sudoku::PartitionCount>()->default_value(4),
                         "Number of partitions to use")
        ("rank,r", bpo::value<sudoku::CellCount>()->default_value(6),
                   "Size of sudoku (rank)")
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
    #if SUDOKU_CONFIG_IS_DEBUG == 1
    std::cerr << "WARNING! This is a Debug build!\n";
    #endif

    auto commandLine = parseCommandLine(argc, argv);

    sudoku::square::Dimensions dims(commandLine["rank"].as<sudoku::CellCount>());
    auto ops = captureOperations(dims, commandLine["opcount"].as<size_t>());
    std::cout << "Captured " << ops.size() << " ops. Replaying...\n";
    
    {
        sudoku::Grid grid(dims);
        replayOperations("Grid", grid, ops);
    }

    sudoku::PartitionCount partitionCount = commandLine["partitions"].as<sudoku::PartitionCount>();
    replayOnPartitionGrid("PartitionGrid(RR)", dims, sudoku::partitionRoundRobin, partitionCount, ops);
    replayOnPartitionGrid("PartitionGrid(RRRot)", dims, sudoku::partitionRoundRobinRotate, partitionCount, ops);
    replayOnPartitionGrid("PartitionGrid(RRTrail)", dims, sudoku::partitionRRTrail, partitionCount, ops);
}
