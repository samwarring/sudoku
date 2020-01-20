#include <sudoku/fork.h>

namespace sudoku
{
    /**
     * Trivially solve a grid until one of the conditions are met:
     * 1. The grid is full
     * 2. The grid contains an empty cell that's completely blocked.
     * 3. The maxBlockEmptyCell has more than one available value.
     * 
     * \return a cell position for each case above:
     *         1. pos == grid dimensions cell count
     *         2. pos of completely blocked cell
     *         3. pos of maxBlockEmptyCell with multiple available values.
     */
    static size_t simplify(Grid& grid)
    {
        const Dimensions& dims = grid.getDimensions();
        size_t cellPos = grid.getMaxBlockEmptyCell();
        while (cellPos != dims.getCellCount()) {
            
            const auto& potential = grid.getCellPotential(cellPos);
            auto blockCount = potential.getAmountBlocked();
            if (blockCount == dims.getMaxCellValue()) {
                // Found a completely blocked cell.
                return cellPos;
            }
            else if (blockCount == dims.getMaxCellValue() - 1) {
                // Found an "obvious" empty cell with exactly one available value
                // Assume this value, and try the next empty cell.
                grid.setCellValue(cellPos, potential.getNextAvailableValue(0));
                cellPos = grid.getMaxBlockEmptyCell();
                continue;
            }

            // Found an empty cell with at least two available values.
            return cellPos;
        }

        // No more empty cells. The grid is full
        return dims.getCellCount();
    }

    /**
     * Construct new grids for each available value at the fork position
     */
    static std::vector<Grid> forkOneAvailableValuePerPeer(
        const Grid& grid,
        size_t forkPos,
        const std::vector<size_t>& availableValues)
    {
        std::vector<Grid> result;
        result.reserve(availableValues.size());
        for (auto availableValue : availableValues) {
            Grid newGrid = grid;
            newGrid.setCellValue(forkPos, availableValue);
            result.emplace_back(std::move(newGrid));
        }
        return result;
    }

    /**
     * Construct peers where each is responsible for several available
     * values at a single position.
     */
    static std::vector<Grid> forkMoreAvailableValuesThanPeers(
        const Grid& grid,
        size_t peerCount,
        size_t forkPos,
        const std::vector<size_t>& availableValues)
    {
        // Create a grid for each peer.
        std::vector<Grid> result(peerCount, grid);

        // Assign the available values in round-robin order.
        size_t currentPeer = 0;
        for (auto availableValue : availableValues) {
            
            // Assign the available value by blocking that value on all peers
            // except the peer being assigned.
            for (size_t otherPeer = 0; otherPeer < peerCount; ++otherPeer) {
                if (otherPeer != currentPeer) {
                    result[otherPeer].restrictCellValue(forkPos, availableValue);
                }
            }

            // Next available value assigned in round-robin order.
            currentPeer = (currentPeer + 1) % peerCount;
        }
        return result;
    }

    /**
     * Recursively generate peers until we reach the requested peer count,
     * or until the sudoku is solved, or until we find a completely blocked
     * cell.
     */
    static std::vector<Grid> forkMorePeersThanAvailableValues(
        const Grid& grid,
        size_t peerCount,
        size_t forkPos,
        const std::vector<size_t>& availableValues)
    {
        // Create peers from the current fork position.
        auto firstPeers = forkOneAvailableValuePerPeer(grid, forkPos, availableValues);

        // Fork each of the firstPeers evenly. Each firstPeer will be forked into
        // roughly 'quotient' recursive peers.
        const size_t quotient = peerCount / firstPeers.size();
        const size_t remainder = peerCount % firstPeers.size();

        // Use a lambda to append to the result vector.
        std::vector<Grid> result;
        auto appendPeers = [&](auto& peers) {
            result.reserve(result.size() + peers.size());
            for (auto& peer : peers) {
                result.emplace_back(std::move(peer));
            }
        };

        // Fork each of the firstPeers.
        for (size_t i = 0; i < firstPeers.size(); ++i) {
            size_t recursivePeerCount = quotient;
            if (i < remainder) {
                recursivePeerCount++;
            }
            auto recursivePeers = fork(firstPeers[i], recursivePeerCount);
            appendPeers(recursivePeers);
        }

        // All recursive peers are in the result.
        return result;
    }

    std::vector<Grid> fork(Grid grid, size_t peerCount)
    {
        // Simplify the original grid.
        size_t forkPos = simplify(grid);

        if (forkPos == grid.getDimensions().getCellCount()) {
            // The grid is already solved. Can't fork. Just return return the
            // original grid.
            return { std::move(grid) };
        }

        // Get the available values for the fork position.
        auto availableValues = grid.getCellPotential(forkPos).getAvailableValues();

        if (availableValues.size() == 0) {
            // The fork position is completely blocked. Can't fork. Just return
            // the original grid.
            return { std::move(grid) };
        }
        else if (availableValues.size() < peerCount) {
            return forkMorePeersThanAvailableValues(grid, peerCount, forkPos, availableValues);
        }
        else if (availableValues.size() == peerCount) {
            return forkOneAvailableValuePerPeer(grid, forkPos, availableValues);
        }
        else { // availableValues.size() > peerCount
            return forkMoreAvailableValuesThanPeers(grid, peerCount, forkPos, availableValues);
        }
    }
}