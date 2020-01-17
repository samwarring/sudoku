#include <sudoku/solver.h>

namespace sudoku
{
    Solver::Solver(const Dimensions& dims, std::vector<size_t> cellValues)
        : dims_(dims)
        , cellValues_(std::move(cellValues))
    {
        if (cellValues_.size() != dims_.getCellCount()) {
            throw SolverException("Number of initial values does not match dimensions");
        }
        initializeCellPotentials();
        validateCellValues();
    }

    bool Solver::computeNextSolution()
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        bool result = sequentialSolve();
        auto stopTime = std::chrono::high_resolution_clock::now();
        solutionDuration_ += (stopTime - startTime);
        return result;
    }

    void Solver::initializeCellPotentials()
    {
        cellPotentials_.reserve(dims_.getCellCount());
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            cellPotentials_.emplace_back(dims_.getMaxCellValue());
        }

        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            if (cellValues_[cellPos] > 0) {
                for (size_t groupNum : dims_.getGroupsForCell(cellPos)) {
                    for (size_t relatedCellPos : dims_.getCellsInGroup(groupNum)) {
                        cellPotentials_[relatedCellPos].block(cellValues_[cellPos]);
                    }
                }
            }
        }
    }

    void Solver::validateCellValues() const
    {
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            if (dims_.getMaxCellValue() < cellValues_[cellPos]) {
                throw SolverException("A cell value is out of range");
            }
        }

        for (size_t groupNum = 0; groupNum < dims_.getNumGroups(); ++groupNum) {
            std::vector<int> occurs(dims_.getMaxCellValue());
            for (size_t cellPos : dims_.getCellsInGroup(groupNum)) {
                size_t cellValue = cellValues_[cellPos];
                if (cellValue != 0) {
                    if (0 < occurs[cellValue - 1]) {
                        throw SolverException("A group contained a repeated value");
                    }
                    occurs[cellValue - 1]++;
                }
            }
        }
    }

    void Solver::pushGuess(size_t cellPos, size_t cellValue)
    {
        cellValues_[cellPos] = cellValue;
        for (size_t groupNum : dims_.getGroupsForCell(cellPos)) {
            for (size_t relatedCellPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedCellPos].block(cellValue);
            }
        }
        guesses_.push({ cellPos, cellValue });
        totalGuesses_++;
    }

    std::pair<size_t, size_t> Solver::popGuess()
    {
        auto prevGuess = guesses_.top();
        guesses_.pop();
        size_t cellPos = prevGuess.first;
        size_t cellValue = prevGuess.second;
        cellValues_[cellPos] = 0;
        for (size_t groupNum : dims_.getGroupsForCell(cellPos)) {
            for (size_t relatedCellPos : dims_.getCellsInGroup(groupNum)) {
                cellPotentials_[relatedCellPos].unblock(cellValue);
            }
        }
        totalBacktracks_++;
        return prevGuess;
    }

    size_t Solver::selectNextCell() const
    {
        size_t leadingPos = dims_.getCellCount();
        size_t leadingNumBlocked = 0;
        const size_t cellCount = dims_.getCellCount();
        const size_t maxCellValue = dims_.getMaxCellValue();
        for (size_t cellPos = 0; cellPos < cellCount; ++cellPos) {
            if (cellValues_[cellPos] == 0) {
                size_t amountBlocked = cellPotentials_[cellPos].getAmountBlocked();
                if (leadingNumBlocked <= amountBlocked) {
                    leadingPos = cellPos;
                    leadingNumBlocked = amountBlocked;
                    // Optimization: If we found an empty cell completely blocked,
                    // don't look at the rest of the cells.
                    if (leadingNumBlocked == maxCellValue) {
                        break;
                    }
                }
            }
        }
        return leadingPos;
    }

    bool Solver::sequentialSolve()
    {
        size_t cellPos = selectNextCell();
        size_t minCellValue = 0;

        // If already solved, pop the last guess.
        if (cellPos == dims_.getCellCount()) {
            if (guesses_.size() == 0) {
                // Nothing to pop. No solution.
                return false;
            }
            auto prevGuess = popGuess();
            cellPos = prevGuess.first;
            minCellValue = prevGuess.second;
        }

        while (cellPos != dims_.getCellCount()) {

            // Does this cell have any remaining potential values?
            size_t cellValue = cellPotentials_[cellPos].getNextAvailableValue(minCellValue);
            if (cellValue == 0) {
                // Backtrack
                if (guesses_.size() == 0) {
                    // Can't backtrack any further. No solution.
                    return false;
                }
                auto prevGuess = popGuess();
                cellPos = prevGuess.first;
                minCellValue = prevGuess.second;
                continue;
            }

            // We have an available value. Try it, and continue guessing.
            pushGuess(cellPos, cellValue);
            cellPos = selectNextCell();
            minCellValue = 0;
        }
        return true;
    }

    size_t Solver::selectForkCell()
    {
        // Make guesses for "obvious" cells until we encounter a cell
        // with multiple unblocked values. This is the "fork" cell.
        size_t cellPos = dims_.getCellCount();
        while (true) {
            cellPos = selectNextCell();
            if (cellPos == dims_.getCellCount()) {
                // Found a cell with no options; therefore, no solution.
                // In this case, there is no fork cell.
                return dims_.getCellCount();
            }
            if (cellPotentials_[cellPos].getAmountBlocked() == dims_.getMaxCellValue() - 1) {
                // Found a cell with only one possibility; don't fork here.
                // Make the guess and try the next one.
                pushGuess(cellPos, cellPotentials_[cellPos].getNextAvailableValue(0));
                continue;
            }
            // Found an empty cell with multiple unblocked values.
            break;
        }
        return cellPos;
    }

    std::vector<std::unique_ptr<Solver>> Solver::fork(size_t numPeers)
    {
        if (numPeers == 0) {
            // Nothing to do.
            return {};
        }

        // Select a cell with mulitple available values.
        size_t forkPos = selectForkCell();
        if (forkPos == dims_.getCellCount()) {
            // Already solved, or no solution.
            return {};
        }

        // Get the available values.
        std::vector<size_t> availableValues;
        Potential& potential = cellPotentials_[forkPos];
        for (size_t cellValue = 1; cellValue <= dims_.getMaxCellValue(); ++cellValue) {
            if (!cellPotentials_[forkPos].isBlocked(cellValue)) {
                availableValues.push_back(cellValue);
            }
        }

        if (numPeers + 1 < availableValues.size()) {
            return forkManyValuesPerPeer(forkPos, availableValues, numPeers);
        }
        else {
            // For now, don't attempt to make more solvers than we have available values.
            return forkOneValuePerPeer(forkPos, availableValues);
        }
    }

    std::vector<std::unique_ptr<Solver>> Solver::forkOneValuePerPeer(
        size_t forkPos,
        const std::vector<size_t>& availableValues)
    {
        // If we have enough solvers (numSolvers == numAvailableValues), then
        // we can give each solver it's own value.

        // Make the peers.
        std::vector<std::unique_ptr<Solver>> peerSolvers;
        peerSolvers.reserve(availableValues.size() - 1);
        size_t availableValue = 0;
        for (size_t peerNum = 0; peerNum < availableValues.size() - 1; ++peerNum) {
            // Each peer starts with the same values as the current solver, except
            // they each have a different value for the fork cell.
            std::vector<size_t> newCellValues = cellValues_;
            availableValue = cellPotentials_[forkPos].getNextAvailableValue(availableValue);
            newCellValues[forkPos] = availableValue;
            peerSolvers.push_back(std::make_unique<Solver>(dims_, std::move(newCellValues)));
        }

        // This solver gets the last available value. That way, it skips over the
        // first ones that were assigned to the peers.
        availableValue = cellPotentials_[forkPos].getNextAvailableValue(availableValue);
        pushGuess(forkPos, availableValue);

        return peerSolvers;
    }

    std::vector<std::unique_ptr<Solver>> Solver::forkManyValuesPerPeer(
        size_t forkPos,
        const std::vector<size_t>& availableValues,
        size_t numPeers)
    {
        // If we have more potential values than requested peers, then at
        // least one peer needs to be responsible for multiple values. For
        // example, if the fork cell has 5 available values and the caller
        // requests 1 peer, the current solver "represses" values 1,2,3 in
        // the fork position, while the peer "represses" values 4,5. We can
        // repress by marking it as "blocked".

        // Make all the peers. Also group them in a contiguous array
        // of pointers with the current solver.
        const size_t numSolvers = numPeers + 1;
        std::vector<std::unique_ptr<Solver>> peerSolvers;
        std::vector<Solver*> allSolvers;
        peerSolvers.reserve(numPeers);
        allSolvers.reserve(numSolvers);
        allSolvers.push_back(this);
        for (size_t peerNum = 0; peerNum < numPeers; ++peerNum) {
            auto newPeer = std::make_unique<Solver>(dims_, cellValues_);
            allSolvers.push_back(newPeer.get());
            peerSolvers.push_back(std::move(newPeer));
        }
        // Loop through the available values and assign them in round-robin order.
        size_t curSolverNum = 0;
        for (size_t availableValue : availableValues) {
            
            // Repress this value on all peers...
            for (size_t solverNum = 0; solverNum < allSolvers.size(); ++solverNum) {
                if (solverNum == curSolverNum) {
                    // ...except the current peer.
                    continue;
                }
                allSolvers[solverNum]->cellPotentials_[forkPos].block(availableValue);
            }

            // The next available value will be assigned to the next peer.
            curSolverNum = (curSolverNum + 1) % allSolvers.size();
        }

        // allSolvers goes out of scope, but does not own memory. We're ok.
        return peerSolvers;
    }

}
