#include <sudoku/solver.h>

namespace sudoku
{
    Solver::Solver(const Dimensions& dims, std::vector<size_t> cellValues)
        : dims_(dims)
        , cellValues_(std::move(cellValues))
        , grid_(dims)
        , haltEvent_(false)
    {
        if (cellValues_.size() != dims_.getCellCount()) {
            throw SolverException("Number of initial values does not match dimensions");
        }
        initializeCellPotentials();
        validateCellValues();
        
        // Check if already solved. If so, the first call to computeNextSolution
        // should return true.
        if (selectNextCell() == dims_.getCellCount()) {
            unreportedSolution_ = true;
        }
    }

    bool Solver::computeNextSolution()
    {
        auto startTime = Metrics::now();
        bool result = sequentialSolve();
        auto stopTime = Metrics::now();
        metrics_.duration += (stopTime - startTime);
        return result;
    }

    void Solver::initializeCellPotentials()
    {
        for (size_t cellPos = 0; cellPos < dims_.getCellCount(); ++cellPos) {
            if (cellValues_[cellPos] > 0) {
                grid_.setCellValue(cellPos, cellValues_[cellPos]);
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
        grid_.setCellValue(cellPos, cellValue);
        guesses_.push({ cellPos, cellValue });
        metrics_.totalGuesses++;
    }

    std::pair<size_t, size_t> Solver::popGuess()
    {
        auto prevGuess = guesses_.top();
        guesses_.pop();
        size_t cellPos = prevGuess.first;
        size_t cellValue = prevGuess.second;
        grid_.clearCellValue(cellPos, cellValue);
        cellValues_[cellPos] = 0;
        metrics_.totalBacktracks++;
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
                size_t amountBlocked = grid_.getCellPotential(cellPos).getAmountBlocked();
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
        if (unreportedSolution_) {
            // We previously found a solution without sequentialSolve().
            // Future calls to sequentialSolve() will continue searching,
            // but not this one.
            unreportedSolution_ = false;
            return true;
        }

        size_t cellPos = selectNextCell();
        size_t minCellValue = 0;

        // If already solved, pop the last guess.
        if (cellPos == dims_.getCellCount()) {
            if (guesses_.size() == 0) {
                // Nothing to pop. No more solutions.
                return false;
            }
            auto prevGuess = popGuess();
            cellPos = prevGuess.first;
            minCellValue = prevGuess.second;
        }

        while (cellPos != dims_.getCellCount()) {
            
            // Check if the solver should stop searching.
            // TODO: If performance becomes an issue, a couple ideas:
            //       1. examine other memory orders (e.g. std::memory_order_relaxed)
            //       2. don't check the event for _every_ iteration of the loop.
            if (haltEvent_.load()) {
                return false;
            }

            // Does this cell have any remaining potential values?
            size_t cellValue = grid_.getCellPotential(cellPos).getNextAvailableValue(minCellValue);
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
                // No more empty cells. The sudoku is already solved.
                // Mark the solution as un-reported so it can be returned
                // from the next call to computeNextSolution().
                unreportedSolution_ = true;
                return dims_.getCellCount();
            }
            if (grid_.getCellPotential(cellPos).getAmountBlocked() == dims_.getMaxCellValue()) {
                // Found a completely-blocked cell. No solution.
                return dims_.getCellCount();
            }
            if (grid_.getCellPotential(cellPos).getAmountBlocked() == dims_.getMaxCellValue() - 1) {
                // Found a cell with only one possibility; don't fork here.
                // Make the guess and try the next one.
                pushGuess(cellPos, grid_.getCellPotential(cellPos).getNextAvailableValue(0));
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
        for (size_t cellValue = 1; cellValue <= dims_.getMaxCellValue(); ++cellValue) {
            if (!grid_.getCellPotential(forkPos).isBlocked(cellValue)) {
                availableValues.push_back(cellValue);
            }
        }

        if (numPeers + 1 < availableValues.size()) {
            return forkManyValuesPerPeer(forkPos, availableValues, numPeers);
        }
        else if (numPeers + 1 == availableValues.size()) {
            return forkOneValuePerPeer(forkPos, availableValues);
        }
        else {
            return forkMorePeersThanValues(forkPos, availableValues, numPeers);
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
            availableValue = grid_.getCellPotential(forkPos).getNextAvailableValue(availableValue);
            newCellValues[forkPos] = availableValue;
            peerSolvers.push_back(std::make_unique<Solver>(dims_, std::move(newCellValues)));
        }

        // This solver gets the last available value. That way, it skips over the
        // first ones that were assigned to the peers.
        availableValue = grid_.getCellPotential(forkPos).getNextAvailableValue(availableValue);
        pushGuess(forkPos, availableValue);

        // This guess might have solved the sudoku. Let's make sure.
        if (selectNextCell() == dims_.getCellCount()) {
            unreportedSolution_ = true;
        }

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
                allSolvers[solverNum]->grid_.restrictCellValue(forkPos, availableValue);
            }

            // The next available value will be assigned to the next peer.
            curSolverNum = (curSolverNum + 1) % allSolvers.size();
        }

        // allSolvers goes out of scope, but does not own memory. We're ok.
        return peerSolvers;
    }

    
    std::vector<std::unique_ptr<Solver>> Solver::forkMorePeersThanValues(
        size_t forkPos,
        const std::vector<size_t>& availableValues,
        size_t numPeers)
    {
        // We have more peers than available values. (e.g. solving a
        // 9x9 sudoku with 25 peers). 

        // Fork peers for all available values
        auto peers = forkOneValuePerPeer(forkPos, availableValues);
        auto remainingNumPeers = numPeers - peers.size();

        // Will use this lambda to extend one vector of peers with another.
        auto extendPeers = [](auto& peers, auto& morePeers) {
            for (size_t i = 0; i < morePeers.size(); ++i) {
                peers.emplace_back(std::move(morePeers[i]));
            }
        };
        
        // Each of the peers we just created can themselves be forked to
        // make up the rest. Divide the remaining (aka "recursive") peers
        // as evenly as possible.
        auto quotient = remainingNumPeers / (peers.size() + 1);
        auto remainder = remainingNumPeers % (peers.size() + 1);

        // Fork *this solver first.
        auto thisNumRecursivePeers = quotient;
        if (remainder > 0) {
            thisNumRecursivePeers++;
        }
        auto allRecursivePeers = fork(thisNumRecursivePeers);
        allRecursivePeers.reserve(remainingNumPeers);
        
        // Fork the original peers next.
        for (size_t peerNum = 0; peerNum < peers.size(); ++peerNum) {
            auto numRecursivePeers = quotient;
            if (peerNum + 1 < remainder) {
                numRecursivePeers++;
            }
            auto recursivePeers = peers[peerNum]->fork(numRecursivePeers);
            extendPeers(allRecursivePeers, recursivePeers);
        }

        // Return a vector containing the initial peers and recursive peers.
        extendPeers(peers, allRecursivePeers);
        return peers;
    }
}
