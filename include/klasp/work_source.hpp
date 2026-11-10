#pragma once

// Where the work loop gets its next chunk from: WorkWindow (MPI RMA window on
// rank 0) or LedgerWorkSource (single process).

#include "klasp/work_counters.hpp"

#include <utility>
#include <vector>

namespace klasp {

class WorkSource {
public:
    virtual ~WorkSource() = default;

    // Claim the next chunk of the most-behind space group. ASSIGN with the
    // amount actually granted, WAIT if work is all claimed but some is still in
    // flight, DONE when every target is met or a stop has been requested.
    virtual Grab grab() = 0;

    // Report a finished chunk: `claimed` was the grant, `produced` the valids
    // actually optimized. Signed, so under-production reopens the shortfall.
    virtual void report(int sg, int claimed, int produced) = 0;

    // Close out a space group that can never produce (see ZeroYieldGuard).
    virtual void retire(int sg) = 0;

    // Claim a contiguous block of `n` generator seeds for a space group,
    // returning the first. The seeds for a space group always form the prefix
    // [0, total), so a landscape is reproducible across rank counts.
    virtual long long claim_seeds(int sg, int n) = 0;

    // Ask everyone to stop claiming work. The run's time budget and a caught
    // termination signal both come through here.
    virtual void request_stop() = 0;

    // How many participants have stopped claiming, as of the last grab().
    virtual int ranks_stopped() const = 0;

    // Total work still owed across every space group, as of the last grab().
    virtual long long last_remaining_total() const = 0;

    // This participant's preferred space groups (empty = no preference).
    virtual void set_home(std::vector<unsigned char> home) = 0;
};

// The single-process WorkSource, over WorkLedger.
class LedgerWorkSource : public WorkSource {
public:
    explicit LedgerWorkSource(WorkLedger ledger) : ledger_(std::move(ledger)) {}

    Grab grab() override {
        const Grab g = ledger_.grab();
        // Sampled at the grab, as WorkWindow does.
        remaining_total_ = ledger_.remaining_total();
        return g;
    }
    void report(int sg, int claimed, int produced) override {
        ledger_.report(sg, claimed, produced);
    }
    void retire(int sg) override { ledger_.retire(sg); }
    long long claim_seeds(int sg, int n) override { return ledger_.claim_seeds(sg, n); }
    void request_stop() override { ledger_.request_stop(); }
    int ranks_stopped() const override { return ledger_.stop_requested() ? 1 : 0; }
    long long last_remaining_total() const override { return remaining_total_; }
    void set_home(std::vector<unsigned char> home) override {
        ledger_.set_home(std::move(home));
    }

    WorkLedger& ledger() { return ledger_; }

private:
    WorkLedger ledger_;
    long long remaining_total_ = 0;
};

}  // namespace klasp
