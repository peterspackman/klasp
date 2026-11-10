#pragma once
// MPI RMA work counters. A single window on rank 0 holds 4N+1 int64 counters:
//   [0  .. N)   avail[sg]    (init target[sg])  structures still to claim
//   [N .. 2N)   inflight[sg] (init 0)           claimed but not yet reported
//   [2N .. 3N)  seed[sg]     (init 0)           next free generator seed
//   [3N .. 4N)  remaining[sg](init target[sg])  structures still to produce
//   [4N]        stop         (init 0)           nonzero: claim no more work
// Every rank claims and reports by atomic Fetch_and_op on rank 0's memory
// (passive-target RMA). Accounting matches WorkLedger (work_counters.hpp).
//
// Termination reads `remaining` alone: reads are atomic per element only, and
// `remaining` only decreases, so a stale read can give a spurious WAIT but never
// a false DONE. grab() must increment inflight before decrementing avail.

#include "klasp/work_source.hpp"

#include <mpi.h>
#include <vector>
#include <cstddef>

namespace klasp {

class WorkWindow : public WorkSource {
public:
    // `seed_start` (optional, per SG) is where each generator sequence starts:
    // 0 for a fresh run, the resume watermark otherwise. `batch_sizes` is the
    // per-SG floor for the tail split (see chunk_for); empty means no floor.
    WorkWindow(const std::vector<long long>& targets,
               const std::vector<int>& unit_sizes, int rank, int nranks,
               const std::vector<long long>& seed_start = {},
               const std::vector<int>& batch_sizes = {})
        : n_(static_cast<int>(targets.size())),
          unit_(unit_sizes),
          batch_(batch_sizes),
          snap_(2 * n_, 0),
          rem_(n_, 0) {
        batch_.resize(n_, 0);
        (void)nranks;
        for (auto& u : unit_) if (u < 1) u = 1;
        const int slots = 4 * n_ + 1;   // +1: the shared stop flag
        MPI_Aint bytes = (rank == 0)
            ? static_cast<MPI_Aint>(slots) * static_cast<MPI_Aint>(sizeof(long long))
            : 0;
        MPI_Win_allocate(bytes, sizeof(long long), MPI_INFO_NULL,
                         MPI_COMM_WORLD, &base_, &win_);
        if (rank == 0) {
            for (int i = 0; i < n_; ++i) {
                base_[i] = targets[i];   // avail
                base_[n_ + i] = 0;       // inflight
                base_[2 * n_ + i] =
                    i < static_cast<int>(seed_start.size()) ? seed_start[i] : 0;
                // Same targets as avail, which on resume are the remainder.
                base_[3 * n_ + i] = targets[i];   // remaining
            }
            base_[4 * n_] = 0;   // stop
        }
        MPI_Barrier(MPI_COMM_WORLD);          // init visible before any RMA epoch
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
    }

    ~WorkWindow() override {
        MPI_Win_unlock_all(win_);
        MPI_Win_free(&win_);
    }

    WorkWindow(const WorkWindow&) = delete;
    WorkWindow& operator=(const WorkWindow&) = delete;

    // This rank's preferred space groups (empty = no preference). Rank-local;
    // the atomic claim settles who gets the work.
    void set_home(std::vector<unsigned char> home) override { home_ = std::move(home); }

    // Ask every rank to stop claiming work; each rank polls the flag in grab().
    void request_stop() override { fao_(4 * n_, 1); }

    // Atomic read of the flag via a zero-delta Fetch_and_op.
    bool stop_requested() {
        last_stop_count_ = static_cast<int>(fao_(4 * n_, 0));
        return last_stop_count_ > 0;
    }

    // Ranks that have stopped claiming work, as of the last stop_requested()
    // (each rank increments the stop slot at most once).
    int ranks_stopped() const override { return last_stop_count_; }

    // Total work still owed across every space group, as of the last read.
    long long last_remaining_total() const override {
        long long t = 0;
        for (long long v : rem_) t += v > 0 ? v : 0;
        return t;
    }

    // Claim the next chunk of the most-behind SG. ASSIGN with the amount granted,
    // WAIT if everything is claimed but some is in flight, DONE when every target
    // is met or a stop has been requested.
    Grab grab() override {
        if (stop_requested()) return {GrabKind::DONE, -1, 0};
        snapshot_();
        const int sg = home_.empty() ? choose_sg(snap_.data(), n_)
                                     : choose_sg_home(snap_.data(), n_, home_.data());
        if (sg < 0) {                          // nothing claimable right now
            read_remaining_();
            for (int i = 0; i < n_; ++i)
                if (rem_[i] > 0) return {GrabKind::WAIT, -1, 0};
            return {GrabKind::DONE, -1, 0};
        }
        const int want = chunk_for(snap_[sg], unit_[sg], batch_[sg]);
        if (want <= 0) return {GrabKind::WAIT, -1, 0};

        fao_(n_ + sg, want);                   // tentative inflight (before avail--)
        const long long prev = fao_(sg, -static_cast<long long>(want));
        long long got = want;
        if (got > prev) got = prev < 0 ? 0 : prev;
        const long long over = want - got;
        if (over > 0) {
            fao_(sg, over);                    // return the over-claim to avail
            fao_(n_ + sg, -over);              // correct tentative inflight down to got
        }
        if (got <= 0) return {GrabKind::WAIT, -1, 0};
        return {GrabKind::ASSIGN, sg, static_cast<int>(got)};
    }

    // Report a finished chunk: `claimed` granted, `produced` valids optimized.
    // Signed (see WorkLedger::report): under-production reopens, over-production
    // credits the surplus toward the target.
    void report(int sg, int claimed, int produced) override {
        fao_(sg, static_cast<long long>(claimed - produced));  // signed: reopen / credit
        fao_(n_ + sg, -static_cast<long long>(claimed));       // drop inflight
        // The termination counter, independent of the pair above.
        fao_(3 * n_ + sg, -static_cast<long long>(produced));
    }

    // Close out a space group that can never produce (see ZeroYieldGuard).
    // One MPI_REPLACE per slot, so no agreement with a concurrent claim is needed.
    void retire(int sg) override {
        const long long zero = 0;
        MPI_Accumulate(&zero, 1, MPI_LONG_LONG, 0, sg, 1, MPI_LONG_LONG,
                       MPI_REPLACE, win_);
        MPI_Accumulate(&zero, 1, MPI_LONG_LONG, 0, 3 * n_ + sg, 1, MPI_LONG_LONG,
                       MPI_REPLACE, win_);
        MPI_Win_flush(0, win_);
    }

    // Claim a contiguous block of `n` seeds for an SG, returning the first. Seeds
    // are a global per-SG cursor into the quasi-random sequence, so the seed set
    // is independent of rank count. A resumed run starts at the database's highest
    // seed plus one; seeds lost to a killed rank below that are not reissued.
    long long claim_seeds(int sg, int n) override {
        return fao_(2 * n_ + sg, static_cast<long long>(n));
    }

private:
    // The remaining block. Each element is independently monotone, so no
    // cross-element consistency is needed.
    void read_remaining_() {
        MPI_Get_accumulate(nullptr, 0, MPI_LONG_LONG,
                           rem_.data(), n_, MPI_LONG_LONG,
                           0, 3 * n_, n_, MPI_LONG_LONG, MPI_NO_OP, win_);
        MPI_Win_flush(0, win_);
    }

    // Atomic-per-element read of the 2N avail/inflight counters (Get_accumulate
    // with NO_OP). Not consistent across SGs; the atomic claim corrects staleness.
    void snapshot_() {
        MPI_Get_accumulate(nullptr, 0, MPI_LONG_LONG,
                           snap_.data(), 2 * n_, MPI_LONG_LONG,
                           0, 0, 2 * n_, MPI_LONG_LONG, MPI_NO_OP, win_);
        MPI_Win_flush(0, win_);
    }

    long long fao_(int slot, long long delta) {
        long long prev = 0;
        MPI_Fetch_and_op(&delta, &prev, MPI_LONG_LONG, 0, slot, MPI_SUM, win_);
        MPI_Win_flush(0, win_);
        return prev;
    }

    int n_;
    std::vector<int> unit_;
    std::vector<int> batch_;
    std::vector<long long> snap_;
    std::vector<long long> rem_;
    int last_stop_count_ = 0;
    std::vector<unsigned char> home_;
    long long* base_ = nullptr;
    MPI_Win win_;
};

}  // namespace klasp
