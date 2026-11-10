#pragma once
// Lock-free work accounting for the CSP work distribution. In a live run the
// counters live in an MPI RMA window updated by atomic Fetch_and_op; this header
// holds the arithmetic plus a single-process `WorkLedger`.
//
// Counters per space-group index i:
//   avail[i]     structures still available to claim (starts at target[i]).
//   inflight[i]  structures claimed but not yet reported (starts at 0).
//   remaining[i] structures still to produce (starts at target[i]).
//
// A worker claims a chunk from the most-behind SG (max avail) and later reports
// how many it produced; any shortfall returns to avail, so total production
// converges to the target exactly.
//
// Termination reads `remaining` only. It equals avail + inflight but is a single
// counter that only decreases, so a stale or non-atomic snapshot can give a
// spurious WAIT but never a false DONE.

#include <vector>
#include <algorithm>
#include <cstddef>

namespace klasp {

enum class GrabKind { ASSIGN, WAIT, DONE };

// Watches for a space group that claims work and returns nothing. report()
// returns a zero-yield chunk to avail, so such a group would otherwise stay the
// most-behind forever. Rank-local: every rank sees the same zero yields.
class ZeroYieldGuard {
public:
    explicit ZeroYieldGuard(int limit = 5) : limit_(limit) {}

    // One completed claim. Any production at all clears the streak.
    void record(int sg_index, int produced) {
        if (produced > 0) { streak_ = 0; sg_ = -1; return; }
        if (sg_index != sg_) { sg_ = sg_index; streak_ = 1; return; }
        ++streak_;
    }

    // After the caller has acted on a trip, watch for the next group.
    void reset() { streak_ = 0; sg_ = -1; }

    bool tripped() const { return streak_ >= limit_; }
    int streak() const { return streak_; }
    int space_group_index() const { return sg_; }

private:
    int limit_;
    int streak_ = 0;
    int sg_ = -1;
};

struct Grab {
    GrabKind kind = GrabKind::DONE;
    int sg_index = -1;
    int chunk = 0;      // amount actually claimed (<= requested)
};

// Most-behind SG: the index with the largest avail, or -1 if none has avail > 0.
inline int choose_sg(const long long* avail, int n) {
    int best = -1;
    long long best_avail = 0;
    for (int i = 0; i < n; ++i)
        if (avail[i] > best_avail) { best_avail = avail[i]; best = i; }
    return best;
}

// Most-behind SG among this rank's home set, falling back to the global pick
// when the home set has no work. Concentrating ranks on space groups lets the
// per-rank stage-3 basin index skip more repeat relaxations.
inline int choose_sg_home(const long long* avail, int n,
                          const unsigned char* home) {
    int best = -1;
    long long best_avail = 0;
    for (int i = 0; i < n; ++i)
        if (home[i] && avail[i] > best_avail) { best_avail = avail[i]; best = i; }
    return best >= 0 ? best : choose_sg(avail, n);
}

// Assign each rank a home set, weighted by each space group's work (symops x
// target) rather than its structure count. A pure function of shared values, so
// all ranks compute the same map without communication.
inline std::vector<std::vector<unsigned char>>
assign_home_sets(const std::vector<long long>& targets,
                 const std::vector<long long>& cost_per_structure,
                 int nranks) {
    const int n = static_cast<int>(targets.size());
    std::vector<std::vector<unsigned char>> home(
        nranks, std::vector<unsigned char>(n, 0));
    if (n == 0 || nranks <= 0) return home;

    std::vector<long long> w(n);
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        w[i] = std::max(1LL, targets[i]) * std::max(1LL, cost_per_structure[i]);
        total += w[i];
    }

    // Heaviest group first.
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return w[a] > w[b]; });

    std::vector<long long> load(nranks, 0);
    for (int idx : order) {
        // Ranks proportional to this group's share of the work, at least one so
        // every group can finish.
        int want = static_cast<int>((static_cast<double>(w[idx]) / double(total))
                                    * nranks + 0.5);
        want = std::max(1, std::min(nranks, want));
        for (int k = 0; k < want; ++k) {
            const int r = static_cast<int>(
                std::min_element(load.begin(), load.end()) - load.begin());
            home[r][idx] = 1;
            load[r] += w[idx] / want;
        }
    }
    return home;
}

// Chunk to attempt for an SG given a (possibly stale) snapshot of its avail.
// A full unit until the SG's remainder drops below a unit. Not divided by rank
// count, which would underfill the GPU.
inline int chunk_for(long long avail_sg, int unit_sg, int batch_sg = 0) {
    if (avail_sg <= 0) return 0;
    const int u = unit_sg > 0 ? unit_sg : 1;
    if (avail_sg >= u) return u;

    // The tail: halve the remainder so several ranks share it, with a floor of one
    // batch so each claim still fills a kernel launch.
    const long long b = batch_sg > 0 ? std::min<long long>(batch_sg, u) : u;
    if (avail_sg <= b) return static_cast<int>(avail_sg);
    return static_cast<int>(std::max<long long>(b, (avail_sg + 1) / 2));
}

// Single-process model of the counters: the exact arithmetic the RMA path
// performs, without remote atomics. grab() re-checks what it actually claimed
// rather than trusting the pre-claim read.
class WorkLedger {
public:
    WorkLedger(std::vector<long long> targets, std::vector<int> unit_sizes, int nranks,
               std::vector<int> batch_sizes = {})
        : n_(static_cast<int>(targets.size())),
          avail_(targets),
          inflight_(n_, 0),
          remaining_(std::move(targets)),
          seed_(n_, 0),
          unit_(std::move(unit_sizes)),
          batch_(std::move(batch_sizes)) {
        (void)nranks;
        for (int i = 0; i < n_; ++i) if (unit_[i] < 1) unit_[i] = 1;
        // Absent means no floor below the unit.
        batch_.resize(n_, 0);
    }

    // This rank's preferred space groups (empty = no preference).
    void set_home(std::vector<unsigned char> home) { home_ = std::move(home); }

    // Stop claiming new work on every rank; each finishes, reports and flushes
    // its unit in hand. WorkWindow holds the shared-memory version of this flag.
    void request_stop() { stop_ = true; }
    bool stop_requested() const { return stop_; }

    Grab grab() {
        // DONE, not WAIT: claimed work is reported before this, so the shortfall is
        // back in avail_ for a --resume.
        if (stop_) return {GrabKind::DONE, -1, 0};
        const int sg = home_.empty() ? choose_sg(avail_.data(), n_)
                                     : choose_sg_home(avail_.data(), n_, home_.data());
        if (sg < 0) {
            bool any_inflight = false;
            for (int i = 0; i < n_; ++i) if (inflight_[i] > 0) { any_inflight = true; break; }
            return {any_inflight ? GrabKind::WAIT : GrabKind::DONE, -1, 0};
        }
        const int want = chunk_for(avail_[sg], unit_[sg], batch_[sg]);
        // Atomic claim: subtract, then clamp to what was actually there. `got` can
        // be less than `want` (or zero -> WAIT) under a concurrent claim.
        const long long prev = avail_[sg];
        avail_[sg] -= want;
        long long got = want;
        if (got > prev) got = prev < 0 ? 0 : prev;   // over-claimed: only `prev` was left
        const long long over = want - got;
        if (over > 0) avail_[sg] += over;             // give the over-claim back
        if (got <= 0) return {GrabKind::WAIT, -1, 0};
        inflight_[sg] += got;
        return {GrabKind::ASSIGN, sg, static_cast<int>(got)};
    }

    // Report a finished chunk: `claimed` was the grant, `produced` the valids
    // optimized. Under-production reopens the shortfall; over-production (a whole
    // block is optimized) credits the extra toward the target.
    void report(int sg, int claimed, int produced) {
        inflight_[sg] -= claimed;
        if (inflight_[sg] < 0) inflight_[sg] = 0;
        avail_[sg] += (claimed - produced);
        remaining_[sg] -= produced;
    }

    // Over-production (a block yields more than the claim) drives this negative,
    // hence <= rather than ==.
    bool done() const {
        for (int i = 0; i < n_; ++i)
            if (remaining_[i] > 0) return false;
        return true;
    }

    // Retire a space group whose generator can never produce. Clears remaining
    // too, so termination can be reached.
    void retire(int sg) { avail_[sg] = 0; remaining_[sg] = 0; }

    // Claim a contiguous block of `n` generator seeds for an SG, returning the
    // first. One cursor per space group, as in the RMA window, so the seeds drawn
    // do not depend on the process count. `seed_start` is the resume cursor.
    long long claim_seeds(int sg, int n) {
        const long long first = seed_[sg];
        seed_[sg] += n;
        return first;
    }

    void set_seed_start(const std::vector<long long>& seed_start) {
        for (int i = 0; i < n_ && i < static_cast<int>(seed_start.size()); ++i)
            seed_[i] = seed_start[i];
    }

    // Total work still owed, with negative (over-produced) remainders clamped to 0
    // as WorkWindow does.
    long long remaining_total() const {
        long long t = 0;
        for (long long v : remaining_) t += v > 0 ? v : 0;
        return t;
    }

    int n() const { return n_; }
    long long avail(int i) const { return avail_[i]; }
    long long inflight(int i) const { return inflight_[i]; }
    long long remaining(int i) const { return remaining_[i]; }

private:
    int n_;
    std::vector<long long> avail_;
    std::vector<long long> inflight_;
    std::vector<long long> remaining_;
    std::vector<long long> seed_;
    std::vector<int> unit_;
    std::vector<int> batch_;
    std::vector<unsigned char> home_;
    bool stop_ = false;
};

}  // namespace klasp
