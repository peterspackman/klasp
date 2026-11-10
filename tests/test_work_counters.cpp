#include <catch2/catch_test_macros.hpp>
#include "klasp/work_counters.hpp"

#include <vector>
#include <algorithm>

using namespace klasp;

// Drive a WorkLedger with `nranks` workers: each holds one claim, produces
// yield_of(sg, chunk) <= chunk structures, reports, and grabs again. Returns
// total produced per SG.
template <typename YieldFn>
static std::vector<long long> simulate(std::vector<long long> targets,
                                       std::vector<int> units,
                                       int nranks, YieldFn yield_of) {
    const int n = static_cast<int>(targets.size());
    WorkLedger ledger(targets, units, nranks);
    std::vector<long long> produced(n, 0);
    std::vector<Grab> hand(nranks);
    for (int r = 0; r < nranks; ++r) hand[r] = ledger.grab();

    long long guard = 0, guard_max = 100LL * 1000 * 1000;
    bool progress = true;
    while (progress && guard++ < guard_max) {
        progress = false;
        for (int r = 0; r < nranks; ++r) {
            if (hand[r].kind == GrabKind::ASSIGN) {
                int yield = yield_of(hand[r].sg_index, hand[r].chunk);
                yield = std::max(0, std::min(yield, hand[r].chunk));
                produced[hand[r].sg_index] += yield;
                ledger.report(hand[r].sg_index, hand[r].chunk, yield);
                hand[r] = ledger.grab();
                progress = true;
            } else if (hand[r].kind == GrabKind::WAIT) {
                hand[r] = ledger.grab();           // a peer's report may have reopened work
                if (hand[r].kind == GrabKind::ASSIGN) progress = true;
            }
        }
    }
    return produced;
}

TEST_CASE("WorkLedger: 100% yield produces exactly the target", "[work_counters]") {
    const int n = 25, nranks = 8, unit = 256;
    const long long target = 10000;
    std::vector<long long> targets(n, target);
    std::vector<int> units(n, unit);
    auto produced = simulate(targets, units, nranks,
                             [](int, int chunk){ return chunk; });   // full yield
    for (int i = 0; i < n; ++i)
        REQUIRE(produced[i] == target);     // exact: no over-, no undersampling
}

TEST_CASE("WorkLedger: heterogeneous low yields still produce exactly", "[work_counters]") {
    const int n = 25, nranks = 8, unit = 256;
    const long long target = 10000;
    std::vector<long long> targets(n, target);
    std::vector<int> units(n, unit);
    // Per-SG yield 30%..96% of a chunk (heavy generation rejection); shortfalls
    // must reopen so production still lands exactly on target.
    auto yfn = [](int sg, int chunk){ return std::max(1, chunk * (30 + (sg % 5) * 15) / 100); };
    auto produced = simulate(targets, units, nranks, yfn);
    for (int i = 0; i < n; ++i)
        REQUIRE(produced[i] == target);
}

TEST_CASE("WorkLedger: uneven targets all hit exactly, none starved", "[work_counters]") {
    std::vector<long long> targets = {500, 10000, 100, 3000, 50, 7777};
    std::vector<int> units(targets.size(), 128);
    auto produced = simulate(targets, units, /*nranks=*/4,
                             [](int, int chunk){ return std::max(1, chunk * 3 / 4); });
    for (size_t i = 0; i < targets.size(); ++i)
        REQUIRE(produced[i] == targets[i]);
}

TEST_CASE("WorkLedger: unit larger than target does not overshoot", "[work_counters]") {
    // Tiny target with a big unit: the chunk clamps to the remaining avail.
    auto produced = simulate({30}, {1024}, /*nranks=*/4, [](int, int c){ return c; });
    REQUIRE(produced[0] == 30);
}

TEST_CASE("WorkLedger: single worker, single SG terminates exactly", "[work_counters]") {
    auto produced = simulate({1000}, {64}, /*nranks=*/1, [](int, int c){ return c; });
    REQUIRE(produced[0] == 1000);
}

TEST_CASE("WorkLedger: over-production credits toward the target", "[work_counters]") {
    // A block yields more valids than the claim and the full count is reported;
    // total produced lands within one unit of the target.
    const long long target = 10000;
    const int unit = 256;
    WorkLedger L({target}, {unit}, 8);
    long long produced_total = 0;
    int guard = 0;
    while (!L.done() && guard++ < 1000000) {
        Grab g = L.grab();
        if (g.kind != GrabKind::ASSIGN) continue;
        int produced = g.chunk + g.chunk / 5;   // 1.2x over-production
        produced_total += produced;
        L.report(g.sg_index, g.chunk, produced);
    }
    REQUIRE(produced_total >= target);
    REQUIRE(produced_total < target + unit);   // overshoot bounded by one block
}

TEST_CASE("WorkLedger: chunk is a full unit until the tail", "[work_counters]") {
    REQUIRE(chunk_for(/*avail*/10000, /*unit*/256) == 256);   // bulk: full unit
    REQUIRE(chunk_for(/*avail*/300,   /*unit*/256) == 256);   // still a full unit
    REQUIRE(chunk_for(/*avail*/100,   /*unit*/256) == 100);   // tail: take the remainder
    REQUIRE(chunk_for(/*avail*/1,     /*unit*/256) == 1);
    REQUIRE(chunk_for(/*avail*/0,     /*unit*/256) == 0);
}

// --- graceful stop ---------------------------------------------------------
// A stop must end claiming without abandoning what is already claimed.

TEST_CASE("a stop request ends the loop without abandoning claimed work",
          "[work][stop]") {
    WorkLedger ledger({1000}, {100}, 4);

    // Two workers holding claims when the stop lands.
    const Grab a = ledger.grab();
    const Grab b = ledger.grab();
    REQUIRE(a.kind == GrabKind::ASSIGN);
    REQUIRE(b.kind == GrabKind::ASSIGN);

    ledger.request_stop();
    REQUIRE(ledger.stop_requested());

    // DONE, not WAIT, so the caller leaves the loop and shuts down.
    REQUIRE(ledger.grab().kind == GrabKind::DONE);

    // Claims in hand still report, so a --resume sees the right work outstanding.
    ledger.report(a.sg_index, a.chunk, a.chunk);
    ledger.report(b.sg_index, b.chunk, b.chunk - 10);   // a shortfall reopens
    REQUIRE(ledger.inflight(0) == 0);
    REQUIRE(ledger.avail(0) == 1000 - a.chunk - b.chunk + 10);
}

TEST_CASE("stopping leaves the remaining target for a resume", "[work][stop]") {
    WorkLedger ledger({500}, {50}, 1);
    long long produced = 0;
    for (int i = 0; i < 3; ++i) {
        const Grab g = ledger.grab();
        REQUIRE(g.kind == GrabKind::ASSIGN);
        ledger.report(g.sg_index, g.chunk, g.chunk);
        produced += g.chunk;
    }
    ledger.request_stop();
    REQUIRE(ledger.grab().kind == GrabKind::DONE);
    // Unclaimed work is still on the books for a resume.
    REQUIRE(ledger.avail(0) == 500 - produced);
    REQUIRE(ledger.inflight(0) == 0);
    REQUIRE_FALSE(ledger.done());   // the target is NOT met, and it must not say it is
}

// --- the zero-yield livelock -----------------------------------------------
// A group that never produces returns its whole chunk each time, so it stays the
// most-behind group.

TEST_CASE("a space group that never yields takes over every claim", "[work][stop]") {
    // Two groups, equal targets. Group 0 produces; group 1 never does.
    WorkLedger ledger({1000, 1000}, {100, 100}, 4);

    int picks_of_dead = 0;
    for (int i = 0; i < 40; ++i) {
        const Grab g = ledger.grab();
        REQUIRE(g.kind == GrabKind::ASSIGN);
        const int produced = g.sg_index == 0 ? g.chunk : 0;
        ledger.report(g.sg_index, g.chunk, produced);
        if (g.sg_index == 1) ++picks_of_dead;
    }
    // The dead group holds its full target while the live one drains, so it is
    // the most-behind group for almost every pick.
    REQUIRE(picks_of_dead > 30);
    REQUIRE(ledger.avail(1) == 1000);
    REQUIRE_FALSE(ledger.done());
}

// The RMA snapshot is atomic per element but not across elements, so an observer
// can sample avail and inflight on either side of a report.
TEST_CASE("termination survives a snapshot that straddles a report", "[work][stop]") {
    WorkLedger ledger({100}, {100}, 2);
    const Grab g = ledger.grab();
    REQUIRE(g.kind == GrabKind::ASSIGN);
    REQUIRE(g.chunk == 100);            // the whole target is claimed
    REQUIRE(ledger.avail(0) == 0);      // ...so nothing is left to claim

    // t1: the observer samples avail (first in the buffer) and sees nothing left.
    const long long seen_avail = ledger.avail(0);

    // The holder now reports a SHORTFALL -- it produced 90 of the 100 it claimed,
    // so 10 reopen and the target is not met.
    ledger.report(0, 100, 90);

    // t2: the observer samples inflight, which the report has since cleared.
    const long long seen_inflight = ledger.inflight(0);

    // These two values alone would wrongly say the run is over.
    REQUIRE(seen_avail <= 0);
    REQUIRE(seen_inflight == 0);

    // remaining is one monotone counter, so the skew cannot reach it.
    REQUIRE(ledger.remaining(0) == 10);
    REQUIRE_FALSE(ledger.done());

    // And the reopened work is really claimable.
    const Grab g2 = ledger.grab();
    REQUIRE(g2.kind == GrabKind::ASSIGN);
    REQUIRE(g2.chunk == 10);
    ledger.report(0, 10, 10);
    REQUIRE(ledger.done());
}

TEST_CASE("over-production still terminates", "[work][stop]") {
    // Over-production drives remaining negative; done() must still hold.
    WorkLedger ledger({100}, {100}, 1);
    const Grab g = ledger.grab();
    ledger.report(g.sg_index, g.chunk, g.chunk + 17);
    REQUIRE(ledger.remaining(0) < 0);
    REQUIRE(ledger.done());
}

TEST_CASE("retiring a dead group lets the rest of the run finish", "[work][stop]") {
    WorkLedger ledger({1000, 1000}, {100, 100}, 4);
    ledger.retire(1);                      // group 1 can never produce

    long long produced = 0;
    int guard = 0;
    for (;;) {
        const Grab g = ledger.grab();
        if (g.kind == GrabKind::DONE) break;
        REQUIRE(g.kind == GrabKind::ASSIGN);
        REQUIRE(g.sg_index == 0);          // the retired group is never handed out
        ledger.report(g.sg_index, g.chunk, g.chunk);
        produced += g.chunk;
        REQUIRE(guard++ < 1000);           // ...and termination is actually reached
    }
    REQUIRE(produced == 1000);             // the live group still hits its target
    REQUIRE(ledger.remaining(1) == 0);
}

TEST_CASE("the zero-yield guard trips on the dead group, not a poor one",
          "[work][stop]") {
    ZeroYieldGuard guard(5);

    // One accepted structure in a unit is not a zero yield.
    for (int i = 0; i < 20; ++i) {
        guard.record(3, 0);
        guard.record(3, 1);
    }
    REQUIRE_FALSE(guard.tripped());

    // The streak is per group.
    for (int i = 0; i < 20; ++i) {
        guard.record(3, 0);
        guard.record(4, 0);
    }
    REQUIRE_FALSE(guard.tripped());

    // Five consecutive empty units from one group is the real signal.
    for (int i = 0; i < 5; ++i) guard.record(7, 0);
    REQUIRE(guard.tripped());
    REQUIRE(guard.space_group_index() == 7);
    REQUIRE(guard.streak() == 5);

    // A single success clears it.
    guard.record(7, 1);
    REQUIRE_FALSE(guard.tripped());
}

// --- space-group affinity ---------------------------------------------------
// Affinity gives each rank a preferred set, but a rank must never idle while
// work exists elsewhere.

TEST_CASE("affinity is deterministic and reaches every space group",
          "[work][affinity]") {
    // fine25-shaped: equal targets, symop counts spanning 1 to 16.
    const std::vector<long long> targets(25, 10000);
    const std::vector<long long> symops = {8,4,4,2,2,8,4,4,4,4,1,8,2,4,8,
                                           4,3,16,8,4,6,16,8,8,6};
    const auto a = assign_home_sets(targets, symops, 32);
    const auto b = assign_home_sets(targets, symops, 32);
    REQUIRE(a == b);            // every rank computes this itself; they must agree

    // Every space group is reachable, or it could never finish.
    for (size_t sg = 0; sg < targets.size(); ++sg) {
        int owners = 0;
        for (const auto& r : a) owners += r[sg] ? 1 : 0;
        REQUIRE(owners >= 1);
    }
    // Every rank has somewhere to start.
    for (const auto& r : a)
        REQUIRE(std::count(r.begin(), r.end(), (unsigned char)1) >= 1);

    // A rank sees far fewer than the 25 space groups.
    double mean = 0;
    for (const auto& r : a) mean += std::count(r.begin(), r.end(), (unsigned char)1);
    mean /= double(a.size());
    REQUIRE(mean < 8.0);
}

TEST_CASE("affinity never idles a rank while work exists", "[work][affinity]") {
    // Rank 0's home set is exhausted early; it must then pick up the other group.
    WorkLedger ledger({100, 1000}, {100, 100}, 2);
    ledger.set_home({1, 0});                 // home = group 0 only

    const Grab first = ledger.grab();
    REQUIRE(first.sg_index == 0);            // prefers home while it has work
    ledger.report(0, first.chunk, first.chunk);

    // Home is now empty but group 1 is untouched: fall back, do not stall.
    const Grab next = ledger.grab();
    REQUIRE(next.kind == GrabKind::ASSIGN);
    REQUIRE(next.sg_index == 1);
}

TEST_CASE("affinity still produces exactly the target", "[work][affinity]") {
    const std::vector<long long> targets = {500, 500, 500};
    const auto home = assign_home_sets(targets, {1, 4, 8}, 4);

    std::vector<WorkLedger> ranks;
    for (int r = 0; r < 4; ++r) {
        ranks.emplace_back(targets, std::vector<int>{50, 50, 50}, 4);
        ranks.back().set_home(home[r]);
    }
    // One shared ledger is the accounting; each rank's preference only selects the
    // home set.
    WorkLedger shared(targets, {50, 50, 50}, 4);
    std::vector<long long> produced(3, 0);
    int guard = 0;
    for (int r = 0; ; r = (r + 1) % 4) {
        shared.set_home(home[r]);
        const Grab g = shared.grab();
        if (g.kind == GrabKind::DONE) break;
        REQUIRE(g.kind == GrabKind::ASSIGN);
        // 90% acceptance, so shortfalls reopen and have to be re-produced.
        const int yield = std::max(1, (g.chunk * 9) / 10);
        shared.report(g.sg_index, g.chunk, yield);
        produced[g.sg_index] += yield;
        REQUIRE(guard++ < 100000);
    }
    for (int i = 0; i < 3; ++i) REQUIRE(produced[i] >= targets[i]);
}

TEST_CASE("the tail of a space group is split, not handed to one rank",
          "[work][chunk]") {
    const int unit = 4096, batch = 1024;

    // The bulk: a full unit whenever there is one to give.
    CHECK(chunk_for(250000, unit, batch) == unit);
    CHECK(chunk_for(4096,   unit, batch) == unit);

    // Below a unit, halve rather than hand it all over.
    CHECK(chunk_for(4000, unit, batch) == 2000);
    CHECK(chunk_for(3000, unit, batch) == 1500);

    // Never below one batch: half of 1500 is under the batch, so 1024.
    CHECK(chunk_for(1500, unit, batch) == batch);
    CHECK(chunk_for(1024, unit, batch) == 1024);

    // Under a batch there is nothing left to split: take what is there.
    CHECK(chunk_for(700, unit, batch) == 700);
    CHECK(chunk_for(1,   unit, batch) == 1);
    CHECK(chunk_for(0,   unit, batch) == 0);

    // Repeated claims drain the SG rather than stalling.
    long long avail = 4000;
    int claims = 0;
    while (avail > 0) {
        const int got = chunk_for(avail, unit, batch);
        REQUIRE(got > 0);
        avail -= got;
        REQUIRE(++claims < 100);
    }
    CHECK(avail == 0);
    // 4000 goes 2000, then 1024 (batch floor), then the last 976.
    CHECK(claims == 3);
}

TEST_CASE("without a batch floor the tail behaves as it always did",
          "[work][chunk]") {
    // The default batch argument must not change the arithmetic.
    CHECK(chunk_for(10000, 256) == 256);
    CHECK(chunk_for(300,   256) == 256);
    CHECK(chunk_for(100,   256) == 100);
    CHECK(chunk_for(1,     256) == 1);
    CHECK(chunk_for(0,     256) == 0);
}
