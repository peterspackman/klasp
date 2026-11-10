#pragma once
// The streaming scheduler: fill every slot from a pending pool, step, harvest
// finished slots, refill, and stop when the pool is dry and the last slot has
// drained. What a step does is a Stepper, a compile-time policy with no virtual
// dispatch.
//
// Stepper, host side (between kernel launches):
//
//   void bind(Kokkos::View<int*, Space> slot_state)
//       Called once before any fill. The engine owns slot_state
//       (SLOT_EMPTY / SLOT_LIVE / SLOT_DONE); initialise per-slot state here.
//
//   SView1D<RealType, Space> energies() const
//       Where evaluate() deposits each slot's energy, passed to accept().
//
//   void evaluate(int round)
//       Run the oracle over the live slots.
//
//   void on_filled(int n)
//       n slots were just refilled; work needed after their first evaluation
//       is latched here and done in after_round().
//
//   void after_round()
//       Bookkeeping after accept(), before the next propose().
//
//   void harvest(Kokkos::View<int*, Space> idx, int n, int base)
//       Emit the accepted state of slots idx(0..n) to destination rows
//       base..base+n, using stream::for_slot_pairs. Destination capacity is
//       Config::harvest_capacity.
//
// Stepper, device side (returned by value and captured into the engine's
// kernels):
//
//   Filler   filler()   -> KOKKOS_INLINE_FUNCTION void operator()(int b, int src) const
//       Seed slot b from pending entry src.
//
//   Proposer proposer() -> KOKKOS_INLINE_FUNCTION void operator()(int b) const
//       Move slot b to the point the oracle should evaluate.
//
//   Accepter accepter() -> KOKKOS_INLINE_FUNCTION bool operator()(int b, RealType E) const
//       Process the energy and return true when the slot is finished.

#include "klasp/batch.hpp"

#include <Kokkos_Core.hpp>
#include <atomic>

namespace klasp {
namespace stream {

// What the engine needs to know that is not the stepper's business.
struct Config {
    int capacity = 0;            // slots; every one of them is used
    int refill_interval = 10;    // rounds between checkpoints
    int source_base = 0;         // index of the first pending entry
    int source_count = 0;        // pending entries available from source_base
    int harvest_capacity = 0;    // rows the destination can accept, in total
    // Checked once per checkpoint, not per round. See OptimizerHooks::abort.
    const std::atomic<bool>* abort = nullptr;
};

struct Result {
    int n_harvested = 0;   // rows written by harvest()
    int n_consumed = 0;    // pending entries pulled into slots
    int rounds = 0;        // propose/evaluate/accept cycles run
    int total_iters = 0;   // sum over rounds of the in-flight slot count
};

// Apply a (slot, index) device functor over the first n entries of idx, pairing
// slot idx(j) with index base + j.
template <typename Space, class F>
void for_slot_pairs(const char* name, Kokkos::View<int*, Space> idx,
                    int n, int base, F f) {
    if (n <= 0) return;
    Kokkos::parallel_for(name, Kokkos::RangePolicy<Space>(0, n),
        KOKKOS_LAMBDA(const int j) { f(idx(j), base + j); });
}

// Fill a slot and mark it live, in one kernel.
template <typename Space, class Filler>
struct FillAndActivate {
    Filler f;
    Kokkos::View<int*, Space> state;
    KOKKOS_INLINE_FUNCTION void operator()(const int b, const int src) const {
        f(b, src);
        state(b) = SLOT_LIVE;
    }
};

// Keep every slot busy until the pending pool is dry and the last slot drains:
// step every live slot refill_interval times, then checkpoint (compact finished
// slots, harvest, refill, release what the pool cannot refill).
template <typename Space, class Stepper>
Result schedule(const Config& cfg, Stepper& st) {
    using Vi = Kokkos::View<int*, Space>;
    const int cap = cfg.capacity;

    // Slot lifecycle and the compacted index list, both device-resident; only
    // the done count crosses to the host.
    Vi state("st_state", cap), done("st_done", cap);
    Kokkos::deep_copy(state, static_cast<int>(SLOT_EMPTY));
    st.bind(state);

    Result res;
    const int source_end = cfg.source_base + cfg.source_count;
    int consumed = cfg.source_base;
    int n_inflight = 0;

    auto do_fill = [&](Vi idx, int n, int base) {
        if (n <= 0) return;
        FillAndActivate<Space, decltype(st.filler())> fa{st.filler(), state};
        for_slot_pairs<Space>("st_fill", idx, n, base, fa);
        st.on_filled(n);
    };

    // One propose / evaluate / accept cycle over every live slot.
    auto run_round = [&]() {
        {
            auto p = st.proposer();
            auto state_l = state;
            Kokkos::parallel_for("st_propose", Kokkos::RangePolicy<Space>(0, cap),
                KOKKOS_LAMBDA(const int b) {
                    if (state_l(b) != SLOT_LIVE) return;
                    p(b);
                });
        }
        st.evaluate(res.rounds);
        {
            auto a = st.accepter();
            auto state_l = state;
            auto energies = st.energies();
            Kokkos::parallel_for("st_accept", Kokkos::RangePolicy<Space>(0, cap),
                KOKKOS_LAMBDA(const int b) {
                    if (state_l(b) != SLOT_LIVE) return;
                    if (a(b, energies(b))) state_l(b) = SLOT_DONE;
                });
        }
        st.after_round();
        ++res.rounds;
    };

    {
        const int n0 = Kokkos::min(cap, source_end - consumed);
        auto done_l = done;
        Kokkos::parallel_for("st_iota", Kokkos::RangePolicy<Space>(0, n0),
            KOKKOS_LAMBDA(const int j) { done_l(j) = j; });
        do_fill(done, n0, consumed);
        consumed += n0;
        n_inflight = n0;
    }

    // A freshly filled slot proposes a null step, so the first round evaluates
    // the starting geometry. Not counted in total_iters.
    run_round();

    while (n_inflight > 0) {
        // Abandon slots still iterating; harvested rows are already out.
        if (cfg.abort && cfg.abort->load(std::memory_order_relaxed)) break;
        for (int k = 0; k < cfg.refill_interval && n_inflight > 0; ++k) {
            run_round();
            res.total_iters += n_inflight;   // approximate
        }

        // Compact finished slots with a scan, not an atomic counter, so they are
        // in slot order and refill and truncation are deterministic.
        int count = 0;
        {
            auto done_l = done;
            auto state_l = state;
            Kokkos::parallel_scan("st_compact", Kokkos::RangePolicy<Space>(0, cap),
                KOKKOS_LAMBDA(const int b, int& upd, const bool final) {
                    const bool is_done = (state_l(b) == SLOT_DONE);
                    if (final && is_done) done_l(upd) = b;
                    if (is_done) ++upd;
                }, count);
        }
        if (count == 0) continue;

        // Never write past harvest_capacity; the caller sizes it to fit.
        const int base = res.n_harvested;
        const int n_take = Kokkos::min(count, cfg.harvest_capacity - res.n_harvested);
        if (n_take > 0) st.harvest(done, n_take, base);
        res.n_harvested += n_take;

        // Refill as many freed slots as the pool can supply; release the rest.
        // Harvest runs first, so a refill may overwrite the slot it reads.
        const int n_fill = Kokkos::min(count, source_end - consumed);
        do_fill(done, n_fill, consumed);
        consumed += n_fill;
        const int n_drain = count - n_fill;
        if (n_drain > 0) {
            auto done_l = done;
            auto state_l = state;
            Kokkos::parallel_for("st_drain", Kokkos::RangePolicy<Space>(n_fill, count),
                KOKKOS_LAMBDA(const int j) { state_l(done_l(j)) = SLOT_EMPTY; });
            n_inflight -= n_drain;
        }
    }

    Space().fence();
    res.n_consumed = consumed - cfg.source_base;
    return res;
}

}  // namespace stream
}  // namespace klasp
