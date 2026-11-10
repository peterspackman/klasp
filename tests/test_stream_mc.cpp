// The Metropolis stepper: the acceptance rule (downhill always, uphill with
// probability exp(-dE/kT)), a chain that depends on its seed alone, and moves
// confined to the prior's support.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "test_support.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/stream_mc.hpp"
#include "klasp/sgdata.hpp"

#include <Kokkos_Random.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace klasp;
using klasp::testing::make_fixture;
using Catch::Matchers::WithinAbs;

namespace {

BatchOptimizeSettings mc_batch_settings() {
    BatchOptimizeSettings s;
    s.n_iterations = 1;          // unused: the chain finishes on its move budget
    s.store_diagnostics = false;
    s.model.taper_on = 0;
    s.model.taper_off = 0;
    s.refill_interval = 2;
    return s;
}

McSettings mc_chain(int moves) {
    McSettings m;
    m.kT = RealType(0.5);
    m.n_moves = moves;
    m.step_trans = RealType(0.02);
    m.step_rot = RealType(4.0 * M_PI / 180.0);
    m.step_strain = RealType(0.005);
    m.min_contact = RealType(1.0);
    // P-1, with a volume budget far above the fixture's ~300 A^3 cell so the
    // bound never binds here.
    m.prior = make_sampling_prior(2, 1, 1.0e6);
    return m;
}

// Park copies of the fixture's structure in a pool under the given identities.
// Identical geometry, different seeds.
StreamPool pool_with_seeds(testing::Fixture& f, const std::vector<int>& seeds) {
    StreamPool pool = allocate_stream_pool(static_cast<int>(seeds.size()),
                                           f.bws.tables.n_indep);
    std::vector<int> one(1);
    for (int s : seeds) {
        one[0] = s;
        stream_pool_append(pool, f.bws, 1, one, {}, /*with_hessian=*/false);
    }
    return pool;
}

std::vector<double> run_chain(const std::vector<int>& seeds, int moves) {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));
    StreamPool pool = pool_with_seeds(f, seeds);
    const auto r = batch_optimize_stream_mc<1>(f.bws, pool, mc_batch_settings(),
                                               mc_chain(moves));
    std::vector<double> out(r.n);
    for (int i = 0; i < r.n; ++i) out[i] = r.energies(i);
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// The acceptance rule
// ---------------------------------------------------------------------------

TEST_CASE("a downhill move is always accepted", "[mc]") {
    // The uniform is consumed on every move, so this must hold for u up to 1.
    for (RealType u : {RealType(0), RealType(0.5), RealType(0.999999)}) {
        REQUIRE(mc_accepts(RealType(-1e-12), RealType(2.5), u));
        REQUIRE(mc_accepts(RealType(-40.0),  RealType(2.5), u));
        REQUIRE(mc_accepts(RealType(0),      RealType(2.5), u));   // flat counts as downhill
    }
}

TEST_CASE("kT <= 0 is the greedy limit, not a division by zero", "[mc]") {
    REQUIRE(mc_accepts(RealType(-1.0), RealType(0), RealType(0.0)));
    REQUIRE_FALSE(mc_accepts(RealType(1e-9), RealType(0), RealType(0.0)));
    REQUIRE_FALSE(mc_accepts(RealType(1e-9), RealType(-1), RealType(0.0)));
}

TEST_CASE("an uphill move is accepted with probability exp(-dE/kT)", "[mc]") {
    // Three ratios spanning two orders of magnitude in the expected rate, so a
    // merely monotone rule or a wrong constant in the exponent fails.
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const RealType kT = RealType(2.494);          // R * 300 K, kJ/mol
    const int N = 400000;
    for (double dE : {0.5, 2.494, 7.482}) {       // dE/kT = 0.2, 1, 3
        int taken = 0;
        for (int i = 0; i < N; ++i)
            if (mc_accepts(RealType(dE), kT, RealType(uni(rng)))) ++taken;
        const double got = double(taken) / N;
        const double want = std::exp(-dE / double(kT));
        // 4 sigma on a binomial of N trials.
        const double sigma = std::sqrt(want * (1 - want) / N);
        std::printf("  dE = %.3f kJ/mol: accepted %.4f, exp(-dE/kT) = %.4f, 4sig = %.4f\n",
                    dE, got, want, 4 * sigma);
        REQUIRE_THAT(got, WithinAbs(want, 4 * sigma));
    }
}

// ---------------------------------------------------------------------------
// The chain's identity
// ---------------------------------------------------------------------------

TEST_CASE("a chain's stream comes from the structure's seed, not its slot", "[mc]") {
    // The per-slot state is overwritten with mc_slot_seed at fill time, so
    // generators at different slot indices must produce the same draws.
    using Gen = Kokkos::Random_XorShift64<ExecSpace>;
    Gen a(mc_slot_seed(7), 0), b(mc_slot_seed(7), 5);
    Gen c(mc_slot_seed(8), 0);
    bool differs_from_other_seed = false;
    for (int i = 0; i < 32; ++i) {
        const double xa = a.drand(), xb = b.drand(), xc = c.drand();
        REQUIRE(xa == xb);
        if (xa != xc) differs_from_other_seed = true;
    }
    REQUIRE(differs_from_other_seed);
}

TEST_CASE("the same seed gives the same chain, in any slot and on any run", "[mc]") {
    // One slot, three entries: entries 2 and 3 arrive by refill at different
    // rounds, with identical geometry and seeds, so they must agree.
    const auto same = run_chain({7, 7, 7}, 60);
    REQUIRE(same.size() == 3);
    for (double e : same) REQUIRE(std::isfinite(e));
    std::printf("  three chains from seed 7: %.12f %.12f %.12f\n",
                same[0], same[1], same[2]);
    REQUIRE(same[1] == same[0]);
    REQUIRE(same[2] == same[0]);

    // The same again from a fresh workspace.
    const auto again = run_chain({7, 7, 7}, 60);
    REQUIRE(again[0] == same[0]);

    // A different seed gives a different chain.
    const auto other = run_chain({8}, 60);
    REQUIRE(std::isfinite(other[0]));
    REQUIRE(other[0] != same[0]);
}

TEST_CASE("the chain harvests the best structure it saw, not the last", "[mc]") {
    // A longer chain from the same seed shares the shorter one's prefix, so it
    // can only improve on it.
    const auto short_run = run_chain({11}, 20);
    const auto long_run  = run_chain({11}, 200);
    std::printf("  20 moves: %.10f   200 moves: %.10f\n", short_run[0], long_run[0]);
    REQUIRE(long_run[0] <= short_run[0]);
    // Both are below the starting energy.
    const auto init_only = run_chain({11}, 1);
    REQUIRE(short_run[0] <= init_only[0]);
}

// ---------------------------------------------------------------------------
// The support
// ---------------------------------------------------------------------------

TEST_CASE("prior_accepts_cell is the generator's support, not a tolerance", "[mc]") {
    // Volume budget sized for a ~6 x 7 x 8 A box (336 A^3 at right angles).
    SamplingPrior p = make_sampling_prior(2, 1, 400.0);   // P-1, triclinic
    const RealType deg = RealType(M_PI / 180.0);

    // Inside on both counts.
    REQUIRE(prior_accepts_cell(p, 6, 7, 8, 90*deg, 90*deg, 90*deg));

    // Outside the angle window, one angle at a time, on both sides.
    REQUIRE_FALSE(prior_accepts_cell(p, 6, 7, 8, 40*deg, 90*deg, 90*deg));
    REQUIRE_FALSE(prior_accepts_cell(p, 6, 7, 8, 90*deg, 140*deg, 90*deg));
    REQUIRE_FALSE(prior_accepts_cell(p, 6, 7, 8, 90*deg, 90*deg, 44.9*deg));
    REQUIRE(prior_accepts_cell(p, 6, 7, 8, 45*deg, 90*deg, 90*deg));     // the edge is in

    // Over the volume budget, with every angle inside the window.
    REQUIRE_FALSE(prior_accepts_cell(p, 9, 9, 9, 90*deg, 90*deg, 90*deg));   // 729 A^3
    // Shearing the same lengths lowers the volume, so it is admitted.
    REQUIRE(prior_accepts_cell(p, 6, 7, 8, 70*deg, 100*deg, 80*deg));

    // Angles that close on no real volume are rejected, not NaN.
    REQUIRE_FALSE(prior_accepts_cell(p, 6, 7, 8, 45*deg, 45*deg, 135*deg));
}
