#include <cstring>
#include "klasp/powder.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "kokkos_listener.hpp"
#include "klasp/batch.hpp"
#include "klasp/descriptors.hpp"
#include "klasp/dedup.hpp"  // AMD_K

using namespace klasp;
using Catch::Matchers::WithinAbs;

// A minimal BatchWorkspace (only what compute_amd reads) with a 2-atom crystal
// per batch.
TEST_CASE("PDD rows average to AMD", "[pdd][descriptors]") {
    const int n_batch = 2, n_indep = 1, n_sites = 2, n_symops = 1;
    const int n_atoms = n_symops * n_sites;      // 2
    const int n_asym  = n_indep * n_sites;       // 2

    BatchWorkspace bws;
    bws.geom.n_batch = n_batch;
    bws.tables.n_indep = n_indep;
    bws.tables.n_sites_per_mol = n_sites;
    bws.geom.n_atoms = n_atoms;

    bws.geom.positions = View3D<RealType>("pos", n_batch, n_atoms, 3);
    bws.geom.cell      = View3D<RealType>("cell", n_batch, 3, 3);
    bws.geom.max_shift = View2D<int>("mshift", n_batch, 3);

    auto h_pos = Kokkos::create_mirror_view(bws.geom.positions);
    auto h_cell = Kokkos::create_mirror_view(bws.geom.cell);
    auto h_ms = Kokkos::create_mirror_view(bws.geom.max_shift);

    const double a_of[2] = {3.0, 4.0};  // different cubic cell per batch (Bohr)
    for (int b = 0; b < n_batch; ++b) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                h_cell(b, r, c) = (r == c) ? a_of[b] : 0.0;
        for (int d = 0; d < 3; ++d) h_ms(b, d) = 2;
        // atom 0 at origin, atom 1 offset, so the rows differ
        h_pos(b, 0, 0) = 0; h_pos(b, 0, 1) = 0;        h_pos(b, 0, 2) = 0;
        h_pos(b, 1, 0) = 1; h_pos(b, 1, 1) = 0.5;      h_pos(b, 1, 2) = 0;
    }
    Kokkos::deep_copy(bws.geom.positions, h_pos);
    Kokkos::deep_copy(bws.geom.cell, h_cell);
    Kokkos::deep_copy(bws.geom.max_shift, h_ms);

    AMDResult res = compute_amd(bws.tables, bws.geom, AMD_K, /*want_pdd=*/true);

    REQUIRE(res.k == AMD_K);
    REQUIRE(res.n_asym == n_asym);
    REQUIRE((int)res.pdd.size() == n_batch);
    for (int b = 0; b < n_batch; ++b)
        REQUIRE((int)res.pdd[b].size() == n_asym * AMD_K);

    // AMD[b][j] must equal the mean over asym rows of PDD[b][i][j].
    for (int b = 0; b < n_batch; ++b) {
        for (int j = 0; j < AMD_K; ++j) {
            double mean = 0;
            for (int i = 0; i < n_asym; ++i)
                mean += res.pdd[b][(size_t)i * AMD_K + j];
            mean /= n_asym;
            REQUIRE_THAT(res.vectors[b][j], WithinAbs(mean, 1e-9));
        }
        // PDD rows are sorted ascending per row (k-NN distances)
        for (int i = 0; i < n_asym; ++i)
            for (int j = 1; j < AMD_K; ++j)
                REQUIRE(res.pdd[b][(size_t)i*AMD_K + j] >=
                        res.pdd[b][(size_t)i*AMD_K + j - 1] - 1e-12);
    }

    // Different cells give different descriptors (column 0 is intramolecular,
    // so compare the full vector).
    REQUIRE(amd_distance(res.vectors[0], res.vectors[1]) > 1e-6);
}

// Brute-force EMD reference for small n: min over row permutations of the
// average L∞ row distance.
static double brute_emd(const std::vector<double>& P, const std::vector<double>& Q,
                        int n, int k) {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    double best = 1e300;
    do {
        double s = 0;
        for (int i = 0; i < n; ++i)
            s += pdd_row_distance(P.data() + (size_t)i*k, Q.data() + (size_t)perm[i]*k, k);
        best = std::min(best, s / n);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

TEST_CASE("PDD EMD: Hungarian matches brute-force optimal assignment", "[pdd][emd]") {
    const int k = 4;
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> U(0.0, 5.0);

    for (int n : {1, 2, 3, 5, 6}) {
        std::vector<double> P(n*k), Q(n*k);
        for (auto& x : P) x = U(rng);
        for (auto& x : Q) x = U(rng);
        // Rows sorted ascending, like real PDD rows.
        for (int i = 0; i < n; ++i) {
            std::sort(P.begin()+i*k, P.begin()+(i+1)*k);
            std::sort(Q.begin()+i*k, Q.begin()+(i+1)*k);
        }
        const double got = pdd_emd(P.data(), Q.data(), n, k);
        const double ref = brute_emd(P, Q, n, k);
        REQUIRE_THAT(got, WithinAbs(ref, 1e-9));
    }
}

TEST_CASE("PDD EMD: metric properties", "[pdd][emd]") {
    const int n = 4, k = 3;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(0.0, 3.0);
    std::vector<double> P(n*k), Q(n*k);
    for (auto& x : P) x = U(rng);
    for (auto& x : Q) x = U(rng);

    // identity: EMD(P,P) == 0
    REQUIRE_THAT(pdd_emd(P.data(), P.data(), n, k), WithinAbs(0.0, 1e-12));
    // symmetry
    REQUIRE_THAT(pdd_emd(P.data(), Q.data(), n, k),
                 WithinAbs(pdd_emd(Q.data(), P.data(), n, k), 1e-12));

    // permuting P's rows leaves EMD(P,Q) unchanged (PDD is a set of rows)
    std::vector<double> Pp(n*k);
    const int order[4] = {2, 0, 3, 1};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) Pp[i*k+j] = P[order[i]*k+j];
    REQUIRE_THAT(pdd_emd(P.data(), Q.data(), n, k),
                 WithinAbs(pdd_emd(Pp.data(), Q.data(), n, k), 1e-12));

    // a uniform constant offset c on every coordinate => EMD == c (L∞ ground)
    std::vector<double> Pc = P;
    for (auto& x : Pc) x += 0.37;
    REQUIRE_THAT(pdd_emd(P.data(), Pc.data(), n, k), WithinAbs(0.37, 1e-12));

    // AMD L∞ distance lower-bounds the PDD EMD (Widdowson & Kurlin)
    std::vector<double> amdP(k, 0), amdQ(k, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) { amdP[j] += P[i*k+j]/n; amdQ[j] += Q[i*k+j]/n; }
    double amd_linf = 0;
    for (int j = 0; j < k; ++j) amd_linf = std::max(amd_linf, std::abs(amdP[j]-amdQ[j]));
    REQUIRE(amd_linf <= pdd_emd(P.data(), Q.data(), n, k) + 1e-12);
}

TEST_CASE("PDD comparison hierarchy: aligned >= grouped >= full EMD", "[pdd][emd]") {
    const int n = 6, k = 4;
    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> U(0.0, 4.0);
    std::vector<double> P(n*k), Q(n*k);
    for (auto& x : P) x = U(rng);
    for (auto& x : Q) x = U(rng);

    const double full    = pdd_emd(P.data(), Q.data(), n, k);
    const double aligned = pdd_distance_aligned(P.data(), Q.data(), n, k);

    // element/equivalence labels: two C-like (0), two O-like (1), two N-like (2)
    const int group[6] = {0, 0, 1, 1, 2, 2};
    const double grouped = pdd_emd_grouped(P.data(), Q.data(), n, k, group);

    // the cost ladder: more permutation freedom => lower cost
    REQUIRE(full   <= grouped + 1e-12);
    REQUIRE(grouped <= aligned + 1e-12);

    // singleton groups => grouped reduces to aligned (forced identity)
    const int singletons[6] = {0, 1, 2, 3, 4, 5};
    REQUIRE_THAT(pdd_emd_grouped(P.data(), Q.data(), n, k, singletons),
                 WithinAbs(aligned, 1e-12));

    // one group => grouped reduces to the full EMD
    const int one_group[6] = {0, 0, 0, 0, 0, 0};
    REQUIRE_THAT(pdd_emd_grouped(P.data(), Q.data(), n, k, one_group),
                 WithinAbs(full, 1e-12));

    // aligned is exact (== full) when there is genuinely no helpful permutation:
    // compare a structure to a small perturbation of itself.
    std::vector<double> Pp = P;
    for (auto& x : Pp) x += 0.01;
    REQUIRE_THAT(pdd_distance_aligned(P.data(), Pp.data(), n, k),
                 WithinAbs(pdd_emd(P.data(), Pp.data(), n, k), 1e-12));
}

TEST_CASE("dedup: aligned-PDD tier separates structures AMD can't", "[pdd][dedup]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;              // AMD tier always "matches" (disabled as a gate)
    cfg.pdd_tol = 0.1;              // aligned-PDD gate at 0.1 Å
    cfg.use_powder = false;         // no pattern on these results; isolate the PDD tier
    cfg.pdd_rows = rows; cfg.pdd_k = kk;
    StructureDedup dd(cfg);

    auto mkR = [](int seed) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.energy = 1.0; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;  // identical AMD
        return r;
    };
    std::vector<double> A(rows*kk), B(rows*kk), C(rows*kk);
    for (int i = 0; i < rows*kk; ++i) { A[i] = 1.0 + 0.01*i; B[i] = A[i] + 0.02; C[i] = A[i] + 1.0; }

    // add A
    REQUIRE(dd.find_duplicate_or_add(mkR(1), A.data(), rows*kk) < 0);  // unique
    // B is within pdd_tol of A -> duplicate of seed 1
    REQUIRE(dd.find_duplicate_or_add(mkR(2), B.data(), rows*kk) == 1);
    // C is far in PDD (1.0 >> 0.1) -> unique despite identical AMD/energy/density
    REQUIRE(dd.find_duplicate_or_add(mkR(3), C.data(), rows*kk) < 0);

    // With the PDD gate off, C collapses onto A (identical AMD).
    DedupConfig cfg2 = cfg; cfg2.pdd_tol = 0.0;
    StructureDedup dd2(cfg2);
    REQUIRE(dd2.find_duplicate_or_add(mkR(1), A.data(), rows*kk) < 0);
    REQUIRE(dd2.find_duplicate_or_add(mkR(3), C.data(), rows*kk) == 1);  // dup (AMD only)
}

// The stage compares PDDs by RMS over every entry. One entry 0.3 A off in a
// 3 x 4 PDD is an RMS of 0.087 A, but a mean row maximum of 0.1 A.
TEST_CASE("dedup: the PDD stage compares by RMS", "[pdd][dedup]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;
    cfg.pdd_tol = 0.09;
    cfg.use_powder = false;
    cfg.pdd_rows = rows; cfg.pdd_k = kk;
    StructureDedup dd(cfg);
    auto mkR = [](int seed) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.energy = 1.0; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;
        return r;
    };
    std::vector<double> A(rows * kk), B;
    for (int i = 0; i < rows * kk; ++i) A[i] = 1.0 + 0.1 * i;
    B = A;
    B[5] += 0.3;
    REQUIRE(dd.find_duplicate_or_add(mkR(1), A.data(), rows * kk) < 0);
    CHECK(dd.find_duplicate_or_add(mkR(2), B.data(), rows * kk) == 1);
}

// A symmetric molecule placed through one of its symmetry operations is the
// same structure with its atoms relabelled: the PDD has the same rows in a
// different order. Rows 1 and 2 swapped here stand for two such atoms.
TEST_CASE("dedup: the PDD stage sees through a symmetric molecule's relabelling",
          "[pdd][dedup]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;
    cfg.pdd_tol = 0.01;
    cfg.use_powder = false;
    cfg.pdd_rows = rows; cfg.pdd_k = kk;
    auto mkR = [](int seed) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.energy = 1.0; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;
        return r;
    };
    std::vector<double> A(rows * kk), B(rows * kk);
    for (int i = 0; i < rows * kk; ++i) A[i] = 1.0 + 0.1 * i;
    const int swap[3] = {0, 2, 1};
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < kk; ++j) B[r * kk + j] = A[swap[r] * kk + j];

    SECTION("rows in order miss it") {
        StructureDedup dd(cfg);
        REQUIRE(dd.find_duplicate_or_add(mkR(1), A.data(), rows * kk) < 0);
        CHECK(dd.find_duplicate_or_add(mkR(2), B.data(), rows * kk) < 0);
    }
    SECTION("the molecule's relabellings find it") {
        cfg.pdd_blocks = {DedupConfig::PddBlock{0, {{0, 1, 2}, {0, 2, 1}}}};
        StructureDedup dd(cfg);
        REQUIRE(dd.find_duplicate_or_add(mkR(1), A.data(), rows * kk) < 0);
        CHECK(dd.find_duplicate_or_add(mkR(2), B.data(), rows * kk) == 1);
    }
}

// The AMD bound and the early exit in front of the PDD stage must not change
// its verdict: duplicate exactly when the RMS distance is below the tolerance.
TEST_CASE("dedup: the PDD stage's shortcuts keep its verdict", "[pdd][dedup]") {
    const int rows = 8;
    std::mt19937 rng(20261019);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::uniform_real_distribution<double> scale(-4.0, -1.0);
    auto with_amd = [&](int seed, const std::vector<double>& pdd) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.energy = 1.0; r.density = 1.0;
        for (int j = 0; j < AMD_K; ++j) {
            double m = 0;
            for (int i = 0; i < rows; ++i) m += pdd[i * AMD_K + j];
            r.amd[j] = m / rows;
        }
        return r;
    };
    for (int pair = 0; pair < 200; ++pair) {
        std::vector<double> A(rows * AMD_K), B(rows * AMD_K);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < AMD_K; ++j) A[i * AMD_K + j] = 1.0 + 0.05 * j + 0.01 * i;
        const double sigma = std::pow(10.0, scale(rng));
        double s = 0;
        for (size_t i = 0; i < A.size(); ++i) {
            B[i] = A[i] + sigma * noise(rng);
            s += (B[i] - A[i]) * (B[i] - A[i]);
        }
        const double rms = std::sqrt(s / A.size());
        for (double tol : {rms * 0.5, rms * 0.999, rms * 1.001, rms * 2.0}) {
            DedupConfig cfg;
            cfg.amd_tol = 1e9;
            cfg.use_powder = false;
            cfg.pdd_tol = tol;
            cfg.pdd_rows = rows; cfg.pdd_k = AMD_K;
            StructureDedup dd(cfg);
            REQUIRE(dd.find_duplicate_or_add(with_amd(1, A), A.data(), rows * AMD_K) < 0);
            INFO("pair " << pair << ", rms " << rms << ", tol " << tol);
            CHECK((dd.find_duplicate_or_add(with_amd(2, B), B.data(), rows * AMD_K) == 1)
                  == (rms < tol));
        }
    }
}

TEST_CASE("dedup keeps the lowest-energy representative online", "[pdd][dedup]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;              // AMD never gates
    cfg.pdd_tol = 0.1;              // same basin iff aligned-PDD within 0.1 Å
    cfg.use_powder = false;         // no pattern on these results; isolate the PDD tier
    cfg.pdd_rows = rows; cfg.pdd_k = kk;
    StructureDedup dd(cfg);

    auto mkR = [](int seed, double E) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.stage = 2;
        r.energy = E; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;
        return r;
    };
    // One basin (same PDD), three arrivals at different energies.
    std::vector<double> P(rows*kk);
    for (int i = 0; i < rows*kk; ++i) P[i] = 1.0 + 0.01*i;

    // 1) higher-energy member arrives first -> unique, nothing displaced
    auto v = dd.classify(mkR(1, -50.0), P.data(), rows*kk);
    REQUIRE(v.is_unique());
    REQUIRE_FALSE(v.displaced());

    // 2) lower-energy member of the SAME basin -> becomes representative,
    //    reports seed 1 (and its key) as displaced
    v = dd.classify(mkR(2, -52.0), P.data(), rows*kk);
    REQUIRE(v.is_unique());              // r is the representative now
    REQUIRE(v.demoted_seed == 1);
    REQUIRE(v.demoted_stage == 2);
    REQUIRE(v.demoted_sg == 14);

    // 3) a still-higher-energy member -> duplicate of the NEW rep (seed 2),
    //    nothing displaced
    v = dd.classify(mkR(3, -49.0), P.data(), rows*kk);
    REQUIRE(v.duplicate_of == 2);
    REQUIRE_FALSE(v.displaced());

    // 4) an even-lower-energy member -> displaces seed 2
    v = dd.classify(mkR(4, -55.0), P.data(), rows*kk);
    REQUIRE(v.is_unique());
    REQUIRE(v.demoted_seed == 2);

    // Four arrivals, one basin: a swap must not bump the unique count.
    REQUIRE(dd.n_unique() == 1);
}

// The representative must not depend on arrival order (the GPU harvest
// schedule under streaming).
TEST_CASE("the kept representative does not depend on arrival order",
          "[pdd][dedup]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;
    cfg.pdd_tol = 0.1;
    cfg.use_powder = false;
    cfg.pdd_rows = rows; cfg.pdd_k = kk;

    auto mkR = [](int seed, double E) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.stage = 2;
        r.energy = E; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;
        return r;
    };
    std::vector<double> P(rows*kk);
    for (int i = 0; i < rows*kk; ++i) P[i] = 1.0 + 0.01*i;

    // Same three members, best-last then best-first, replaying the swaps.
    auto representative_after = [&](std::vector<std::pair<int,double>> arrivals) {
        StructureDedup dd(cfg);
        int rep = -1;
        for (auto [seed, E] : arrivals) {
            auto v = dd.classify(mkR(seed, E), P.data(), rows*kk);
            if (v.is_unique()) rep = seed;
        }
        REQUIRE(dd.n_unique() == 1);
        return rep;
    };

    const int worst_first = representative_after({{1,-50.0},{2,-52.0},{3,-55.0}});
    const int best_first  = representative_after({{3,-55.0},{2,-52.0},{1,-50.0}});
    const int shuffled    = representative_after({{2,-52.0},{3,-55.0},{1,-50.0}});
    REQUIRE(worst_first == 3);   // seed 3 has the lowest energy
    REQUIRE(best_first == 3);
    REQUIRE(shuffled == 3);
}

// With cross_sg the displaced row's key (seed, space_group, stage) must carry
// its own space group, not the arrival's.
TEST_CASE("a cross-space-group displacement reports the displaced row's own key",
          "[pdd][dedup]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;
    cfg.pdd_tol = 0.1;
    cfg.use_powder = false;
    cfg.cross_sg = true;             // one bin space across space groups
    cfg.pdd_rows = rows; cfg.pdd_k = kk;
    StructureDedup dd(cfg);

    auto mkR = [](int seed, int sg, double E) {
        StructureResult r{};
        r.seed = seed; r.space_group = sg; r.stage = 2;
        r.energy = E; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;
        return r;
    };
    std::vector<double> P(rows*kk);
    for (int i = 0; i < rows*kk; ++i) P[i] = 1.0 + 0.01*i;

    REQUIRE(dd.classify(mkR(1, 14, -50.0), P.data(), rows*kk).is_unique());
    const auto v = dd.classify(mkR(2, 19, -52.0), P.data(), rows*kk);
    REQUIRE(v.is_unique());
    REQUIRE(v.demoted_seed == 1);
    REQUIRE(v.demoted_sg == 14);     // the displaced row's SG, not the arrival's 19
    REQUIRE(dd.n_unique() == 1);
}

// A structure's stage-2, stage-3 and stage-0 rows share descriptors but not an
// energy scale, so they must never match each other.
TEST_CASE("dedup never matches across stages", "[pdd][dedup]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;              // AMD never gates
    cfg.pdd_tol = 0.1;
    cfg.use_powder = false;         // no pattern on these results; isolate the PDD tier
    cfg.pdd_rows = rows; cfg.pdd_k = kk;
    StructureDedup dd(cfg);

    auto mkR = [](int seed, int stage, double E) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.stage = stage;
        r.energy = E; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;
        return r;
    };
    std::vector<double> P(rows*kk);
    for (int i = 0; i < rows*kk; ++i) P[i] = 1.0 + 0.01*i;

    // Same descriptors and energy, different stage: each stage gets its own
    // representative.
    REQUIRE(dd.find_duplicate_or_add(mkR(1, 2, -50.0), P.data(), rows*kk) < 0);
    REQUIRE(dd.find_duplicate_or_add(mkR(1, 3, -50.0), P.data(), rows*kk) < 0);
    REQUIRE(dd.find_duplicate_or_add(mkR(1, 0, -50.0), P.data(), rows*kk) < 0);

    // Within a stage, dedup still works.
    REQUIRE(dd.find_duplicate_or_add(mkR(2, 2, -50.0), P.data(), rows*kk) == 1);
    REQUIRE(dd.find_duplicate_or_add(mkR(2, 3, -50.0), P.data(), rows*kk) == 1);

    // matches() states the same invariant for direct callers.
    StructureDedup::Entry e{};
    e.seed = 1; e.space_group = 14; e.stage = 2;
    e.energy = -50.0; e.density = 1.0;
    for (int i = 0; i < AMD_K; ++i) e.amd[i] = 0.0;
    e.pdd.assign(P.begin(), P.end());
    REQUIRE(dd.matches(mkR(9, 2, -50.0), e, P.data()));
    REQUIRE_FALSE(dd.matches(mkR(9, 3, -50.0), e, P.data()));
}

// The stage-3 gate (SGWorker::set_pet_dedup) relaxes only structures whose
// basin a rank-local index has not seen; rank 0's global index sets the unique
// flags. The gate must cover every basin with one relaxation per basin per rank.
// It does not guarantee the covered structure is rank 0's representative (first
// arrival vs deepest); the last section pins that gap, which
// `klasp rerank --top-up` closes. Randomised, since the hazard is ordering.
TEST_CASE("the rank-local stage-3 gate covers every basin at one relax per rank",
          "[pdd][dedup]") {
    const int rows = 2, kk = 4;
    const int n_ranks = 8, n_basins = 12, n_structs = 600;

    DedupConfig cfg;
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;              // AMD never gates
    cfg.pdd_tol = 0.1;              // same basin iff aligned-PDD within 0.1 A
    cfg.use_powder = false;         // isolate the PDD tier
    cfg.pdd_rows = rows; cfg.pdd_k = kk;

    // One signature per basin, spaced far enough apart that matching is an
    // equivalence relation, as the production thresholds assume.
    std::vector<std::vector<double>> sig(n_basins, std::vector<double>(rows * kk));
    for (int b = 0; b < n_basins; ++b)
        for (int i = 0; i < rows * kk; ++i) sig[b][i] = 1.0 + 10.0 * b + 0.001 * i;

    StructureDedup global(cfg);
    std::vector<StructureDedup> local(n_ranks);
    for (auto& d : local) d.configure(cfg);

    std::mt19937 rng(20260810);
    std::uniform_int_distribution<int> pick_basin(0, n_basins - 1);
    std::uniform_int_distribution<int> pick_rank(0, n_ranks - 1);
    // Basin members differ in energy only by optimiser residue, well inside one
    // energy bin, so binning cannot split a basin.
    std::uniform_real_distribution<double> jitter(-0.4, 0.4);
    auto basin_energy = [](int b) { return -60.0 + 1.5 * b; };

    // Per structure: its basin and whether its rank's gate relaxed it.
    struct Row { int basin; bool relaxed; };
    std::vector<Row> made;
    made.reserve(n_structs);
    std::vector<int> rep_of_basin(n_basins, -1);   // current global representative

    for (int s = 0; s < n_structs; ++s) {
        const int b = pick_basin(rng);
        StructureResult r{};
        r.seed = s; r.space_group = 14; r.stage = STAGE_MULTIPOLE;
        r.energy = basin_energy(b) + jitter(rng); r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;

        // The gate, exactly as SGWorker applies it.
        const int rk = pick_rank(rng);
        const bool relaxed = !local[rk].seen_basin(r, sig[b].data(), rows * kk);
        made.push_back({b, relaxed});

        // Rank 0's index, same order; it keeps the deepest member.
        if (global.classify(r, sig[b].data(), rows * kk).duplicate_of < 0)
            rep_of_basin[b] = s;
    }

    REQUIRE(global.n_unique() == n_basins);

    // Coverage: every basin was relaxed at least once.
    std::vector<int> relaxed_in_basin(n_basins, 0);
    int n_relaxed = 0;
    for (const auto& m : made)
        if (m.relaxed) { ++n_relaxed; ++relaxed_in_basin[m.basin]; }
    for (int b = 0; b < n_basins; ++b) {
        INFO("basin " << b);
        CHECK(relaxed_in_basin[b] > 0);
    }

    // Cost: exactly one relaxation per basin per rank (every rank sees every
    // basin here).
    CHECK(n_relaxed == n_ranks * n_basins);
    CHECK(n_relaxed < n_structs / 3);   // the saving is real, not incidental

    // The gap the top-up exists for: rank 0's representative (deepest) is in
    // general not the gate's (first arrival).
    int reps_without_a_row = 0;
    for (int b = 0; b < n_basins; ++b) {
        REQUIRE(rep_of_basin[b] >= 0);
        if (!made[rep_of_basin[b]].relaxed) ++reps_without_a_row;
    }
    CHECK(reps_without_a_row > 0);
}

// The production config decides duplicates in the powder tier (cosine plus
// CDTW), with mol-cspy's thresholds; check both obvious cases.
TEST_CASE("dedup: the powder tier is what decides in the production config",
          "[pdd][dedup][powder]") {
    const int rows = 3, kk = 4;
    DedupConfig cfg;                       // defaults: use_powder on, cos 0.80, cdtw 10
    cfg.energy_bin = 5.0; cfg.density_bin = 0.05;
    cfg.amd_tol = 1e9;                     // let everything reach the powder tier
    cfg.pdd_tol = 0.0;                     // and put the decision entirely on it
    cfg.pdd_rows = rows; cfg.pdd_k = kk;
    REQUIRE(cfg.use_powder);
    StructureDedup dd(cfg);

    // Two Gaussian peaks; `shift` moves them in 2-theta bins.
    auto pattern = [](StructureResult& r, int shift) {
        for (int i = 0; i < POWDER_N_BINS; ++i) r.powder[i] = 0.0f;
        double total = 0;
        for (int i = 0; i < POWDER_N_BINS; ++i) {
            const double d1 = i - (POWDER_N_BINS * 0.25 + shift);
            const double d2 = i - (POWDER_N_BINS * 0.55 + shift);
            const double v = std::exp(-d1*d1/8.0) + 0.6 * std::exp(-d2*d2/12.0);
            r.powder[i] = static_cast<float>(v);
            total += v;
        }
        for (int i = 0; i < POWDER_N_BINS; ++i) r.powder[i] /= static_cast<float>(total);
    };
    auto mkR = [&](int seed, int shift) {
        StructureResult r{};
        r.seed = seed; r.space_group = 14; r.stage = 2;
        r.energy = 1.0; r.density = 1.0;
        for (int i = 0; i < AMD_K; ++i) r.amd[i] = 0.0;
        pattern(r, shift);
        return r;
    };
    std::vector<double> pdd(rows*kk, 1.0);

    REQUIRE(dd.find_duplicate_or_add(mkR(1, 0), pdd.data(), rows*kk) < 0);   // first is unique
    // Same pattern back again: the tier has to merge it.
    REQUIRE(dd.find_duplicate_or_add(mkR(2, 0), pdd.data(), rows*kk) == 1);
    // Peaks moved far enough that neither the cosine nor the warp can follow.
    REQUIRE(dd.find_duplicate_or_add(mkR(3, POWDER_N_BINS / 5), pdd.data(), rows*kk) < 0);
}

// Powder patterns are stored as their nonzero bins; the encoding must be
// lossless.
TEST_CASE("powder blobs round trip through the sparse encoding", "[powder]") {
    auto roundtrip = [](const std::vector<float>& p) {
        const std::vector<char> blob = encode_powder(p.data(), static_cast<int>(p.size()));
        std::vector<float> back(p.size(), -1.0f);
        REQUIRE(decode_powder(blob.data(), blob.size(), back.data(),
                              static_cast<int>(p.size())));
        for (std::size_t i = 0; i < p.size(); ++i)
            REQUIRE(back[i] == p[i]);          // exact: it is not lossy
        return blob.size();
    };

    SECTION("a typical sparse pattern is smaller than dense") {
        std::vector<float> p(POWDER_N_BINS, 0.0f);
        for (int i = 0; i < POWDER_N_BINS; i += 4) p[i] = 0.001f * (i % 37);
        const std::size_t n = roundtrip(p);
        CHECK(n < POWDER_N_BINS * sizeof(float));
    }
    SECTION("all zero") {
        roundtrip(std::vector<float>(POWDER_N_BINS, 0.0f));
    }
    SECTION("fully dense falls back rather than growing") {
        std::vector<float> p(POWDER_N_BINS);
        for (int i = 0; i < POWDER_N_BINS; ++i) p[i] = 1.0f + i;
        const std::size_t n = roundtrip(p);
        CHECK(n == POWDER_N_BINS * sizeof(float));
    }
    SECTION("a single nonzero bin, at each end") {
        std::vector<float> p(POWDER_N_BINS, 0.0f);
        p[0] = 2.5f;                       roundtrip(p);
        p[0] = 0.0f; p[POWDER_N_BINS-1] = 2.5f; roundtrip(p);
    }
}

// Raw-float (dense) blobs must keep loading.
TEST_CASE("a dense blob written by an older klasp still decodes", "[powder]") {
    std::vector<float> p(POWDER_N_BINS);
    for (int i = 0; i < POWDER_N_BINS; ++i) p[i] = 0.01f * i;
    std::vector<char> legacy(POWDER_N_BINS * sizeof(float));
    std::memcpy(legacy.data(), p.data(), legacy.size());

    std::vector<float> back(POWDER_N_BINS, -1.0f);
    REQUIRE(decode_powder(legacy.data(), legacy.size(), back.data(), POWDER_N_BINS));
    for (int i = 0; i < POWDER_N_BINS; ++i) REQUIRE(back[i] == p[i]);
}

TEST_CASE("a truncated powder blob is refused, not half-read", "[powder]") {
    std::vector<float> p(POWDER_N_BINS, 0.0f);
    for (int i = 0; i < POWDER_N_BINS; i += 8) p[i] = 0.5f;
    std::vector<char> blob = encode_powder(p.data(), POWDER_N_BINS);
    REQUIRE(blob.size() < POWDER_N_BINS * sizeof(float));   // it went sparse

    std::vector<float> back(POWDER_N_BINS, -1.0f);
    blob.resize(blob.size() - 3);
    CHECK_FALSE(decode_powder(blob.data(), blob.size(), back.data(), POWDER_N_BINS));
    CHECK(back[0] == -1.0f);                                // untouched
    CHECK_FALSE(decode_powder(blob.data(), 2, back.data(), POWDER_N_BINS));
    CHECK_FALSE(decode_powder(nullptr, 16, back.data(), POWDER_N_BINS));
}

// Full-matrix DTW straight from the definition, cells outside the band infinite.
static float reference_cdtw(const float* a, const float* b, int n, int r) {
    constexpr double INF = std::numeric_limits<double>::infinity();
    std::vector<double> D(static_cast<size_t>(n) * n, INF);
    auto at = [&](int i, int j) { return (i < 0 || j < 0) ? INF : D[i * n + j]; };
    for (int i = 0; i < n; ++i)
        for (int j = std::max(0, i - r); j <= std::min(n - 1, i + r); ++j) {
            const double best = (i == 0 && j == 0)
                ? 0.0 : std::min(at(i - 1, j - 1), std::min(at(i - 1, j), at(i, j - 1)));
            D[i * n + j] = std::fabs(double(a[i]) - double(b[j])) + best;
        }
    return static_cast<float>(D[(n - 1) * n + (n - 1)]);
}

TEST_CASE("CDTW matches the full-matrix definition", "[powder][dedup]") {
    std::mt19937 rng(20261016);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int n : {2, 3, 7, 64, 300}) {
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; ++i) {
            a[i] = u(rng) < 0.28f ? u(rng) : 0.0f;
            b[i] = u(rng) < 0.28f ? u(rng) : 0.0f;
        }
        for (int r : {1, 2, 5, 9, POWDER_CDTW_MAX_BAND, POWDER_CDTW_MAX_BAND + 3, n - 1, n + 5}) {
            if (r < 1) continue;
            INFO("n = " << n << ", r = " << r);
            CHECK(powder_cdtw_sakoe_chiba(a.data(), b.data(), n, r)
                  == reference_cdtw(a.data(), b.data(), n, r));
        }
    }
}

// Scratch lives on the stack up to POWDER_CDTW_MAX_BAND and on the heap past it.
TEST_CASE("CDTW agrees across its scratch paths", "[powder][dedup]") {
    std::mt19937 rng(20250810);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> a(POWDER_N_BINS), b(POWDER_N_BINS);
    for (int i = 0; i < POWDER_N_BINS; ++i) {
        // ~28% nonzero, which is what a real pattern looks like (powder.hpp).
        a[i] = u(rng) < 0.28f ? u(rng) : 0.0f;
        b[i] = u(rng) < 0.28f ? u(rng) : 0.0f;
    }

    CHECK(powder_cdtw_sakoe_chiba(a.data(), b.data(), POWDER_N_BINS, 0)
              == powder_l1_distance(a.data(), b.data(), POWDER_N_BINS));

    // A wider band can only find a cheaper warping, never a dearer one.
    float prev = powder_cdtw_sakoe_chiba(a.data(), b.data(), POWDER_N_BINS, 1);
    for (int r = 2; r <= POWDER_CDTW_MAX_BAND + 4; ++r) {
        const float d = powder_cdtw_sakoe_chiba(a.data(), b.data(), POWDER_N_BINS, r);
        INFO("band r = " << r);
        CHECK(d <= prev * (1.0f + 1e-6f));
        CHECK(std::isfinite(d));
        prev = d;
    }

    // A pattern is a zero-distance match with itself under every band, and
    // powder_cdtw_within agrees with a direct sweep either side of a threshold.
    for (int r : {1, POWDER_CDTW_MAX_BAND, POWDER_CDTW_MAX_BAND + 1})
        CHECK(powder_cdtw_sakoe_chiba(a.data(), a.data(), POWDER_N_BINS, r) == 0.0f);
    const float best = powder_cdtw_sakoe_chiba(a.data(), b.data(), POWDER_N_BINS, 9);
    CHECK(powder_cdtw_within(a.data(), b.data(), POWDER_N_BINS, 10, best));
    CHECK_FALSE(powder_cdtw_within(a.data(), b.data(), POWDER_N_BINS, 10,
                                   best * 0.5f));
}

// powder_cdtw_within (one early-abandoning pass at the widest band) gives the
// same verdict as mol-cspy's r = 0..band-1 sweep, on near-duplicates and on
// thresholds either side of the distance.
TEST_CASE("CDTW single-band verdict equals the band sweep", "[powder][dedup]") {
    std::mt19937 rng(20260923);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    constexpr int kBand = 10;
    auto sweep_min = [](const float* a, const float* b) {
        float best = powder_l1_distance(a, b, POWDER_N_BINS);
        for (int r = 1; r < kBand; ++r)
            best = std::min(best, powder_cdtw_sakoe_chiba(a, b, POWDER_N_BINS, r));
        return best;
    };
    std::vector<float> a(POWDER_N_BINS), b(POWDER_N_BINS);
    for (int pair = 0; pair < 200; ++pair) {
        for (int i = 0; i < POWDER_N_BINS; ++i) a[i] = u(rng) < 0.28f ? u(rng) : 0.0f;
        if (pair % 2 == 0) {
            // A shifted, slightly perturbed copy: the case a warp actually helps.
            const int shift = 1 + pair % 5;
            for (int i = 0; i < POWDER_N_BINS; ++i) {
                const int j = std::min(POWDER_N_BINS - 1, i + shift);
                b[i] = a[j] * (0.9f + 0.2f * u(rng));
            }
        } else {
            for (int i = 0; i < POWDER_N_BINS; ++i) b[i] = u(rng) < 0.28f ? u(rng) : 0.0f;
        }
        const float widest = powder_cdtw_sakoe_chiba(a.data(), b.data(), POWDER_N_BINS, kBand - 1);
        const float swept = sweep_min(a.data(), b.data());
        INFO("pair " << pair);
        CHECK(swept == widest);
        // Both sides of the decision, plus the value itself.
        for (float thr : {widest * 0.5f, widest * 0.999f, widest, widest * 1.001f, widest * 2.0f}) {
            CHECK(powder_cdtw_within(a.data(), b.data(), POWDER_N_BINS, kBand, thr)
                  == (swept <= thr));
        }
    }
}

TEST_CASE("CDTW envelope bound never rejects a pair the CDTW accepts", "[powder][dedup]") {
    std::mt19937 rng(20261017);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> a(POWDER_N_BINS), b(POWDER_N_BINS);
    int rejected = 0;
    for (int pair = 0; pair < 200; ++pair) {
        for (int i = 0; i < POWDER_N_BINS; ++i) a[i] = u(rng) < 0.28f ? u(rng) : 0.0f;
        const int shift = pair % 7;
        for (int i = 0; i < POWDER_N_BINS; ++i) {
            const int j = std::min(POWDER_N_BINS - 1, i + shift);
            b[i] = pair % 3 == 2 ? (u(rng) < 0.28f ? u(rng) : 0.0f)
                                 : a[j] * (0.9f + 0.2f * u(rng));
        }
        for (int band : {1, 2, 10}) {
            const PowderEnvelope env = powder_envelope(a.data(), band - 1);
            const float d = band > 1
                ? powder_cdtw_sakoe_chiba(a.data(), b.data(), POWDER_N_BINS, band - 1)
                : powder_l1_distance(a.data(), b.data(), POWDER_N_BINS);
            for (float thr : {d * 0.5f, d, d * 2.0f, 10.0f}) {
                INFO("pair " << pair << ", band " << band << ", thr " << thr);
                const bool accepts = powder_cdtw_within(b.data(), a.data(), POWDER_N_BINS, band, thr);
                const bool reach = powder_cdtw_can_reach(b.data(), env, POWDER_N_BINS, thr);
                if (accepts) CHECK(reach);
                if (!reach) ++rejected;
            }
        }
    }
    CHECK(rejected > 0);
}
