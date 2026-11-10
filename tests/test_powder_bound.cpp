// powder_cosine_can_reach must never reject a pair the cosine would accept.
#include "klasp/powder.hpp"
#include "klasp/dedup.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>
#include <vector>

using namespace klasp;

namespace {

// Non-negative binned intensities: a handful of Gaussian peaks on a sparse grid.
std::vector<float> make_pattern(std::mt19937& rng, int n_peaks) {
    std::vector<float> p(POWDER_N_BINS, 0.0f);
    std::uniform_int_distribution<int> where(0, POWDER_N_BINS - 1);
    std::uniform_real_distribution<float> height(0.1f, 1.0f);
    for (int k = 0; k < n_peaks; ++k) {
        const int c = where(rng);
        const float h = height(rng);
        for (int d = -6; d <= 6; ++d) {
            const int i = c + d;
            if (i < 0 || i >= POWDER_N_BINS) continue;
            p[i] += h * std::exp(-0.5f * float(d * d) / 4.0f);
        }
    }
    return p;
}

}  // namespace

TEST_CASE("the cosine bound never rejects a pair the cosine accepts",
          "[powder][dedup]") {
    std::mt19937 rng(20260828);
    std::vector<std::vector<float>> pats;
    std::vector<PowderNorms> norms;
    for (int i = 0; i < 300; ++i) {
        pats.push_back(make_pattern(rng, 3 + (i % 40)));
        norms.push_back(powder_group_norms(pats.back().data()));
    }

    // Near-duplicates, the pairs that must survive the bound.
    std::normal_distribution<float> jitter(0.0f, 0.01f);
    for (int i = 0; i < 100; ++i) {
        std::vector<float> q = pats[i];
        for (auto& v : q) v = std::max(0.0f, v + jitter(rng));
        pats.push_back(q);
        norms.push_back(powder_group_norms(q.data()));
    }

    int n_accepted = 0, n_rejected = 0, n_survived_bound = 0;
    for (double thresh : {0.5, 0.8, 0.95, 0.99}) {
        for (size_t i = 0; i < pats.size(); ++i) {
            for (size_t j = i + 1; j < pats.size(); ++j) {
                const double cos = powder_cosine_similarity(
                    pats[i].data(), pats[j].data(), POWDER_N_BINS);
                const bool can = powder_cosine_can_reach(norms[i], norms[j], thresh);
                if (cos >= thresh) {
                    REQUIRE(can);
                    ++n_accepted;
                }
                if (can) ++n_survived_bound; else ++n_rejected;
            }
        }
    }
    REQUIRE(n_accepted > 0);        // the invariant was actually exercised
    REQUIRE(n_rejected > 0);        // ...and the bound does reject things
    REQUIRE(n_rejected > n_survived_bound);
}

TEST_CASE("the bound is symmetric and admits a pattern against itself",
          "[powder][dedup]") {
    std::mt19937 rng(7);
    const auto a = make_pattern(rng, 12);
    const auto b = make_pattern(rng, 12);
    const auto na = powder_group_norms(a.data());
    const auto nb = powder_group_norms(b.data());
    // cos(a,a) == 1, so no threshold in [0,1] may reject it.
    REQUIRE(powder_cosine_can_reach(na, na, 1.0));
    REQUIRE(powder_cosine_can_reach(na, nb, 0.5) ==
            powder_cosine_can_reach(nb, na, 0.5));
    // The cosine returns 0 for an empty pattern, so the bound must not claim it
    // reaches a positive threshold.
    PowderNorms zero{};
    REQUIRE_FALSE(powder_cosine_can_reach(zero, na, 0.5));
}

TEST_CASE("dedup verdicts do not depend on the cosine bound", "[powder][dedup]") {
    // Through the real classify(), where the bound sits inside matches().
    std::mt19937 rng(99);
    DedupConfig cfg;
    cfg.amd_tol = 1e9;              // isolate the powder tier
    cfg.pdd_tol = 0.0;
    cfg.use_powder = true;
    cfg.powder_cdtw_thresh = 0.0;   // cosine decides, no CDTW stage
    StructureDedup dd(cfg);

    // Five recurring basins plus singletons, so a bound that rejects everything
    // would be caught.
    std::vector<std::vector<float>> basins;
    for (int b = 0; b < 5; ++b) {
        std::mt19937 brng(1000 + b);
        basins.push_back(make_pattern(brng, 10));
    }

    std::vector<StructureResult> rs;
    for (int i = 0; i < 120; ++i) {
        StructureResult r{};
        r.seed = i; r.space_group = 14; r.stage = 2;
        r.energy = -50.0 + 0.05 * i; r.density = 1.2;
        if (i % 3 == 0) {
            std::copy(basins[(i / 3) % 5].begin(), basins[(i / 3) % 5].end(), r.powder);
        } else {
            std::mt19937 prng(5000 + i);
            const auto p = make_pattern(prng, 10);
            std::copy(p.begin(), p.end(), r.powder);
        }
        rs.push_back(r);
    }

    int n_dup = 0;
    for (const auto& r : rs)
        if (!dd.classify(r).is_unique()) ++n_dup;

    REQUIRE(n_dup > 0);
    REQUIRE(dd.n_unique() < static_cast<int>(rs.size()));
}
