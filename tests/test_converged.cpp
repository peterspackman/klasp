#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/database.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/dedup.hpp"
#include "klasp/descriptors.hpp"

#include <sqlite_modern_cpp.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinRel;

// `klasp converged` derives every figure from duplicate_of, which is a pointer
// to a representative, not a tally.

namespace {

struct TempDb {
    std::filesystem::path path;
    explicit TempDb(const char* tag) {
        path = std::filesystem::temp_directory_path() /
               ("klasp_converged_" + std::string(tag) + ".db");
        std::filesystem::remove(path);
    }
    ~TempDb() { std::filesystem::remove(path); }
    std::string str() const { return path.string(); }
};

StructureResult row(int seed, int sg, double energy) {
    StructureResult r{};
    r.seed = seed;
    r.space_group = sg;
    r.stage = 2;
    r.n_mols = 1;
    r.energy = energy;
    r.density = 1.2;
    for (int k = 0; k < 3; ++k) r.cell_params[k] = 10.0;
    for (int k = 3; k < 6; ++k) r.cell_params[k] = 90.0;
    return r;
}

// The command's own defaults, except what a test varies.
ConvergedResult run(const std::string& db, int tail_k = 0, int bootstrap = 0) {
    ConvergedOptions o;
    o.db_path = db;
    o.windows = {1.0, 2.0, 5.0, 10.0};
    o.tail_k = tail_k;
    o.n_bootstrap = bootstrap;
    o.stage = 2;
    return cmd_converged(o);
}

}  // namespace

TEST_CASE("Basin counts come from resolved duplicate chains", "[converged][db]") {
    // Databases before schema v4 can hold chained duplicate_of pointers; they
    // must be followed to the representative.
    TempDb t("chain");
    {
        CSPDatabase db;
        db.open(t.str());
        auto rep = row(0, 14, -50.0);
        db.insert_result(&rep, true);                  // the representative
        auto d1 = row(1, 14, -49.9);
        db.insert_result(&d1, false, 0, 14);           // -> 0, directly
        for (int seed = 2; seed < 6; ++seed) {         // -> 1 -> 0, a chain
            auto d = row(seed, 14, -49.8);
            db.insert_result(&d, false, 1, 14);
        }
        // A second basin nothing points through, as a control.
        auto rep2 = row(6, 14, -40.0);
        db.insert_result(&rep2, true);
        db.close();
    }

    const ConvergedResult r = run(t.str());
    REQUIRE(r.ok);
    REQUIRE(r.basins == 2);
    // 7 rows, including the four behind seed 1.
    REQUIRE(r.searches == 7);
    REQUIRE_THAT(r.min_energy, WithinRel(-50.0, 1e-12));
}

TEST_CASE("A duplicate pointing at nothing is dropped, not counted",
          "[converged][db]") {
    TempDb t("dangle");
    {
        CSPDatabase db;
        db.open(t.str());
        auto rep = row(0, 14, -50.0);
        db.insert_result(&rep, true);
        auto bad = row(1, 14, -49.0);
        db.insert_result(&bad, false, 999, 14);   // no such representative
        db.close();
    }
    const ConvergedResult r = run(t.str());
    REQUIRE(r.ok);
    REQUIRE(r.basins == 1);
    REQUIRE(r.searches == 1);       // the dangling row is not attributed anywhere
}

TEST_CASE("The stopping rule fires only once a window is re-found enough",
          "[converged][stats]") {
    // Boender-Rinnooy Kan: W_hat = w(N-1)/(N-w-2), stop when W_hat < w + 1/2.
    // Two basins each found many times is converged; found once each, not.
    SECTION("well re-found") {
        TempDb t("stop");
        {
            CSPDatabase db;
            db.open(t.str());
            int seed = 0;
            for (int basin = 0; basin < 2; ++basin) {
                const int rep_seed = seed++;
                auto rep = row(rep_seed, 14, -50.0 + 0.1 * basin);
                db.insert_result(&rep, true);
                for (int i = 0; i < 40; ++i) {
                    auto d = row(seed++, 14, -50.0 + 0.1 * basin);
                    db.insert_result(&d, false, rep_seed, 14);
                }
            }
            db.close();
        }
        const ConvergedResult r = run(t.str());
        REQUIRE(r.basins == 2);
        REQUIRE(r.searches == 82);
        // w=2, N=82: W_hat = 2*81/78 = 2.077 < 2.5.
        REQUIRE(r.converged_window == 1.0);
    }
    SECTION("every search its own basin") {
        TempDb t("nostop");
        {
            CSPDatabase db;
            db.open(t.str());
            for (int i = 0; i < 40; ++i) {
                auto rep = row(i, 14, -50.0 + 0.01 * i);
                db.insert_result(&rep, true);
            }
            db.close();
        }
        const ConvergedResult r = run(t.str());
        REQUIRE(r.basins == 40);
        REQUIRE(r.searches == 40);
        // N - w - 2 is negative: the estimator is undefined, not satisfied.
        REQUIRE(r.converged_window < 0);
    }
}

TEST_CASE("A non-physical basin is dropped with the searches behind it",
          "[converged][db]") {
    // A collapsed representative's duplicates must stay in the denominator.
    TempDb t("outlier");
    {
        CSPDatabase db;
        db.open(t.str());
        auto bad = row(0, 14, -5.0e10);
        db.insert_result(&bad, true);
        for (int seed = 1; seed < 21; ++seed) {
            auto d = row(seed, 14, -5.0e10);
            db.insert_result(&d, false, 0, 14);
        }
        auto good = row(100, 14, -50.0);
        db.insert_result(&good, true);
        for (int seed = 101; seed < 111; ++seed) {
            auto d = row(seed, 14, -49.99);
            db.insert_result(&d, false, 100, 14);
        }
        db.close();
    }
    const ConvergedResult r = run(t.str());
    REQUIRE(r.ok);
    REQUIRE(r.basins == 1);
    REQUIRE(r.searches == 11);
    REQUIRE_THAT(r.min_energy, WithinRel(-50.0, 1e-12));
}

TEST_CASE("An endpoint is only claimed when the tail supports one",
          "[converged][stats]") {
    // Energies drawn as E* + (i/n)^(1/alpha) have a floor at E*; the fit should
    // land near it, and no endpoint is reported with too few minima.
    SECTION("a tail with a floor") {
        TempDb t("endpoint");
        constexpr double kFloor = -60.0;
        {
            CSPDatabase db;
            db.open(t.str());
            for (int i = 0; i < 400; ++i) {
                const double u = (i + 1) / 400.0;
                const double e = kFloor + 10.0 * std::sqrt(u);   // alpha = 2
                auto rep = row(i, 14, e);
                db.insert_result(&rep, true);
                // Enough re-finds that the basin counts are well posed.
                for (int j = 0; j < 3; ++j) {
                    auto d = row(1000 + 4 * i + j, 14, e);
                    db.insert_result(&d, false, i, 14);
                }
            }
            db.close();
        }
        const ConvergedResult r = run(t.str(), 40, 0);
        REQUIRE(r.endpoint_ok);
        // The lowest sample is 0.5 kJ/mol above the floor; the fit should do better.
        REQUIRE(r.endpoint < r.min_energy);
        REQUIRE(std::abs(r.endpoint - kFloor) < 1.0);
    }
    SECTION("too few minima to fit") {
        TempDb t("tiny");
        {
            CSPDatabase db;
            db.open(t.str());
            for (int i = 0; i < 3; ++i) {
                auto rep = row(i, 14, -50.0 + i);
                db.insert_result(&rep, true);
            }
            db.close();
        }
        const ConvergedResult r = run(t.str());
        REQUIRE(r.ok);
        REQUIRE_FALSE(r.endpoint_ok);
    }
}

TEST_CASE("An empty landscape is reported, not crashed on", "[converged][db]") {
    TempDb t("empty");
    {
        CSPDatabase db;
        db.open(t.str());
        db.close();
    }
    const ConvergedResult r = run(t.str());
    REQUIRE_FALSE(r.ok);
}
