#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/flex_descriptors.hpp"
#include "klasp/dedup.hpp"
#include "klasp/constants.hpp"

#include <array>
#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;

// Descriptors for a geometry with no rigid-molecule description, as a PET
// relaxation leaves.

namespace {

// A simple cubic lattice of one element, `n` per side, edge `a` Bohr. AMD comes
// back in Angstrom.
CrystalGeometry cubic(int n, double a) {
    CrystalGeometry g;
    g.cell[0][0] = a * n; g.cell[1][1] = a * n; g.cell[2][2] = a * n;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                g.positions.push_back({i * a, j * a, k * a});
                g.atomic_numbers.push_back(6);
            }
    return g;
}

}  // namespace

TEST_CASE("AMD of a simple cubic lattice has the shells it should", "[flex]") {
    const double a = 3.0;
    const auto d = compute_flex_descriptors({cubic(3, a)});
    REQUIRE(d.amd.size() == 1);
    REQUIRE(d.amd[0].size() == static_cast<std::size_t>(AMD_K));

    // Every atom is equivalent, so the AMD is that atom's own neighbour shells:
    // 6 at a, then 12 at a*sqrt(2), then 8 at a*sqrt(3).
    const double aA = a * BOHR_TO_ANGSTROM;   // the descriptor is in Angstrom
    for (int i = 0; i < 6; ++i)
        CHECK_THAT(d.amd[0][i], WithinAbs(aA, 1e-9));
    for (int i = 6; i < 18; ++i)
        CHECK_THAT(d.amd[0][i], WithinAbs(aA * std::sqrt(2.0), 1e-9));
    for (int i = 18; i < 26; ++i)
        CHECK_THAT(d.amd[0][i], WithinAbs(aA * std::sqrt(3.0), 1e-9));
    // Monotonic by construction: sorted distances, averaged over equal rows.
    for (std::size_t i = 1; i < d.amd[0].size(); ++i)
        CHECK(d.amd[0][i] >= d.amd[0][i - 1] - 1e-12);
}

// A relaxed cell can carry atoms outside it; the descriptor must wrap them.
TEST_CASE("translating an atom by a lattice vector does not change the AMD",
          "[flex]") {
    const double a = 3.0;
    CrystalGeometry home = cubic(3, a);
    const double edge = a * 3;
    // Jitter off the ideal lattice, so powder peaks do not sit exactly on bin
    // edges where rounding moves them between bins.
    for (std::size_t i = 0; i < home.positions.size(); ++i)
        for (int d = 0; d < 3; ++d)
            home.positions[i][d] += 0.05 * std::sin(double(7 * i + 3 * d));

    CrystalGeometry drifted = home;
    // Push three atoms out of the cell, in different directions and by more
    // than one cell each.
    drifted.positions[0][0] += edge;
    drifted.positions[5][1] -= 2 * edge;
    drifted.positions[11][2] += 3 * edge;

    const auto ref = compute_flex_descriptors({home});
    const auto out = compute_flex_descriptors({drifted});
    REQUIRE(ref.amd.size() == 1);
    REQUIRE(out.amd.size() == 1);
    // After wrapping these are the same point set, so only rounding differs.
    for (std::size_t i = 0; i < ref.amd[0].size(); ++i)
        CHECK_THAT(out.amd[0][i], WithinAbs(ref.amd[0][i], 1e-12));

    REQUIRE(ref.powder[0].size() == out.powder[0].size());
    double worst = 0;
    for (std::size_t i = 0; i < ref.powder[0].size(); ++i)
        worst = std::max<double>(worst, std::fabs(ref.powder[0][i] - out.powder[0][i]));
    CHECK(worst < 1e-5);
}

TEST_CASE("a batch is described structure by structure", "[flex]") {
    // Two different lattices in one call must not bleed into each other.
    const auto d = compute_flex_descriptors({cubic(3, 3.0), cubic(3, 4.0)});
    REQUIRE(d.amd.size() == 2);
    CHECK_THAT(d.amd[0][0], WithinAbs(3.0 * BOHR_TO_ANGSTROM, 1e-9));
    CHECK_THAT(d.amd[1][0], WithinAbs(4.0 * BOHR_TO_ANGSTROM, 1e-9));
}

TEST_CASE("degenerate input is handled rather than indexed", "[flex]") {
    CHECK(compute_flex_descriptors({}).amd.empty());
    CrystalGeometry one;
    one.cell[0][0] = one.cell[1][1] = one.cell[2][2] = 10.0;
    one.positions.push_back({0, 0, 0});
    one.atomic_numbers.push_back(6);
    CHECK(compute_flex_descriptors({one}).amd.empty());   // a single atom has no neighbours
}

// The search radius is estimated from mean density and must be grown until the
// furthest neighbour used sits inside it, so a hopeless start converges on the
// same answer as a generous one.
TEST_CASE("too small a search radius is grown, not accepted", "[flex]") {
    const auto g = cubic(3, 3.0);
    const auto automatic = compute_flex_descriptors({g});
    const auto tiny      = compute_flex_descriptors({g}, false, /*cutoff_bohr=*/0.5);
    const auto generous  = compute_flex_descriptors({g}, false, /*cutoff_bohr=*/200.0);

    REQUIRE(automatic.amd.size() == 1);
    for (std::size_t i = 0; i < automatic.amd[0].size(); ++i) {
        CHECK_THAT(tiny.amd[0][i],     WithinAbs(generous.amd[0][i], 1e-9));
        CHECK_THAT(automatic.amd[0][i], WithinAbs(generous.amd[0][i], 1e-9));
    }
}

// A larger k only appends further neighbours; the shells k = 100 sees stay put.
TEST_CASE("a larger neighbour count extends the PDD rows", "[flex][pdd]") {
    const CrystalGeometry g = cubic(3, 3.0);
    const auto d100 = compute_flex_descriptors({g}, true, 0.0, 100);
    const auto d400 = compute_flex_descriptors({g}, true, 0.0, 400);
    REQUIRE(d100.n_asym == d400.n_asym);
    for (int i = 0; i < d100.n_asym; ++i) {
        for (int j = 0; j < 100; ++j)
            CHECK_THAT(d400.pdd[0][i * 400 + j], WithinAbs(d100.pdd[0][i * 100 + j], 1e-12));
        CHECK(d400.pdd[0][i * 400 + 399] > d400.pdd[0][i * 400 + 99]);
    }
    CHECK_THROWS(compute_flex_descriptors({g}, true, 0.0, 150));
}

// Folding averages each asymmetric-unit atom's PDD rows over its copies. The
// copies here are exact inversion images, so folded rows equal the originals.
TEST_CASE("PDD rows fold to one per asymmetric-unit atom", "[flex][pdd]") {
    CrystalGeometry g;
    const double cell[3][3] = {{7.1, 0.4, 0.9}, {0.0, 6.3, -0.7}, {0.0, 0.0, 8.2}};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) g.cell[r][c] = cell[r][c];
    const std::array<double, 3> a{1.0, 0.5, 0.3}, b{2.2, 1.1, 0.4};
    g.positions = {a, b, {-a[0], -a[1], -a[2]}, {-b[0], -b[1], -b[2]}};
    g.atomic_numbers = {6, 8, 6, 8};

    const auto rows = compute_flex_descriptors({g}, /*want_pdd=*/true);
    const auto folded = compute_flex_descriptors({g}, /*want_pdd=*/true, 0.0, AMD_K,
                                                 /*fold_rows=*/2);
    REQUIRE(rows.pdd.size() == 1);
    REQUIRE(rows.pdd[0].size() == 4u * AMD_K);
    REQUIRE(folded.pdd.size() == 1);
    REQUIRE(folded.pdd[0].size() == 2u * AMD_K);
    CHECK(folded.n_asym == 2);
    for (int j = 0; j < AMD_K; ++j) {
        CHECK_THAT(rows.pdd[0][2 * AMD_K + j], WithinAbs(rows.pdd[0][j], 1e-9));
        CHECK_THAT(rows.pdd[0][3 * AMD_K + j], WithinAbs(rows.pdd[0][AMD_K + j], 1e-9));
        CHECK_THAT(folded.pdd[0][j], WithinAbs(rows.pdd[0][j], 1e-9));
        CHECK_THAT(folded.pdd[0][AMD_K + j], WithinAbs(rows.pdd[0][AMD_K + j], 1e-9));
    }

    const auto misfit = compute_flex_descriptors({g}, true, 0.0, AMD_K, /*fold_rows=*/3);
    CHECK(misfit.pdd.empty());
}
