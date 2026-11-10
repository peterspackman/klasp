#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/vacuum.hpp"
#include "klasp/warnings.hpp"

#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;

// A vacuum slab is invisible to density, so the test is geometric: project
// every atom onto a lattice-plane normal and look for a gap.

namespace {

// n atoms spread evenly through the fractional range [0, occupied], so
// everything above `occupied` is empty.
std::vector<double> slab_along_c(int n, double occupied) {
    std::vector<double> f;
    f.reserve(3 * n);
    for (int i = 0; i < n; ++i) {
        const double z = occupied * double(i) / double(n);
        f.push_back(0.5 * ((i * 7) % 5) / 5.0);   // spread in a and b
        f.push_back(0.5 * ((i * 3) % 4) / 4.0);
        f.push_back(z);
    }
    return f;
}

}  // namespace

TEST_CASE("A cubic cell with half its c axis empty reports that gap", "[vacuum]") {
    const double cp[6] = {10, 10, 20, 90, 90, 90};
    const auto f = slab_along_c(40, 0.5);          // atoms in z in [0, 0.5)
    const VacuumSlab v = max_vacuum_gap(cp, f.data(), 40);
    // Half of a 20 A axis is empty, less one inter-atom step.
    REQUIRE_THAT(v.gap, WithinAbs(10.0, 0.6));
    REQUIRE(v.l != 0);                              // found on a plane normal to c
}

TEST_CASE("A well-filled cell reports no meaningful gap", "[vacuum]") {
    // A 5x5x5 grid: occupied in every direction, so no plane finds a void.
    const double cp[6] = {10, 10, 10, 90, 90, 90};
    std::vector<double> f;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 5; ++k)
                f.insert(f.end(), {i / 5.0, j / 5.0, k / 5.0});
    const VacuumSlab v = max_vacuum_gap(cp, f.data(), 125);
    REQUIRE(v.gap < WARN_VACUUM_GAP_A);
}

TEST_CASE("The gap is measured across the periodic wrap, not just inside the cell",
          "[vacuum]") {
    // A gap in the middle and one across the cell boundary are the same slab.
    const double cp[6] = {10, 10, 20, 90, 90, 90};
    std::vector<double> mid, edge;
    for (int i = 0; i < 30; ++i) {
        const double t = 0.4 * double(i) / 30.0;
        mid.insert(mid.end(),  {0.1, 0.2, 0.3 + t});          // occupied [0.3,0.7)
        edge.insert(edge.end(), {0.1, 0.2, std::fmod(0.8 + t, 1.0)});  // wraps
    }
    const double a = max_vacuum_gap(cp, mid.data(), 30).gap;
    const double b = max_vacuum_gap(cp, edge.data(), 30).gap;
    REQUIRE_THAT(a, WithinAbs(b, 1e-9));
}

TEST_CASE("A slab on a diagonal plane is found too", "[vacuum]") {
    // Layers on (1 1 0): only x+y is confined, to a band 0.3 wide, so no
    // axis-aligned plane sees the void. a = b = 14 makes d(110) = 9.9 A and the
    // empty band a 6.9 A slab.
    const double cp[6] = {14, 14, 10, 90, 90, 90};
    std::vector<double> f;
    for (int i = 0; i < 60; ++i) {
        const double x = double(i % 10) / 10.0;
        const double s = 0.30 * double(i / 10) / 6.0;   // x + y stays in [0, 0.30)
        double y = s - x;
        y -= std::floor(y);
        f.insert(f.end(), {x, y, double(i % 6) / 6.0});
    }
    const VacuumSlab v = max_vacuum_gap(cp, f.data(), 60);
    REQUIRE(v.gap > WARN_VACUUM_GAP_A);
    REQUIRE(v.h != 0);
    REQUIRE(v.k != 0);            // a plane involving both a and b, i.e. diagonal
}

TEST_CASE("The warning bit follows the gap", "[vacuum]") {
    const double cp[6] = {10, 10, 10, 90, 90, 90};
    REQUIRE((classify_warnings(cp, 1.2, 12.0) & WARN_VACUUM_SLAB) != 0);
    REQUIRE((classify_warnings(cp, 1.2, 1.5) & WARN_VACUUM_SLAB) == 0);
    // Not computed (-1) must not raise the flag.
    REQUIRE((classify_warnings(cp, 1.2, -1.0) & WARN_VACUUM_SLAB) == 0);
    // Independent of density.
    REQUIRE((classify_warnings(cp, 0.62, 8.3) & WARN_VACUUM_SLAB) != 0);
    REQUIRE((classify_warnings(cp, 0.62, 8.3) & WARN_DENSITY) == 0);
}

TEST_CASE("Degenerate input is handled rather than indexed", "[vacuum]") {
    const double cp[6] = {10, 10, 10, 90, 90, 90};
    const double one[3] = {0.5, 0.5, 0.5};
    REQUIRE(max_vacuum_gap(cp, one, 1).gap == 0.0);      // a single atom has no gap
    REQUIRE(max_vacuum_gap(cp, nullptr, 8).gap == 0.0);
    const double flat[6] = {10, 10, 10, 0, 0, 0};        // collapsed cell
    const auto f = slab_along_c(8, 1.0);
    REQUIRE(max_vacuum_gap(flat, f.data(), 8).gap == 0.0);
}
