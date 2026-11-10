#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/warnings.hpp"

using namespace klasp;
using Catch::Matchers::WithinAbs;

TEST_CASE("A well-shaped cell carries no warnings", "[warnings]") {
    const double ortho[6] = {10.0, 12.0, 14.0, 90.0, 90.0, 90.0};
    REQUIRE_THAT(cell_shape_factor(ortho), WithinAbs(1.0, 1e-12));
    REQUIRE(classify_warnings(ortho, 1.20) == WARN_NONE);

    const double mono[6] = {8.0, 11.0, 13.0, 90.0, 105.0, 90.0};
    REQUIRE(classify_warnings(mono, 1.35) == WARN_NONE);
}

// A real P-1 cell overbound by ~35 kJ/mol; its density is normal, only its shape
// gives it away.
TEST_CASE("The collapsed P-1 cell is flagged degenerate", "[warnings]") {
    const double bad[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    const int w = classify_warnings(bad, 1.1654);
    REQUIRE((w & WARN_DEGENERATE_CELL) != 0);
    REQUIRE((w & WARN_DENSITY) == 0);
}

TEST_CASE("Warning codes compose", "[warnings]") {
    // A real cell: 0.10 g/cm^3 across a 200 Bohr axis, and collapsed.
    const double gas[6] = {7.6, 206.9, 218.1, 177.70, 96.38, 82.96};
    REQUIRE(classify_warnings(gas, 0.1024) == (WARN_DEGENERATE_CELL | WARN_DENSITY));
}

TEST_CASE("An unset density is not itself a warning", "[warnings]") {
    const double ortho[6] = {10.0, 12.0, 14.0, 90.0, 90.0, 90.0};
    REQUIRE(classify_warnings(ortho, 0.0) == WARN_NONE);
    REQUIRE(classify_warnings(ortho, -1.0) == WARN_NONE);
}

TEST_CASE("The shape factor tracks how oblique a cell is", "[warnings]") {
    // 30 degrees is still 0.22.
    const double cube[6]   = {10.0, 10.0, 10.0, 90.0, 90.0, 90.0};
    const double tilted[6] = {10.0, 10.0, 10.0, 80.0, 80.0, 80.0};
    const double flat[6]   = {10.0, 10.0, 10.0, 20.0, 20.0, 20.0};
    REQUIRE(cell_shape_factor(cube) > cell_shape_factor(tilted));
    REQUIRE(cell_shape_factor(tilted) > cell_shape_factor(flat));
    REQUIRE(classify_warnings(cube, 1.2) == WARN_NONE);
    REQUIRE((classify_warnings(flat, 1.2) & WARN_DEGENERATE_CELL) != 0);
}

// --- the vacuum threshold a run actually uses -------------------------------

TEST_CASE("the vacuum threshold follows the potential cutoff", "[warnings]") {
    // Negative means derive it: half the cutoff, since slabs interact until the
    // gap reaches the cutoff.
    REQUIRE_THAT(resolve_max_vacuum(-1.0, 15.0), WithinAbs(7.5, 1e-12));
    REQUIRE_THAT(resolve_max_vacuum(-1.0, 8.0), WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(resolve_max_vacuum(-1.0, 30.0), WithinAbs(15.0, 1e-12));

    // The default sits between real packings (gaps up to about 2.1 A) and the
    // catastrophes (10 A and up).
    REQUIRE(resolve_max_vacuum(-1.0, 15.0) > WARN_VACUUM_GAP_A);
    REQUIRE(resolve_max_vacuum(-1.0, 15.0) < 10.0);
}

TEST_CASE("an explicit vacuum threshold is honoured", "[warnings]") {
    REQUIRE_THAT(resolve_max_vacuum(3.0, 15.0), WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(resolve_max_vacuum(6.0, 8.0), WithinAbs(6.0, 1e-12));
    // Zero is a real setting (keep every structure, only flag), not "unset".
    REQUIRE_THAT(resolve_max_vacuum(0.0, 15.0), WithinAbs(0.0, 1e-12));
}
