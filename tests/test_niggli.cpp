#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/niggli.hpp"
#include "klasp/cell.hpp"
#include "klasp/warnings.hpp"
#include "klasp/io.hpp"

#include <cmath>
#include <random>

using namespace klasp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

double det3(const RealType m[3][3]) {
    return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
         - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
         + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
}

int det3i(const int p[3][3]) {
    return p[0][0]*(p[1][1]*p[2][2] - p[1][2]*p[2][1])
         - p[0][1]*(p[1][0]*p[2][2] - p[1][2]*p[2][0])
         + p[0][2]*(p[1][0]*p[2][1] - p[1][1]*p[2][0]);
}

// Shortest interplanar spacing, which sets how many periodic images a cutoff needs.
double min_spacing(const RealType m[3][3]) {
    const double vol = std::abs(det3(m));
    double worst = 1e300;
    for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        const double cx = m[1][j]*m[2][k] - m[2][j]*m[1][k];
        const double cy = m[2][j]*m[0][k] - m[0][j]*m[2][k];
        const double cz = m[0][j]*m[1][k] - m[1][j]*m[0][k];
        worst = std::min(worst, vol / std::sqrt(cx*cx + cy*cy + cz*cz));
    }
    return worst;
}

double shape_factor(const RealType m[3][3]) {
    const double vol = std::abs(det3(m));
    double prod = 1.0;
    for (int j = 0; j < 3; ++j)
        prod *= std::sqrt(m[0][j]*m[0][j] + m[1][j]*m[1][j] + m[2][j]*m[2][j]);
    return vol / prod;
}

}  // namespace

// Seed 12830, SG 2: a cell whose spacings were too small for the run's image count.
TEST_CASE("The collapsed P-1 cell reduces to a usable basis", "[niggli]") {
    const RealType params[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    RealType m[3][3];
    cell_params_to_matrix(params, m);

    const NiggliResult r = niggli_reduce(m);
    REQUIRE(r.converged);

    INFO("shape " << shape_factor(m) << " -> " << shape_factor(r.cell));
    INFO("min spacing " << min_spacing(m) << " -> " << min_spacing(r.cell));
    REQUIRE(shape_factor(r.cell) > shape_factor(m));
    REQUIRE(min_spacing(r.cell) > min_spacing(m));
    // Good enough that a 15 A (28.35 Bohr) cutoff needs a modest image count.
    REQUIRE(28.35 / min_spacing(r.cell) < 10.0);
}

TEST_CASE("Reduction preserves the lattice", "[niggli]") {
    const RealType params[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    RealType m[3][3];
    cell_params_to_matrix(params, m);
    const NiggliResult r = niggli_reduce(m);

    REQUIRE(std::abs(det3i(r.transform)) == 1);          // unimodular
    REQUIRE_THAT(std::abs(det3(r.cell)),
                 WithinRel(std::abs(det3(m)), 1e-10));   // same volume
    REQUIRE(det3(r.cell) > 0);                           // right-handed

    // reduced == original * P, to floating point.
    for (int rr = 0; rr < 3; ++rr)
        for (int c = 0; c < 3; ++c) {
            double acc = 0;
            for (int k = 0; k < 3; ++k) acc += m[rr][k] * r.transform[k][c];
            REQUIRE_THAT(double(r.cell[rr][c]), WithinAbs(acc, 1e-9));
        }
}

// All angles acute and edges ordered, so every Niggli condition holds on entry.
// (Mixed acute and obtuse angles would not be reduced.)
TEST_CASE("An already-reduced cell is left alone", "[niggli]") {
    const RealType params[6] = {10.0, 11.0, 12.0, 88.0, 89.0, 87.0};
    RealType m[3][3];
    cell_params_to_matrix(params, m);
    const NiggliResult r = niggli_reduce(m);
    REQUIRE(r.converged);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(r.transform[i][j] == (i == j ? 1 : 0));
}

TEST_CASE("Reduction is idempotent", "[niggli]") {
    const RealType params[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    RealType m[3][3];
    cell_params_to_matrix(params, m);
    const NiggliResult a = niggli_reduce(m);
    const NiggliResult b = niggli_reduce(a.cell);
    REQUIRE(b.converged);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE_THAT(double(b.cell[i][j]), WithinAbs(double(a.cell[i][j]), 1e-9));
}

// A fractional coordinate mapped through the transform lands on the same point.
TEST_CASE("Fractional coordinates follow the transform", "[niggli]") {
    const RealType params[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    RealType m[3][3];
    cell_params_to_matrix(params, m);
    const NiggliResult r = niggli_reduce(m);

    const double frac[3] = {0.31, -0.72, 0.15};
    double frac2[3];
    niggli_map_fractional(r.transform, frac, frac2);

    for (int d = 0; d < 3; ++d) {
        const double before = m[d][0]*frac[0]  + m[d][1]*frac[1]  + m[d][2]*frac[2];
        const double after  = r.cell[d][0]*frac2[0] + r.cell[d][1]*frac2[1] + r.cell[d][2]*frac2[2];
        REQUIRE_THAT(after, WithinAbs(before, 1e-8));
    }
}

// Random oblique cells, to find cycles between bases the comparisons cannot
// tell apart.
TEST_CASE("Reduction converges and improves conditioning on random cells", "[niggli]") {
    std::mt19937 rng(12830);
    std::uniform_real_distribution<double> len(5.0, 60.0);
    std::uniform_real_distribution<double> ang(20.0, 160.0);

    int tried = 0, degenerate_before = 0;
    for (int t = 0; t < 2000; ++t) {
        RealType p[6] = {RealType(len(rng)), RealType(len(rng)), RealType(len(rng)),
                         RealType(ang(rng)), RealType(ang(rng)), RealType(ang(rng))};
        RealType m[3][3];
        cell_params_to_matrix(p, m);
        if (!std::isfinite(double(m[2][2])) || std::abs(det3(m)) < 1e-6) continue;  // not a real cell
        ++tried;
        if (shape_factor(m) < 0.15) ++degenerate_before;

        const NiggliResult r = niggli_reduce(m);
        REQUIRE(r.converged);
        REQUIRE(std::abs(det3i(r.transform)) == 1);
        REQUIRE_THAT(std::abs(det3(r.cell)), WithinRel(std::abs(det3(m)), 1e-8));
        // Minimum spacing is not monotonic under reduction, so assert conditioning.
        REQUIRE(shape_factor(r.cell) >= 0.15);
    }
    INFO("cells " << tried << ", degenerate before reduction " << degenerate_before);
    REQUIRE(tried > 500);
    REQUIRE(degenerate_before > 0);   // the sample has to contain the hard case
}

// Only P1 and P-1 have operations that hold in any basis.
TEST_CASE("Only P1 and P-1 may be reduced", "[niggli]") {
    REQUIRE(niggli_applicable(1));
    REQUIRE(niggli_applicable(2));
    for (int sg : {4, 5, 9, 14, 15, 19, 29, 33, 61, 227})
        REQUIRE_FALSE(niggli_applicable(sg));
}
