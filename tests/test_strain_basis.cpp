// Applying any allowed strain must preserve the lattice type's metric
// constraints; otherwise S_cart is not orthogonal and symmetry copies shear.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/batch.hpp"
#include "klasp/cell.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/types.hpp"

#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;

namespace {

// Cell columns are the lattice vectors; returns |a|, |b|, |c| and the angles.
struct Metric { double a, b, c, alpha, beta, gamma; };

Metric metric_of(const double cell[3][3]) {
    auto col = [&](int j, double v[3]) { for (int i = 0; i < 3; ++i) v[i] = cell[i][j]; };
    auto nrm = [](const double v[3]) { return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); };
    auto ang = [&](const double u[3], const double v[3]) {
        const double d = u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
        return std::acos(d / (nrm(u)*nrm(v))) * 180.0 / M_PI;
    };
    double A[3], B[3], C[3];
    col(0, A); col(1, B); col(2, C);
    return {nrm(A), nrm(B), nrm(C), ang(B, C), ang(A, C), ang(A, B)};
}

// cell' = (I + eps) cell, with the Voigt convention of batch_shared.cpp.
void apply_strain(const double cell[3][3], const double e[6], double out[3][3]) {
    double eps[3][3] = {
        {e[0],       0.5*e[5],   0.5*e[4]},
        {0.5*e[5],   e[1],       0.5*e[3]},
        {0.5*e[4],   0.5*e[3],   e[2]    }};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double v = 0;
            for (int k = 0; k < 3; ++k)
                v += ((r == k ? 1.0 : 0.0) + eps[r][k]) * cell[k][c];
            out[r][c] = v;
        }
}

void cell_from_params(double a, double b, double c,
                      double al, double be, double ga, double out[3][3]) {
    RealType cp[6] = {RealType(a), RealType(b), RealType(c),
                      RealType(al), RealType(be), RealType(ga)};
    RealType m[3][3];
    cell_params_to_matrix(cp, m);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i][j] = static_cast<double>(m[i][j]);
}

// Apply each basis vector at a few magnitudes and report the worst violation of
// the given metric constraint.
template <typename Check>
double worst_violation(int lattice, const double cell[3][3], Check check) {
    RealType basis[36], cell_rt[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cell_rt[i*3 + j] = static_cast<RealType>(cell[i][j]);
    const int ndof = set_strain_basis_cell(lattice, cell_rt, basis, /*fix_angles=*/false);
    REQUIRE(ndof > 0);
    double worst = 0;
    for (int s = 0; s < ndof; ++s)
        for (double mag : {-0.05, -0.01, 0.01, 0.05}) {
            double e[6];
            for (int v = 0; v < 6; ++v) e[v] = mag * static_cast<double>(basis[s*6 + v]);
            double strained[3][3];
            apply_strain(cell, e, strained);
            worst = std::max(worst, check(metric_of(strained)));
        }
    return worst;
}

}  // namespace

TEST_CASE("tetragonal strain basis keeps a == b", "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(8.0, 8.0, 11.0, 90, 90, 90, cell);
    const double v = worst_violation(static_cast<int>(TETRAGONAL), cell,
        [](const Metric& m) {
            return std::max({std::abs(m.a - m.b),
                             std::abs(m.alpha - 90.0), std::abs(m.beta - 90.0),
                             std::abs(m.gamma - 90.0)});
        });
    REQUIRE(v < 1e-12);
}

TEST_CASE("hexagonal strain basis keeps a == b and gamma == 120", "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(7.0, 7.0, 9.5, 90, 90, 120, cell);
    const double v = worst_violation(static_cast<int>(HEXAGONAL), cell,
        [](const Metric& m) {
            return std::max({std::abs(m.a - m.b),
                             std::abs(m.alpha - 90.0), std::abs(m.beta - 90.0),
                             std::abs(m.gamma - 120.0)});
        });
    REQUIRE(v < 1e-12);
}

TEST_CASE("cubic strain basis keeps a == b == c and all angles 90", "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(9.0, 9.0, 9.0, 90, 90, 90, cell);
    const double v = worst_violation(static_cast<int>(CUBIC), cell,
        [](const Metric& m) {
            return std::max({std::abs(m.a - m.b), std::abs(m.b - m.c),
                             std::abs(m.alpha - 90.0), std::abs(m.beta - 90.0),
                             std::abs(m.gamma - 90.0)});
        });
    REQUIRE(v < 1e-12);
}

// Rhombohedral: the 3-fold runs along a+b+c. Both DOFs must preserve a == b == c
// and alpha == beta == gamma.
TEST_CASE("rhombohedral strain basis keeps a == b == c and equal angles",
          "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(8.0, 8.0, 8.0, 72, 72, 72, cell);
    const double v = worst_violation(static_cast<int>(RHOMBOHEDRAL), cell,
        [](const Metric& m) {
            return std::max({std::abs(m.a - m.b), std::abs(m.b - m.c),
                             std::abs(m.alpha - m.beta), std::abs(m.beta - m.gamma)});
        });
    REQUIRE(v < 1e-10);
}

TEST_CASE("monoclinic strain basis keeps alpha == gamma == 90", "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(7.0, 9.0, 11.0, 90, 104, 90, cell);
    const double v = worst_violation(static_cast<int>(MONOCLINIC), cell,
        [](const Metric& m) {
            return std::max(std::abs(m.alpha - 90.0), std::abs(m.gamma - 90.0));
        });
    REQUIRE(v < 1e-12);
}

TEST_CASE("orthorhombic strain basis keeps all angles 90", "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(7.0, 9.0, 11.0, 90, 90, 90, cell);
    const double v = worst_violation(static_cast<int>(ORTHORHOMBIC), cell,
        [](const Metric& m) {
            return std::max({std::abs(m.alpha - 90.0), std::abs(m.beta - 90.0),
                             std::abs(m.gamma - 90.0)});
        });
    REQUIRE(v < 1e-12);
}

// On a tetragonal cell, eps_xx alone breaks a == b and must not be expressible.
TEST_CASE("an uncoupled diagonal strain breaks the tetragonal metric",
          "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(8.0, 8.0, 11.0, 90, 90, 90, cell);
    double e[6] = {0.01, 0, 0, 0, 0, 0}, strained[3][3];
    apply_strain(cell, e, strained);
    const Metric m = metric_of(strained);
    REQUIRE(std::abs(m.a - m.b) > 1e-3);

    // Projecting onto the allowed directions keeps only the symmetric part.
    RealType basis[36];
    const int ndof = set_strain_basis(static_cast<int>(TETRAGONAL), basis);
    REQUIRE(ndof == 2);
    RealType eps[6] = {0.01, 0, 0, 0, 0, 0}, q[6], back[6];
    basis_project(basis, eps, ndof, q);
    basis_expand(basis, q, ndof, back);
    REQUIRE_THAT(double(back[0]), WithinAbs(0.005, 1e-12));
    REQUIRE_THAT(double(back[1]), WithinAbs(0.005, 1e-12));
}

// The amplitude <-> Voigt maps are inverse only because the basis rows are
// orthonormal.
TEST_CASE("strain basis rows are orthonormal in every lattice type",
          "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(8.0, 8.0, 8.0, 72, 72, 72, cell);
    RealType cell_rt[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cell_rt[i*3 + j] = static_cast<RealType>(cell[i][j]);

    for (int lt = 0; lt <= 6; ++lt) {
        RealType basis[36];
        const int ndof = set_strain_basis_cell(lt, cell_rt, basis);
        REQUIRE(ndof > 0);
        for (int s = 0; s < ndof; ++s)
            for (int t = 0; t < ndof; ++t) {
                double d = 0;
                for (int v = 0; v < 6; ++v) d += basis[s*6+v] * basis[t*6+v];
                REQUIRE_THAT(d, WithinAbs(s == t ? 1.0 : 0.0, 1e-12));
            }
        RealType q[6] = {0.01, -0.02, 0.03, -0.01, 0.02, 0.005}, eps[6], q2[6];
        basis_expand(basis, q, ndof, eps);
        basis_project(basis, eps, ndof, q2);
        for (int s = 0; s < ndof; ++s)
            REQUIRE_THAT(double(q2[s]), WithinAbs(double(q[s]), 1e-12));
    }
}

// The trust region scales the whole step by this fraction rather than clamping
// components, so the step stays a straight line in DOF space.
TEST_CASE("strain step fraction stops exactly at the bound", "[strain][optimiser]") {
    RealType basis[36];
    const int ndof = set_strain_basis(static_cast<int>(ORTHORHOMBIC), basis);
    REQUIRE(ndof == 3);

    SECTION("a step that stays inside is not scaled") {
        RealType q0[3] = {0, 0, 0}, dq[3] = {0.1, -0.05, 0.02};
        REQUIRE_THAT(double(strain_step_fraction(basis, q0, dq, ndof)), WithinAbs(1.0, 1e-12));
    }
    SECTION("a step that overshoots lands on the bound") {
        RealType q0[3] = {0.2, 0, 0}, dq[3] = {0.4, 0, 0};   // would reach 0.6
        const RealType t = strain_step_fraction(basis, q0, dq, ndof);
        REQUIRE_THAT(double(t), WithinAbs(0.5, 1e-12));       // 0.2 + 0.5*0.4 = 0.4
        RealType q1[3], eps[6];
        for (int s = 0; s < ndof; ++s) q1[s] = q0[s] + t * dq[s];
        basis_expand(basis, q1, ndof, eps);
        REQUIRE_THAT(double(eps[0]), WithinAbs(double(STRAIN_MAX_DIAG), 1e-12));
    }
    SECTION("shear has the tighter bound") {
        RealType tb[36];
        const int nt = set_strain_basis(static_cast<int>(TRICLINIC), tb);
        REQUIRE(nt == 6);
        RealType q0[6] = {0,0,0,0,0,0}, dq[6] = {0,0,0,0,0,0.6};
        const RealType t = strain_step_fraction(tb, q0, dq, nt);
        REQUIRE_THAT(double(t), WithinAbs(double(STRAIN_MAX_SHEAR) / 0.6, 1e-12));
    }
}
