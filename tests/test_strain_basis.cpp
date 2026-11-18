// Applying any allowed strain must preserve the lattice type's metric
// constraints; otherwise S_cart is not orthogonal and symmetry copies shear.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <catch2/matchers/catch_matchers_string.hpp>

#include "klasp/batch.hpp"
#include "klasp/cell.hpp"
#include "klasp/log.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/strain_basis.hpp"
#include "klasp/types.hpp"
#include "klasp/wyckoff.hpp"
#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstring>
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

std::vector<RealType> rotations_of(const std::vector<SymopData>& ops) {
    std::vector<RealType> w;
    for (const auto& op : ops)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) w.push_back(static_cast<RealType>(op.rot[r][c]));
    return w;
}

PointGroup point_group_of_sg(const SpaceGroupInfo& sg) {
    std::vector<SymopData> ops(sg.symops, sg.symops + sg.n_symops);
    const auto w = rotations_of(ops);
    return point_group_of(w.data(), sg.n_symops);
}

void to_flat(const double cell[3][3], RealType out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i*3 + j] = static_cast<RealType>(cell[i][j]);
}

// The basis of space group `sg_number`'s operations in `cell`.
int sg_strain_basis(int sg_number, const double cell[3][3], RealType basis[36]) {
    RealType c[9];
    to_flat(cell, c);
    return symmetric_strain_basis(point_group_of_sg(get_space_group(sg_number)), c, false, basis);
}

// Apply each basis vector at a few magnitudes and report the worst violation of
// the given metric constraint.
template <typename Check>
double worst_violation(int sg_number, const double cell[3][3], Check check) {
    RealType basis[36];
    const int ndof = sg_strain_basis(sg_number, cell, basis);
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
    const double v = worst_violation(88, cell,
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
    const double v = worst_violation(176, cell,
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
    const double v = worst_violation(205, cell,
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
    const double v = worst_violation(148, cell,
        [](const Metric& m) {
            return std::max({std::abs(m.a - m.b), std::abs(m.b - m.c),
                             std::abs(m.alpha - m.beta), std::abs(m.beta - m.gamma)});
        });
    REQUIRE(v < 1e-10);
}

TEST_CASE("monoclinic strain basis keeps alpha == gamma == 90", "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(7.0, 9.0, 11.0, 90, 104, 90, cell);
    const double v = worst_violation(14, cell,
        [](const Metric& m) {
            return std::max(std::abs(m.alpha - 90.0), std::abs(m.gamma - 90.0));
        });
    REQUIRE(v < 1e-12);
}

TEST_CASE("orthorhombic strain basis keeps all angles 90", "[strain][symmetry]") {
    double cell[3][3];
    cell_from_params(7.0, 9.0, 11.0, 90, 90, 90, cell);
    const double v = worst_violation(61, cell,
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
    const int ndof = sg_strain_basis(88, cell, basis);
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
    for (int sg : {2, 14, 61, 88, 148, 176, 205}) {
        CAPTURE(sg);
        double cell[3][3];
        canonical_cell(get_space_group(sg).lattice, cell);
        RealType basis[36];
        const int ndof = sg_strain_basis(sg, cell, basis);
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
    const int ndof = testing::lattice_strain_basis(ORTHORHOMBIC, nullptr, basis, 6);
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
        const int nt = testing::lattice_strain_basis(TRICLINIC, nullptr, tb, 6);
        REQUIRE(nt == 6);
        RealType q0[6] = {0,0,0,0,0,0}, dq[6] = {0,0,0,0,0,0.6};
        const RealType t = strain_step_fraction(tb, q0, dq, nt);
        REQUIRE_THAT(double(t), WithinAbs(double(STRAIN_MAX_SHEAR) / 0.6, 1e-12));
    }
}

// ---------------------------------------------------------------------------
// The basis derived from the symmetry operations
// ---------------------------------------------------------------------------

namespace {

// Cartesian rotations R = C W C^-1 of a point group in a cell.
std::vector<std::array<double, 9>> cartesian_rotations(const PointGroup& pg,
                                                       const double cell[3][3]) {
    double inv[3][3] = {};
    REQUIRE(mat3_inv(cell, inv));
    std::vector<std::array<double, 9>> out;
    for (const auto& w : pg.rotations) {
        std::array<double, 9> R{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        R[i*3 + j] += cell[i][k] * w[k*3 + l] * inv[l][j];
        out.push_back(R);
    }
    return out;
}

// Largest |R eps R^T - eps| over the group, eps from a Voigt row.
double worst_invariance(const std::vector<std::array<double, 9>>& Rs, const RealType* row) {
    const double e[3][3] = {
        {double(row[0]),       0.5*double(row[5]), 0.5*double(row[4])},
        {0.5*double(row[5]),   double(row[1]),     0.5*double(row[3])},
        {0.5*double(row[4]),   0.5*double(row[3]), double(row[2])    }};
    double worst = 0;
    for (const auto& R : Rs)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double t = 0;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l) t += R[i*3 + k] * e[k][l] * R[j*3 + l];
                worst = std::max(worst, std::abs(t - e[i][j]));
            }
    return worst;
}

// Largest |W^T G W - G| / max|G| over the group, G the metric of `cell`.
double worst_metric_violation(const PointGroup& pg, const double cell[3][3]) {
    double G[3][3] = {}, gmax = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) G[i][j] += cell[k][i] * cell[k][j];
            gmax = std::max(gmax, std::abs(G[i][j]));
        }
    double worst = 0;
    for (const auto& w : pg.rotations)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double t = 0;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l) t += w[k*3 + i] * G[k][l] * w[l*3 + j];
                worst = std::max(worst, std::abs(t - G[i][j]) / gmax);
            }
    return worst;
}

// A cell whose metric the group keeps: a generic metric averaged over the group.
void symmetric_cell(const PointGroup& pg, double cell[3][3]) {
    double c0[3][3];
    cell_from_params(1.0, 1.19, 1.37, 86, 103, 94, c0);
    double G0[3][3] = {}, G[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) G0[i][j] += c0[k][i] * c0[k][j];
    const double order = static_cast<double>(pg.rotations.size());
    for (const auto& w : pg.rotations)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        G[i][j] += w[k*3 + i] * G0[k][l] * w[l*3 + j] / order;
    const double a = std::sqrt(G[0][0]), b = std::sqrt(G[1][1]), c = std::sqrt(G[2][2]);
    const double deg = 180.0 / M_PI;
    cell_from_params(a, b, c, std::acos(G[1][2] / (b*c)) * deg,
                     std::acos(G[0][2] / (a*c)) * deg, std::acos(G[0][1] / (a*b)) * deg, cell);
}

bool rows_orthonormal(const RealType* basis, int n, double tol) {
    for (int s = 0; s < n; ++s)
        for (int t = 0; t < n; ++t) {
            double d = 0;
            for (int v = 0; v < 6; ++v) d += double(basis[s*6 + v]) * double(basis[t*6 + v]);
            if (std::abs(d - (s == t ? 1.0 : 0.0)) > tol) return false;
        }
    return true;
}

}  // namespace

// The rows must be the lattice-type rows exactly, so landscapes do not move;
// rhombohedral rows depend on the cell and match to round-off.
TEST_CASE("the operations give the lattice-type basis in every standard setting",
          "[strain][symmetry]") {
    std::vector<int> differing;
    auto compare = [&](int number, LatticeType lt, const PointGroup& pg,
                       const double cell[3][3]) {
        RealType c[9];
        to_flat(cell, c);
        for (bool fix : {false, true}) {
            RealType ref[36], got[36];
            const int n_ref = testing::lattice_strain_basis(lt, c, ref,
                                                            testing::lattice_max_dof(lt, fix));
            const int n_got = symmetric_strain_basis(pg, c, fix, got);
            bool same = n_got == n_ref;
            if (same && lt != RHOMBOHEDRAL) {
                same = std::memcmp(ref, got, sizeof(ref)) == 0;
            } else if (same) {
                for (int s = 0; s < n_ref; ++s) {
                    double dp = 0, dm = 0;
                    for (int v = 0; v < 6; ++v) {
                        dp = std::max(dp, std::abs(double(got[s*6+v]) - double(ref[s*6+v])));
                        dm = std::max(dm, std::abs(double(got[s*6+v]) + double(ref[s*6+v])));
                    }
                    same = same && std::min(dp, dm) < 1e-12;
                }
            }
            if (!same) differing.push_back(fix ? -number : number);
        }
    };

    int n_rhombohedral = 0;
    for (int n = 1; n <= 230; ++n) {
        const auto& sg = get_space_group(n);
        const PointGroup pg = point_group_of_sg(sg);
        double cell[3][3];
        canonical_cell(sg.lattice, cell);
        compare(n, sg.lattice, pg, cell);
        if (sg.lattice == RHOMBOHEDRAL) {
            ++n_rhombohedral;
            cell_from_params(1.0, 1.0, 1.0, 101, 101, 101, cell);
            compare(n, sg.lattice, pg, cell);
        }
    }
    CHECK(n_rhombohedral == 7);
    // Negative numbers are the lengths-only (fix_angles) comparison.
    CAPTURE(differing);
    REQUIRE(differing.empty());
}

TEST_CASE("the operations give a strain basis in every gemmi setting", "[strain][symmetry]") {
    const auto settings = all_space_group_settings();
    REQUIRE(settings.size() > 500);
    for (const auto& st : settings) {
        CAPTURE(st.number, st.name);
        const auto w = rotations_of(st.ops);
        const PointGroup pg = point_group_of(w.data(), static_cast<int>(st.ops.size()));
        double cell[3][3];
        symmetric_cell(pg, cell);
        REQUIRE(worst_metric_violation(pg, cell) < 1e-12);
        const auto Rs = cartesian_rotations(pg, cell);
        RealType c[9];
        to_flat(cell, c);

        RealType basis[36], diag[36];
        const int n = symmetric_strain_basis(pg, c, false, basis);
        const int nd = symmetric_strain_basis(pg, c, true, diag);
        REQUIRE(n == pg.strain_dim);
        REQUIRE(nd >= 1);
        REQUIRE(nd <= n);
        REQUIRE(rows_orthonormal(basis, n, 1e-12));
        for (int s = 0; s < n; ++s) {
            CAPTURE(s);
            CHECK(worst_invariance(Rs, basis + s*6) < 1e-10);
            double e[6], strained[3][3];
            for (int v = 0; v < 6; ++v) e[v] = 0.03 * double(basis[s*6 + v]);
            apply_strain(cell, e, strained);
            CHECK(worst_metric_violation(pg, strained) < 1e-10);
        }
        for (int s = 0; s < nd; ++s)
            for (int v = 3; v < 6; ++v) CHECK(diag[s*6 + v] == RealType(0));
        // The lengths-only rows lead the full basis.
        CHECK(std::memcmp(diag, basis, sizeof(RealType) * 6 * nd) == 0);
    }
}

TEST_CASE("operations that are not closed are closed into their point group",
          "[strain][symmetry]") {
    SECTION("a 4-fold alone generates 4") {
        const RealType w[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
        const PointGroup pg = point_group_of(w, 1);
        REQUIRE(pg.rotations.size() == 4);
        REQUIRE(pg.strain_dim == 2);
    }
    SECTION("a 2-fold and the inversion generate 2/m") {
        const RealType w[18] = {-1, 0, 0, 0, 1, 0, 0, 0, -1,
                                -1, 0, 0, 0, -1, 0, 0, 0, -1};
        const PointGroup pg = point_group_of(w, 2);
        REQUIRE(pg.rotations.size() == 4);
        REQUIRE(pg.strain_dim == 4);
    }
    SECTION("repeated operations count once") {
        const RealType w[18] = {1, 0, 0, 0, 1, 0, 0, 0, 1,
                                1, 0, 0, 0, 1, 0, 0, 0, 1};
        const PointGroup pg = point_group_of(w, 2);
        REQUIRE(pg.rotations.size() == 1);
        REQUIRE(pg.strain_dim == 6);
    }
    SECTION("a non-integer rotation is refused") {
        const RealType w[9] = {0.5, 0, 0, 0, 1, 0, 0, 0, 1};
        REQUIRE_THROWS(point_group_of(w, 1));
    }
}

TEST_CASE("a cell whose metric does not fit the operations warns", "[strain][symmetry]") {
    const PointGroup pg = point_group_of_sg(get_space_group(75));   // P4
    double cell[3][3];
    cell_from_params(8.0, 8.02, 11.0, 90, 90, 90, cell);
    RealType c[9], basis[36];
    to_flat(cell, c);
    logging::Capture capture(logging::Level::Warn);
    const int n = symmetric_strain_basis(pg, c, false, basis);
    CHECK(n == 2);
    CHECK_THAT(capture.text(), Catch::Matchers::ContainsSubstring("does not fit"));
}

TEST_CASE("a cell that fits the operations does not warn", "[strain][symmetry]") {
    const PointGroup pg = point_group_of_sg(get_space_group(75));
    double cell[3][3];
    cell_from_params(8.0, 8.0, 11.0, 90, 90, 90, cell);
    RealType c[9], basis[36];
    to_flat(cell, c);
    logging::Capture capture(logging::Level::Warn);
    CHECK(symmetric_strain_basis(pg, c, false, basis) == 2);
    CHECK(capture.text().empty());
}
