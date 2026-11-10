// Rigid-body gradient vs finite difference under non-trivial symmetry operators.
// P1 and P-1 only have +I and -I, which cannot distinguish a correct rotation
// from a transposed one; a 3-fold on hexagonal axes has off-diagonal Cartesian
// entries of +-sqrt(3)/2.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/multipole.hpp"
#include "klasp/strain_basis.hpp"
#include "klasp/sgdata.hpp"
#include "test_support.hpp"

#include <cstdio>
#include <cmath>
#include <array>
#include <vector>

using namespace klasp;

namespace {

using klasp::testing::Fixture;

// Fixture with a caller-supplied set of fractional symmetry operators, so the
// same molecule and cell can be run under a 3-fold and under a control operator.
Fixture make_sym_fixture(double a_A, double b_A, double c_A,
                         double alpha, double beta, double gamma,
                         const std::vector<std::array<int, 9>>& rots) {
    Fixture f;
    const int n_indep = 1, n_sites = 2;
    const int n_symops = static_cast<int>(rots.size());
    // Allocated to rank 4 so the fixture runs at every instantiated rank.
    f.mol = allocate_molecule_data(n_indep, n_sites, n_symops, /*max_multipole_rank=*/4);
    f.ws  = allocate_workspace(64, 30000, 50000, n_indep * n_sites, /*max_multipole_rank=*/4);

    f.mol.cell_params[0] = a_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[1] = b_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[2] = c_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[3] = alpha;
    f.mol.cell_params[4] = beta;
    f.mol.cell_params[5] = gamma;

    auto h_bp = Kokkos::create_mirror_view(f.mol.body_positions);
    auto h_bc = Kokkos::create_mirror_view(f.mol.body_charges);
    auto h_bt_types = Kokkos::create_mirror_view(f.mol.body_types);
    h_bp(0, 0) = -0.6 * ANGSTROM_TO_BOHR; h_bp(0, 1) = 0.0; h_bp(0, 2) = 0.0;
    h_bp(1, 0) =  0.6 * ANGSTROM_TO_BOHR; h_bp(1, 1) = 0.1 * ANGSTROM_TO_BOHR; h_bp(1, 2) = 0.0;
    h_bc(0) =  0.35; h_bc(1) = -0.35;
    h_bt_types(0) = 0; h_bt_types(1) = 0;
    Kokkos::deep_copy(f.mol.body_positions, h_bp);
    Kokkos::deep_copy(f.mol.body_charges, h_bc);
    Kokkos::deep_copy(f.mol.body_types, h_bt_types);

    // Multipoles to rank 4, every component non-zero and off every cell axis,
    // decaying with rank.
    auto h_mp = Kokkos::create_mirror_view(f.mol.body_multipoles);
    Kokkos::deep_copy(h_mp, RealType(0));
    const int n_mp = static_cast<int>(f.mol.body_multipoles.extent(1));
    for (int c = 0; c < n_mp; ++c) {
        const double s = std::sin(1.7 * (c + 1)) * std::exp(-0.15 * c);
        h_mp(0, c) = static_cast<RealType>( 0.35 * s);
        h_mp(1, c) = static_cast<RealType>(-0.29 * std::cos(2.3 * (c + 1)) * std::exp(-0.15 * c));
    }
    h_mp(0, 0) =  0.35; h_mp(1, 0) = -0.35;
    Kokkos::deep_copy(f.mol.body_multipoles, h_mp);

    auto h_sr = Kokkos::create_mirror_view(f.mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(f.mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(f.mol.sym_indep_index);
    for (int op = 0; op < n_symops; ++op) {
        for (int c = 0; c < 9; ++c) h_sr(op, c) = rots[op][c];
        for (int d = 0; d < 3; ++d) h_st(op, d) = 0;
        h_si(op) = 0;
    }
    Kokkos::deep_copy(f.mol.sym_rotations, h_sr);
    Kokkos::deep_copy(f.mol.sym_translations, h_st);
    Kokkos::deep_copy(f.mol.sym_indep_index, h_si);

    f.mol.n_indep = n_indep;
    f.mol.n_symops = n_symops;
    f.mol.n_sites_per_mol = n_sites;
    f.ws.n_types = 1;
    f.ws.buck_cutoff = 8.0 * ANGSTROM_TO_BOHR;

    f.bws = allocate_batch_workspace(1, f.ws, f.mol);

    {
        auto hA = Kokkos::create_mirror_view(f.bws.tables.buck_A);
        auto hB = Kokkos::create_mirror_view(f.bws.tables.buck_B);
        auto hC = Kokkos::create_mirror_view(f.bws.tables.buck_C);
        auto hR = Kokkos::create_mirror_view(f.bws.tables.buck_Rmin);
        hA(0,0) = 60.0; hB(0,0) = 1.9; hC(0,0) = 40.0; hR(0,0) = 0;
        Kokkos::deep_copy(f.bws.tables.buck_A, hA);
        Kokkos::deep_copy(f.bws.tables.buck_B, hB);
        Kokkos::deep_copy(f.bws.tables.buck_C, hC);
        Kokkos::deep_copy(f.bws.tables.buck_Rmin, hR);
    }

    // Off-centre and off-axis, so no gradient component vanishes by symmetry.
    auto h_bt = Kokkos::create_mirror_view(f.bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(f.bws.dofs.rot_vectors);
    h_bt(0,0,0) = 0.11; h_bt(0,0,1) = 0.07; h_bt(0,0,2) = 0.13;
    h_br(0,0,0) = 0.21; h_br(0,0,1) = 0.14; h_br(0,0,2) = 0.09;
    Kokkos::deep_copy(f.bws.dofs.translations, h_bt);
    Kokkos::deep_copy(f.bws.dofs.rot_vectors, h_br);

    auto h_cp = Kokkos::create_mirror_view(f.bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = f.mol.cell_params[i];
    Kokkos::deep_copy(f.bws.geom.cell_params, h_cp);
    batch_setup_cells(f.bws.tables, f.bws.geom);

    f.n_dof = n_indep * 6;
    f.settings.n_iterations = 1;
    f.settings.max_step = 0.05;
    f.settings.lbfgs_memory = 5;
    f.settings.model.taper_on = 0;
    f.settings.model.taper_off = 0;
    f.settings.model.wolf = true;
    f.settings.energy_only = false;
    return f;
}

// Identity, 3+ (-y, x-y, z) and 3- (-x+y, -x, z): the P3 rotation set on
// hexagonal axes.
const std::vector<std::array<int, 9>> THREEFOLD = {
    {1, 0, 0,  0, 1, 0,  0, 0, 1},
    {0,-1, 0,  1,-1, 0,  0, 0, 1},
    {-1, 1, 0, -1, 0, 0,  0, 0, 1},
};

// Identity + 2-fold about b (a signed permutation in Cartesian): the control.
const std::vector<std::array<int, 9>> TWOFOLD = {
    {1, 0, 0,  0, 1, 0,  0, 0, 1},
    {-1, 0, 0,  0, 1, 0,  0, 0,-1},
};

// P-1: identity + inversion, compatible with any cell shape.
const std::vector<std::array<int, 9>> PBAR1 = {
    {1, 0, 0,  0, 1, 0,  0, 0, 1},
    {-1, 0, 0,  0,-1, 0,  0, 0,-1},
};

// P4 about c: identity, 4+ (-y, x, z), 2 (-x, -y, z), 4- (y, -x, z). Signed
// permutations that are not involutions (S^-1 != S).
const std::vector<std::array<int, 9>> FOURFOLD = {
    {1, 0, 0,  0, 1, 0,  0, 0, 1},
    {0,-1, 0,  1, 0, 0,  0, 0, 1},
    {-1, 0, 0,  0,-1, 0,  0, 0, 1},
    {0, 1, 0, -1, 0, 0,  0, 0, 1},
};

}  // namespace

TEST_CASE("rigid-body gradient matches FD under a 2-fold (charges only)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 8.2, 9.5, 90, 97, 90, TWOFOLD);
    std::printf("  monoclinic, 2-fold, rank 0:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<0>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

TEST_CASE("rigid-body gradient matches FD under a 2-fold (multipoles)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 8.2, 9.5, 90, 97, 90, TWOFOLD);
    std::printf("  monoclinic, 2-fold, rank 1:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<1>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

TEST_CASE("rigid-body gradient matches FD under a 3-fold (charges only)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 120, THREEFOLD);
    std::printf("  hexagonal, 3-fold, rank 0:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<0>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

TEST_CASE("rigid-body gradient matches FD under a 3-fold (multipoles)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 120, THREEFOLD);
    std::printf("  hexagonal, 3-fold, rank 1:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<1>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

TEST_CASE("rigid-body gradient matches FD under a 3-fold (multipoles, Ewald)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 120, THREEFOLD);
    f.settings.model.wolf = false;
    std::printf("  hexagonal, 3-fold, rank 1, Ewald reciprocal:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<1>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

// Rank 2 and rank 4, the ranks production runs evaluate.
TEST_CASE("rigid-body gradient matches FD under a 2-fold (rank 2)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 8.2, 9.5, 90, 97, 90, TWOFOLD);
    std::printf("  monoclinic, 2-fold, rank 2:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<2>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

TEST_CASE("rigid-body gradient matches FD under a 3-fold (rank 2)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 120, THREEFOLD);
    std::printf("  hexagonal, 3-fold, rank 2:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<2>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

TEST_CASE("rigid-body gradient matches FD under a 2-fold (rank 4)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 8.2, 9.5, 90, 97, 90, TWOFOLD);
    std::printf("  monoclinic, 2-fold, rank 4:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<4>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

TEST_CASE("rigid-body gradient matches FD under a 3-fold (rank 4)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 120, THREEFOLD);
    std::printf("  hexagonal, 3-fold, rank 4:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<4>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

// A 4-fold: a signed permutation but not an involution.
TEST_CASE("rigid-body gradient matches FD under a 4-fold (rank 4)",
          "[gradients][symmetry]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 90, FOURFOLD);
    std::printf("  tetragonal, 4-fold, rank 4:\n");
    REQUIRE(testing::worst_dof_gradient_mismatch<4>(f.bws, f.settings, 0, 1e-6) < 1e-5);
}

// ---------------------------------------------------------------------------
// rotate_cartesian_multipole_deriv must be the exact derivative of
// rotate_cartesian_multipole, along a direction dR, for a dense rotation and a
// signed permutation, at every rank.
// ---------------------------------------------------------------------------
namespace {

void axis_angle_mat(double ax, double ay, double az, double th, RealType R[3][3]) {
    const double n = std::sqrt(ax*ax + ay*ay + az*az);
    ax /= n; ay /= n; az /= n;
    const double c = std::cos(th), s = std::sin(th), t = 1 - c;
    R[0][0] = t*ax*ax + c;    R[0][1] = t*ax*ay - s*az; R[0][2] = t*ax*az + s*ay;
    R[1][0] = t*ax*ay + s*az; R[1][1] = t*ay*ay + c;    R[1][2] = t*ay*az - s*ax;
    R[2][0] = t*ax*az - s*ay; R[2][1] = t*ay*az + s*ax; R[2][2] = t*az*az + c;
}

double rot_deriv_mismatch(int rank, const RealType R0[3][3], const RealType dR[3][3]) {
    const int N = nhermsum(rank);
    RealType body[35];
    for (int c = 0; c < N; ++c)
        body[c] = static_cast<RealType>(std::sin(1.7*(c+1)) * std::exp(-0.15*c));

    RealType an[35];
    rotate_cartesian_multipole_deriv(body, R0, dR, rank, an);

    const double h = 1e-6;
    RealType Rp[3][3], Rm[3][3], lp[35], lm[35];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Rp[i][j] = R0[i][j] + static_cast<RealType>(h) * dR[i][j];
            Rm[i][j] = R0[i][j] - static_cast<RealType>(h) * dR[i][j];
        }
    rotate_cartesian_multipole(body, Rp, rank, lp);
    rotate_cartesian_multipole(body, Rm, rank, lm);

    double worst = 0.0;
    for (int c = 0; c < N; ++c) {
        const double fd = (lp[c] - lm[c]) / (2*h);
        const double mag = std::max(std::abs((double)an[c]), std::abs(fd));
        const double rel = mag < 1e-9 ? 0.0 : std::abs(an[c] - fd) / mag;
        worst = std::max(worst, rel);
    }
    return worst;
}

}  // namespace

TEST_CASE("multipole rotation derivative is exact for a dense rotation",
          "[multipole][symmetry]") {
    RealType R0[3][3], dR[3][3];
    axis_angle_mat(0.3, -0.7, 0.5, 0.9, R0);
    axis_angle_mat(0.2,  0.5, -0.8, 1.3, dR);
    for (int rank = 1; rank <= 4; ++rank) {
        const double w = rot_deriv_mismatch(rank, R0, dR);
        std::printf("    dense R, rank %d: worst rel = %.3e\n", rank, w);
        REQUIRE(w < 1e-6);
    }
}

TEST_CASE("multipole rotation derivative is exact for a signed permutation",
          "[multipole][symmetry]") {
    RealType R0[3][3] = {{0,-1,0},{1,0,0},{0,0,1}};   // 4-fold about z
    RealType dR[3][3];
    axis_angle_mat(0.2, 0.5, -0.8, 1.3, dR);
    for (int rank = 1; rank <= 4; ++rank) {
        const double w = rot_deriv_mismatch(rank, R0, dR);
        std::printf("    perm R,  rank %d: worst rel = %.3e\n", rank, w);
        REQUIRE(w < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// The energy path builds a symmetry copy's lab multipoles by composition
// (Rot(S) . Rot(R)); the rotation gradient differentiates the product M = S R.
// They agree only if Rot(S) . Rot(R) == Rot(S R).
// ---------------------------------------------------------------------------
namespace {

double homomorphism_mismatch(int rank, const RealType S[3][3], const RealType R[3][3]) {
    const int N = nhermsum(rank);
    RealType body[35];
    for (int c = 0; c < N; ++c)
        body[c] = static_cast<RealType>(std::sin(1.7*(c+1)) * std::exp(-0.15*c));

    // Composition: body -> R -> S, the way the expansion kernel builds a copy.
    RealType viaR[35], composed[35];
    rotate_cartesian_multipole(body, R, rank, viaR);
    rotate_cartesian_multipole(viaR, S, rank, composed);

    // Single product M = S R, the way the rotation gradient differentiates it.
    RealType M[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            RealType s = 0;
            for (int k = 0; k < 3; ++k) s += S[i][k] * R[k][j];
            M[i][j] = s;
        }
    RealType product[35];
    rotate_cartesian_multipole(body, M, rank, product);

    double worst = 0.0;
    for (int c = 0; c < N; ++c) {
        const double mag = std::max(std::abs((double)composed[c]), std::abs((double)product[c]));
        const double rel = mag < 1e-9 ? 0.0 : std::abs(composed[c] - product[c]) / mag;
        worst = std::max(worst, rel);
    }
    return worst;
}

}  // namespace

TEST_CASE("multipole rotation composes: Rot(S).Rot(R) == Rot(SR), dense S",
          "[multipole][symmetry]") {
    RealType S[3][3], R[3][3];
    // A 3-fold about z in a hexagonal cell, in Cartesian form.
    axis_angle_mat(0, 0, 1, 2.0 * M_PI / 3.0, S);
    axis_angle_mat(0.21, 0.14, 0.09, 0.27, R);
    for (int rank = 1; rank <= 4; ++rank) {
        const double w = homomorphism_mismatch(rank, S, R);
        std::printf("    dense S (3-fold), rank %d: worst rel = %.3e\n", rank, w);
        REQUIRE(w < 1e-9);
    }
}

TEST_CASE("multipole rotation composes: Rot(S).Rot(R) == Rot(SR), permutation S",
          "[multipole][symmetry]") {
    RealType S[3][3] = {{0,-1,0},{1,0,0},{0,0,1}};   // 4-fold about z
    RealType R[3][3];
    axis_angle_mat(0.21, 0.14, 0.09, 0.27, R);
    for (int rank = 1; rank <= 4; ++rank) {
        const double w = homomorphism_mismatch(rank, S, R);
        std::printf("    perm S (4-fold),  rank %d: worst rel = %.3e\n", rank, w);
        REQUIRE(w < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Whether the representation matrix D(S) of a symmetry operator on the stored
// multipole components is orthogonal. A gradient with respect to the components
// transforms by the adjoint, which equals D(S) only where D(S) is orthogonal.
// ---------------------------------------------------------------------------
namespace {

// D(S) as an N x N matrix: column c is Rot(S) applied to the c-th basis multipole.
void representation_matrix(int rank, const RealType S[3][3], double* D, int N) {
    for (int c = 0; c < N; ++c) {
        RealType e[35] = {0}, out[35];
        e[c] = 1;
        rotate_cartesian_multipole(e, S, rank, out);
        for (int r = 0; r < N; ++r) D[r*N + c] = out[r];
    }
}

double worst_orthogonality_defect(int rank, const RealType S[3][3]) {
    const int N = nhermsum(rank);
    std::vector<double> D(N*N);
    representation_matrix(rank, S, D.data(), N);
    double worst = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0;
            for (int k = 0; k < N; ++k) s += D[k*N + i] * D[k*N + j];   // (D^T D)_ij
            worst = std::max(worst, std::abs(s - (i == j ? 1.0 : 0.0)));
        }
    return worst;
}

}  // namespace

TEST_CASE("multipole representation is orthogonal for a signed permutation",
          "[multipole][symmetry]") {
    RealType S[3][3] = {{0,-1,0},{1,0,0},{0,0,1}};   // 4-fold about z
    for (int rank = 1; rank <= 4; ++rank)
        std::printf("    perm S (4-fold),  rank %d: |D^T D - I| = %.3e\n",
                    rank, worst_orthogonality_defect(rank, S));
}

TEST_CASE("multipole representation is orthogonal for a dense rotation",
          "[multipole][symmetry]") {
    RealType S[3][3];
    axis_angle_mat(0, 0, 1, 2.0 * M_PI / 3.0, S);    // 3-fold about z
    for (int rank = 1; rank <= 4; ++rank)
        std::printf("    dense S (3-fold), rank %d: |D^T D - I| = %.3e\n",
                    rank, worst_orthogonality_defect(rank, S));
}

// ---------------------------------------------------------------------------
// The stored Cartesian components carry multinomial multiplicities
// g = l!/(t!u!v!), and D is orthogonal in that metric: D^T G D == G, so the
// adjoint is D^-T = G D G^-1. For a signed permutation G D G^-1 == D.
// ---------------------------------------------------------------------------
namespace {

double factorial_d(int n) { double f = 1; for (int i = 2; i <= n; ++i) f *= i; return f; }

// Multinomial multiplicity of each stored component, indexed by hermite_index.
std::vector<double> multiplicity_weights(int rank, int N) {
    std::vector<double> g(N, 1.0);
    for (int l = 0; l <= rank; ++l)
        for (int t = l; t >= 0; --t)
            for (int uv = l - t; uv >= 0; --uv) {
                const int u = l - t - uv, v = uv;
                g[hermite_index(t, u, v)] =
                    factorial_d(l) / (factorial_d(t) * factorial_d(u) * factorial_d(v));
            }
    return g;
}

double weighted_orthogonality_defect(int rank, const RealType S[3][3]) {
    const int N = nhermsum(rank);
    std::vector<double> D(N*N);
    representation_matrix(rank, S, D.data(), N);
    const std::vector<double> g = multiplicity_weights(rank, N);
    double worst = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0;                                   // (D^T G D)_ij
            for (int k = 0; k < N; ++k) s += D[k*N + i] * g[k] * D[k*N + j];
            worst = std::max(worst, std::abs(s - (i == j ? g[i] : 0.0)));
        }
    return worst;
}

}  // namespace

TEST_CASE("multipole representation is orthogonal in the multiplicity metric",
          "[multipole][symmetry]") {
    RealType Sd[3][3], Sp[3][3] = {{0,-1,0},{1,0,0},{0,0,1}};
    axis_angle_mat(0, 0, 1, 2.0 * M_PI / 3.0, Sd);
    for (int rank = 1; rank <= 4; ++rank)
        std::printf("    dense S (3-fold), rank %d: |D^T G D - G| = %.3e\n",
                    rank, weighted_orthogonality_defect(rank, Sd));
    for (int rank = 1; rank <= 4; ++rank)
        std::printf("    perm S (4-fold),  rank %d: |D^T G D - G| = %.3e\n",
                    rank, weighted_orthogonality_defect(rank, Sp));
}

// ---------------------------------------------------------------------------
// Cell-DOF gradient under the symmetry-adapted strain basis: eps = B^T q and
// dE/dq = B dE/deps. Rhombohedral is the only lattice whose basis rows mix
// Voigt components and depend on the cell's orientation.
// ---------------------------------------------------------------------------
namespace {

// Central difference of dE/dq_s against the analytic B dE/deps, for every cell
// DOF the given lattice allows.
template <int RANK>
double worst_cell_dof_mismatch(Fixture& f, int lattice, double h, bool verbose = true) {
    f.bws.dofs.cell_ref   = decltype(f.bws.dofs.cell_ref)("cell_ref", 1, 3, 3);
    f.bws.dofs.strain     = decltype(f.bws.dofs.strain)("strain", 1, 6);
    f.bws.dofs.d_g_strain = decltype(f.bws.dofs.d_g_strain)("d_g_strain", 1, 6);
    const int ncd = batch_set_strain_dofs(f.bws.geom, f.bws.dofs, lattice, /*fix_angles=*/false);
    Kokkos::deep_copy(f.bws.dofs.cell_ref, f.bws.geom.cell);
    Kokkos::deep_copy(f.bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(f.bws.dofs.d_g_strain, RealType(0));

    RealType basis[36];
    {
        auto hb = testing::host_snapshot(f.bws.dofs.cell_dof_basis);
        for (int s = 0; s < 6; ++s)
            for (int v = 0; v < 6; ++v) basis[s*6 + v] = hb(0, s, v);
    }

    // Analytic dE/dq at q = 0.
    RealType eps0[6] = {0,0,0,0,0,0};
    energy_at_strain<RANK>(f, eps0);
    RealType gq[6];
    {
        auto hg = testing::host_snapshot(f.bws.dofs.d_g_strain);
        RealType geps[6];
        for (int v = 0; v < 6; ++v) geps[v] = hg(0, v);
        basis_project(basis, geps, ncd, gq);
    }

    double worst = 0.0;
    for (int s = 0; s < ncd; ++s) {
        RealType qp[6] = {0,0,0,0,0,0}, qm[6] = {0,0,0,0,0,0}, ep[6], em[6];
        qp[s] =  static_cast<RealType>(h);
        qm[s] = -static_cast<RealType>(h);
        basis_expand(basis, qp, ncd, ep);
        basis_expand(basis, qm, ncd, em);
        const double Ep = energy_at_strain<RANK>(f, ep);
        const double Em = energy_at_strain<RANK>(f, em);
        const double fd = (Ep - Em) / (2.0 * h);
        const double an = gq[s];
        // Floor the denominator so components zero by symmetry are not judged
        // by a ratio against FD noise.
        const double mag = std::max({std::abs(an), std::abs(fd), 1e-10});
        const double rel = klasp::testing::fd_agrees(an, fd)
                         ? 0.0 : std::abs(an - fd) / mag;
        if (verbose)
            std::printf("    q_%d: analytic=% .8e  fd=% .8e  rel=%.2e\n", s, an, fd, rel);
        worst = std::max(worst, rel);
    }
    return worst;
}

// R-3 on rhombohedral axes: the 3-fold about a+b+c cyclically permutes the
// fractional axes, and -3 adds the inversions.
const std::vector<std::array<int, 9>> RHOMB_R3 = {
    {1, 0, 0,  0, 1, 0,  0, 0, 1},
    {0, 0, 1,  1, 0, 0,  0, 1, 0},
    {0, 1, 0,  0, 0, 1,  1, 0, 0},
    {-1, 0, 0,  0,-1, 0,  0, 0,-1},
    {0, 0,-1, -1, 0, 0,  0,-1, 0},
    {0,-1, 0,  0, 0,-1, -1, 0, 0},
};

}  // namespace

// P-1 for the triclinic control: a 2-fold with a free 6-DOF strain would be
// sheared out of its own symmetry.
TEST_CASE("cell-DOF gradient matches FD, triclinic basis", "[gradients][strain]") {
    auto f = make_sym_fixture(7.0, 8.2, 9.5, 88, 97, 103, PBAR1);
    std::printf("  triclinic basis (6 DOF), rank 4:\n");
    REQUIRE(worst_cell_dof_mismatch<4>(f, static_cast<int>(TRICLINIC), 1e-6) < 1e-5);
}

TEST_CASE("cell-DOF gradient matches FD, hexagonal basis", "[gradients][strain]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 120, THREEFOLD);
    std::printf("  hexagonal basis (2 DOF), rank 4:\n");
    REQUIRE(worst_cell_dof_mismatch<4>(f, static_cast<int>(HEXAGONAL), 1e-6) < 1e-5);
}

// Space group 148.
TEST_CASE("cell-DOF gradient matches FD, rhombohedral basis", "[gradients][strain]") {
    auto f = make_sym_fixture(8.0, 8.0, 8.0, 72, 72, 72, RHOMB_R3);
    std::printf("  rhombohedral basis (2 DOF), rank 4:\n");
    REQUIRE(worst_cell_dof_mismatch<4>(f, static_cast<int>(RHOMBOHEDRAL), 1e-6) < 1e-5);
}

// ---------------------------------------------------------------------------
// The rhombohedral basis is built once per stage and must stay valid after
// clip_cell_step re-anchors cell_ref to the current cell without rebuilding it.
// ---------------------------------------------------------------------------
namespace {

template <int RANK>
double worst_cell_dof_mismatch_at(Fixture& f, int lattice, const double* q0_in,
                                  double h, bool reanchor, bool verbose = true) {
    f.bws.dofs.cell_ref   = decltype(f.bws.dofs.cell_ref)("cell_ref", 1, 3, 3);
    f.bws.dofs.strain     = decltype(f.bws.dofs.strain)("strain", 1, 6);
    f.bws.dofs.d_g_strain = decltype(f.bws.dofs.d_g_strain)("d_g_strain", 1, 6);
    const int ncd = batch_set_strain_dofs(f.bws.geom, f.bws.dofs, lattice, /*fix_angles=*/false);
    Kokkos::deep_copy(f.bws.dofs.cell_ref, f.bws.geom.cell);
    Kokkos::deep_copy(f.bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(f.bws.dofs.d_g_strain, RealType(0));

    RealType basis[36];
    {
        auto hb = testing::host_snapshot(f.bws.dofs.cell_dof_basis);
        for (int s = 0; s < 6; ++s)
            for (int v = 0; v < 6; ++v) basis[s*6 + v] = hb(0, s, v);
    }

    RealType q0[6] = {0,0,0,0,0,0};
    for (int s = 0; s < ncd; ++s) q0[s] = static_cast<RealType>(q0_in[s]);

    if (reanchor) {
        // Deform to q0, make that the reference and reset amplitudes, without
        // rebuilding the basis, as clip_cell_step does.
        RealType e0[6];
        basis_expand(basis, q0, ncd, e0);
        energy_at_strain<RANK>(f, e0);
        Kokkos::deep_copy(f.bws.dofs.cell_ref, f.bws.geom.cell);
        for (int s = 0; s < ncd; ++s) q0[s] = 0;
    }

    RealType eb[6];
    basis_expand(basis, q0, ncd, eb);
    energy_at_strain<RANK>(f, eb);
    RealType gq[6];
    {
        auto hg = testing::host_snapshot(f.bws.dofs.d_g_strain);
        RealType geps[6];
        for (int v = 0; v < 6; ++v) geps[v] = hg(0, v);
        basis_project(basis, geps, ncd, gq);
    }

    double worst = 0.0;
    for (int s = 0; s < ncd; ++s) {
        RealType qp[6], qm[6], ep[6], em[6];
        for (int t = 0; t < 6; ++t) { qp[t] = q0[t]; qm[t] = q0[t]; }
        qp[s] += static_cast<RealType>(h);
        qm[s] -= static_cast<RealType>(h);
        basis_expand(basis, qp, ncd, ep);
        basis_expand(basis, qm, ncd, em);
        const double Ep = energy_at_strain<RANK>(f, ep);
        const double Em = energy_at_strain<RANK>(f, em);
        const double fd = (Ep - Em) / (2.0 * h);
        const double an = gq[s];
        // Floor the denominator so components zero by symmetry are not judged
        // by a ratio against FD noise.
        const double mag = std::max({std::abs(an), std::abs(fd), 1e-10});
        const double rel = klasp::testing::fd_agrees(an, fd)
                         ? 0.0 : std::abs(an - fd) / mag;
        if (verbose)
            std::printf("    q_%d: analytic=% .8e  fd=% .8e  rel=%.2e\n", s, an, fd, rel);
        worst = std::max(worst, rel);
    }
    return worst;
}

}  // namespace

// Away from the reference cell the gradient must be the derivative with respect
// to the cumulative strain, which differs from the virial at the current cell
// by F^-T (F = I + eps).
TEST_CASE("cell-DOF gradient matches FD at nonzero strain, rhombohedral",
          "[gradients][strain]") {
    auto f = make_sym_fixture(8.0, 8.0, 8.0, 72, 72, 72, RHOMB_R3);
    const double q0[6] = {0.035, -0.022, 0, 0, 0, 0};
    std::printf("  rhombohedral, deformed cell, rank 4:\n");
    REQUIRE(worst_cell_dof_mismatch_at<4>(f, static_cast<int>(RHOMBOHEDRAL), q0,
                                          1e-6, /*reanchor=*/false) < 1e-5);
}

TEST_CASE("cell-DOF gradient matches FD after a re-anchor, rhombohedral",
          "[gradients][strain]") {
    auto f = make_sym_fixture(8.0, 8.0, 8.0, 72, 72, 72, RHOMB_R3);
    const double q0[6] = {0.035, -0.022, 0, 0, 0, 0};
    std::printf("  rhombohedral, re-anchored reference, rank 4:\n");
    REQUIRE(worst_cell_dof_mismatch_at<4>(f, static_cast<int>(RHOMBOHEDRAL), q0,
                                          1e-6, /*reanchor=*/true) < 1e-5);
}

// The same check on a second lattice: the cumulative-strain factor is a
// property of the parameterisation, not of any one lattice.
TEST_CASE("cell-DOF gradient matches FD at nonzero strain, hexagonal",
          "[gradients][strain]") {
    auto f = make_sym_fixture(7.0, 7.0, 9.5, 90, 90, 120, THREEFOLD);
    const double q0[6] = {0.035, -0.022, 0, 0, 0, 0};
    std::printf("  hexagonal, deformed cell, rank 4:\n");
    REQUIRE(worst_cell_dof_mismatch_at<4>(f, static_cast<int>(HEXAGONAL), q0,
                                          1e-6, /*reanchor=*/false) < 1e-5);
}

// ---------------------------------------------------------------------------
// A rhombohedral strain basis is derived from the cell (its 3-fold runs along
// a+b+c), so it belongs to the structure; other lattices' bases are constants.
// ---------------------------------------------------------------------------
namespace {

// Row-major, COLUMNS are the lattice vectors (klasp convention).
void rhombohedral_cell(double a, double ang_deg, RealType out[9]) {
    const double r = ang_deg * M_PI / 180.0;
    const double ca = std::cos(r), sg = std::sin(r);
    const double bx = a * ca, by = a * sg;
    const double cx = a * ca, cy = a * (ca - ca * ca) / sg;
    const double cz = std::sqrt(std::max(0.0, a*a - cx*cx - cy*cy));
    const double col[3][3] = {{a, 0, 0}, {bx, by, 0}, {cx, cy, cz}};
    for (int c = 0; c < 3; ++c)
        for (int r2 = 0; r2 < 3; ++r2) out[r2*3 + c] = static_cast<RealType>(col[c][r2]);
}

// Worst departure from a=b=c, alpha=beta=gamma after straining `cell` along each
// row of `basis`.
double rhombohedral_decay(const RealType cell[9], const RealType* basis, int ndof) {
    double worst = 0.0;
    for (int s = 0; s < ndof; ++s)
        for (double mag : {-0.05, -0.02, 0.02, 0.05}) {
            double eps[3][3] = {};
            const RealType* B = basis + s*6;
            eps[0][0] = mag*B[0]; eps[1][1] = mag*B[1]; eps[2][2] = mag*B[2];
            eps[1][2] = eps[2][1] = 0.5*mag*B[3];
            eps[0][2] = eps[2][0] = 0.5*mag*B[4];
            eps[0][1] = eps[1][0] = 0.5*mag*B[5];
            double d[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    double v = 0;
                    for (int k = 0; k < 3; ++k)
                        v += ((r == k ? 1.0 : 0.0) + eps[r][k]) * cell[k*3 + c];
                    d[r][c] = v;
                }
            auto len = [&](int c) {
                return std::sqrt(d[0][c]*d[0][c] + d[1][c]*d[1][c] + d[2][c]*d[2][c]);
            };
            auto ang = [&](int i, int j) {
                double dot = 0;
                for (int r = 0; r < 3; ++r) dot += d[r][i]*d[r][j];
                return std::acos(std::max(-1.0, std::min(1.0, dot/(len(i)*len(j)))))
                       * 180.0 / M_PI;
            };
            worst = std::max({worst,
                              std::abs(len(0) - len(1)), std::abs(len(1) - len(2)),
                              std::abs(ang(1,2) - ang(0,2)), std::abs(ang(0,2) - ang(0,1))});
        }
    return worst;
}

}  // namespace

TEST_CASE("a rhombohedral strain basis is specific to the cell it was built from",
          "[strain][symmetry]") {
    RealType cellA[9], cellB[9], basisA[36], basisB[36];
    rhombohedral_cell(8.0, 72.0, cellA);
    rhombohedral_cell(8.0, 96.0, cellB);          // a differently-oriented 3-fold
    const int nA = build_strain_basis(static_cast<int>(RHOMBOHEDRAL), cellA, basisA, 2);
    const int nB = build_strain_basis(static_cast<int>(RHOMBOHEDRAL), cellB, basisB, 2);
    REQUIRE(nA == 2);
    REQUIRE(nB == 2);

    // Each cell's own basis holds its lattice type exactly...
    CHECK(rhombohedral_decay(cellA, basisA, nA) < 1e-10);
    CHECK(rhombohedral_decay(cellB, basisB, nB) < 1e-10);

    // ...and the other structure's does not, so a refilled slot must rebuild
    // its basis.
    CHECK(rhombohedral_decay(cellB, basisA, nA) > 1e-3);
}

// The constant-basis lattices do not depend on the cell.
TEST_CASE("non-rhombohedral strain bases do not depend on the cell",
          "[strain][symmetry]") {
    RealType cellA[9], cellB[9], bA[36], bB[36];
    rhombohedral_cell(8.0, 72.0, cellA);
    rhombohedral_cell(9.5, 96.0, cellB);
    for (int lt : {static_cast<int>(TRICLINIC), static_cast<int>(MONOCLINIC),
                   static_cast<int>(ORTHORHOMBIC), static_cast<int>(TETRAGONAL),
                   static_cast<int>(HEXAGONAL), static_cast<int>(CUBIC)}) {
        const int nA = build_strain_basis(lt, cellA, bA, 6);
        const int nB = build_strain_basis(lt, cellB, bB, 6);
        REQUIRE(nA == nB);
        for (int i = 0; i < 36; ++i) REQUIRE(bA[i] == bB[i]);
    }
}
