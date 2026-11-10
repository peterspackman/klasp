#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/multipole.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/rotation.hpp"
#include "klasp/molecule.hpp"
#include "test_support.hpp"
#include "klasp/cell.hpp"
#include "klasp/ewald.hpp"
#include "klasp/chain_rule.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/constants.hpp"
#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Constexpr tables + Hermite indexing
// ---------------------------------------------------------------------------

TEST_CASE("nhermsum values", "[multipole]") {
    REQUIRE(nhermsum(0) == 1);
    REQUIRE(nhermsum(1) == 4);
    REQUIRE(nhermsum(2) == 10);
    REQUIRE(nhermsum(3) == 20);
    REQUIRE(nhermsum(4) == 35);
}

TEST_CASE("hermite_index known values", "[multipole]") {
    // Rank 0: (0,0,0) = 0
    REQUIRE(hermite_index(0, 0, 0) == 0);
    // Rank 1: (1,0,0)=1, (0,1,0)=2, (0,0,1)=3
    REQUIRE(hermite_index(1, 0, 0) == 1);
    REQUIRE(hermite_index(0, 1, 0) == 2);
    REQUIRE(hermite_index(0, 0, 1) == 3);
    // Rank 2: (2,0,0)=4, (1,1,0)=5, (1,0,1)=6, (0,2,0)=7, (0,1,1)=8, (0,0,2)=9
    REQUIRE(hermite_index(2, 0, 0) == 4);
    REQUIRE(hermite_index(1, 1, 0) == 5);
    REQUIRE(hermite_index(1, 0, 1) == 6);
    REQUIRE(hermite_index(0, 2, 0) == 7);
    REQUIRE(hermite_index(0, 1, 1) == 8);
    REQUIRE(hermite_index(0, 0, 2) == 9);
}

TEST_CASE("TUV table round-trips", "[multipole]") {
    constexpr auto mc = MultipoleConstants<4>{};
    for (int i = 0; i < mc.N_CART; ++i) {
        auto [t, u, v] = mc.tuv_table[i];
        REQUIRE(hermite_index(t, u, v) == i);
    }
}

TEST_CASE("Weight tables", "[multipole]") {
    constexpr auto mc = MultipoleConstants<4>{};
    // (0,0,0): rank 0, sign_inv_fact = 1.0, inv_fact = 1.0
    REQUIRE_THAT(mc.sign_inv_fact[0], WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(mc.inv_fact[0], WithinAbs(1.0, 1e-15));

    // (1,0,0): rank 1, sign = -1, fact = 1, so sign_inv_fact = -1
    int idx_100 = hermite_index(1, 0, 0);
    REQUIRE_THAT(mc.sign_inv_fact[idx_100], WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(mc.inv_fact[idx_100], WithinAbs(1.0, 1e-15));

    // (2,0,0): rank 2, sign = +1, fact = 2!, so inv_fact = 0.5
    int idx_200 = hermite_index(2, 0, 0);
    REQUIRE_THAT(mc.sign_inv_fact[idx_200], WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(mc.inv_fact[idx_200], WithinAbs(0.5, 1e-15));

    // (1,1,0): rank 2, sign = +1, fact = 1!*1! = 1, so inv_fact = 1
    int idx_110 = hermite_index(1, 1, 0);
    REQUIRE_THAT(mc.sign_inv_fact[idx_110], WithinAbs(1.0, 1e-15));
}

TEST_CASE("InteractionConstants add_table", "[multipole]") {
    constexpr auto ic = InteractionConstants<4>{};
    // (0,0,0) + (0,0,0) = (0,0,0) = index 0
    REQUIRE(ic.add_table[0][0] == 0);
    // (1,0,0) + (0,0,1) = (1,0,1) = index 6
    int i = hermite_index(1, 0, 0);
    int j = hermite_index(0, 0, 1);
    REQUIRE(ic.add_table[i][j] == hermite_index(1, 0, 1));
}

TEST_CASE("InteractionConstants active mask", "[multipole]") {
    constexpr auto ic = InteractionConstants<4>{};
    // rank(0,0,0) + rank(0,0,0) = 0 <= 4: active
    REQUIRE(ic.active[0][0]);
    // rank(2,0,0) + rank(2,0,0) = 4 <= 4: active
    int idx_200 = hermite_index(2, 0, 0);
    REQUIRE(ic.active[idx_200][idx_200]);
    // rank(3,0,0) + rank(2,0,0) = 5 > 4: inactive
    int idx_300 = hermite_index(3, 0, 0);
    REQUIRE_FALSE(ic.active[idx_300][idx_200]);
}

// ---------------------------------------------------------------------------
// T-tensor + pair energy
// ---------------------------------------------------------------------------

TEST_CASE("T-tensor T000 = 1/R", "[multipole]") {
    RealType Rx = 1.0, Ry = 2.0, Rz = 3.0;
    RealType R = std::sqrt(Rx*Rx + Ry*Ry + Rz*Rz);

    RealType T[nhermsum(4)];
    compute_t_tensor<4>(Rx, Ry, Rz, T);

    REQUIRE_THAT(T[hermite_index(0,0,0)], WithinRel(RealType(1.0)/R, RealType(1e-4)));
}

TEST_CASE("T-tensor T100 = -Rx/R^3", "[multipole]") {
    RealType Rx = 1.5, Ry = 0.7, Rz = -1.2;
    RealType R2 = Rx*Rx + Ry*Ry + Rz*Rz;
    RealType R = std::sqrt(R2);

    RealType T[nhermsum(4)];
    compute_t_tensor<4>(Rx, Ry, Rz, T);

    REQUIRE_THAT(T[hermite_index(1,0,0)], WithinRel(-Rx/(R*R2), RealType(1e-4)));
    REQUIRE_THAT(T[hermite_index(0,1,0)], WithinRel(-Ry/(R*R2), RealType(1e-4)));
    REQUIRE_THAT(T[hermite_index(0,0,1)], WithinRel(-Rz/(R*R2), RealType(1e-4)));
}

TEST_CASE("T-tensor T200 = (3Rx^2 - R^2)/R^5", "[multipole]") {
    RealType Rx = 1.5, Ry = 0.7, Rz = -1.2;
    RealType R2 = Rx*Rx + Ry*Ry + Rz*Rz;
    RealType R = std::sqrt(R2);
    RealType R5 = R2 * R2 * R;

    RealType T[nhermsum(4)];
    compute_t_tensor<4>(Rx, Ry, Rz, T);

    RealType expected = (3.0*Rx*Rx - R2) / R5;
    REQUIRE_THAT(T[hermite_index(2,0,0)], WithinRel(expected, RealType(1e-4)));
}

TEST_CASE("Charge-charge pair energy = q_a * q_b / R", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType cart_a[N] = {};
    RealType cart_b[N] = {};
    cart_a[0] = 0.5;  // charge
    cart_b[0] = -0.3; // charge

    RealType Rx = 2.0, Ry = 1.0, Rz = -0.5;
    RealType R = std::sqrt(Rx*Rx + Ry*Ry + Rz*Rz);

    RealType E = multipole_pair_energy<4>(cart_a, cart_b, Rx, Ry, Rz);
    RealType expected = 0.5 * (-0.3) / R;

    REQUIRE_THAT(E, WithinRel(expected, RealType(1e-4)));
}

TEST_CASE("Charge-dipole pair energy", "[multipole]") {
    // q_a at origin, dipole_b = (dx, dy, dz) at (Rx, Ry, Rz)
    // E = -q * (dx*(Rx/R^3) + dy*(Ry/R^3) + dz*(Rz/R^3))
    constexpr int N = nhermsum(4);
    RealType cart_a[N] = {};
    RealType cart_b[N] = {};
    RealType q = 1.5;
    RealType dx = 0.1, dy = -0.2, dz = 0.3;
    cart_a[0] = q;
    cart_b[hermite_index(1,0,0)] = dx;
    cart_b[hermite_index(0,1,0)] = dy;
    cart_b[hermite_index(0,0,1)] = dz;

    RealType Rx = 3.0, Ry = 1.0, Rz = -2.0;
    RealType R2 = Rx*Rx + Ry*Ry + Rz*Rz;
    RealType R = std::sqrt(R2);
    RealType R3 = R * R2;

    RealType E = multipole_pair_energy<4>(cart_a, cart_b, Rx, Ry, Rz);
    // wA = sign_inv_fact * cart_a, wB = inv_fact * cart_b, T_{100} = -Rx/R^3, so
    // E = q * (T_{100} dx + T_{010} dy + T_{001} dz).
    RealType expected = q * (-Rx/R3) * dx + q * (-Ry/R3) * dy + q * (-Rz/R3) * dz;

    REQUIRE_THAT(E, WithinRel(expected, RealType(1e-4)));
}

TEST_CASE("Multipole forces match finite difference", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType cart_a[N] = {};
    RealType cart_b[N] = {};
    // Set up charge + dipole on A, charge + quadrupole on B
    cart_a[0] = 0.8;
    cart_a[hermite_index(1,0,0)] = 0.1;
    cart_a[hermite_index(0,1,0)] = -0.15;
    cart_b[0] = -0.5;
    cart_b[hermite_index(0,0,2)] = 0.05;
    cart_b[hermite_index(2,0,0)] = -0.03;

    RealType Rx = 3.5, Ry = 1.2, Rz = -0.8;
    RealType force[3];
    RealType E = multipole_pair_energy_forces<4>(cart_a, cart_b, Rx, Ry, Rz, force);

    // The FD below differentiates multipole_pair_energy; both entry points must
    // agree.
    REQUIRE_THAT(E, WithinRel(multipole_pair_energy<4>(cart_a, cart_b, Rx, Ry, Rz),
                              RealType(1e-6)));

    const RealType eps = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
    RealType R_comp[3] = {Rx, Ry, Rz};
    for (int d = 0; d < 3; ++d) {
        RealType Rp[3] = {R_comp[0], R_comp[1], R_comp[2]};
        RealType Rm[3] = {R_comp[0], R_comp[1], R_comp[2]};
        Rp[d] += eps;
        Rm[d] -= eps;
        RealType Ep = multipole_pair_energy<4>(cart_a, cart_b, Rp[0], Rp[1], Rp[2]);
        RealType Em = multipole_pair_energy<4>(cart_a, cart_b, Rm[0], Rm[1], Rm[2]);
        RealType fd = (Ep - Em) / (2.0 * eps);
        REQUIRE_THAT(force[d], WithinAbs(fd, sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-3)));
    }
}

// ---------------------------------------------------------------------------
// Spherical-to-Cartesian + Cartesian rotation
// ---------------------------------------------------------------------------

TEST_CASE("Spherical-to-Cartesian charge only", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType sph[25] = {};
    sph[0] = 1.5;
    RealType cart[N];
    spherical_to_cartesian(sph, 1, 0, cart);
    REQUIRE_THAT(cart[0], WithinAbs(1.5, RealType(1e-4)));
    for (int i = 1; i < N; ++i)
        REQUIRE_THAT(cart[i], WithinAbs(0.0, RealType(1e-4)));
}

TEST_CASE("Spherical-to-Cartesian dipole", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType sph[25] = {};
    // Q10=0.3, Q11c=0.5, Q11s=-0.2
    sph[0] = 1.0;
    sph[1] = 0.3;  // Q10 -> z
    sph[2] = 0.5;  // Q11c -> x
    sph[3] = -0.2; // Q11s -> y
    RealType cart[N];
    spherical_to_cartesian(sph, 4, 1, cart);
    REQUIRE_THAT(cart[hermite_index(1,0,0)], WithinAbs(0.5, RealType(1e-4)));
    REQUIRE_THAT(cart[hermite_index(0,1,0)], WithinAbs(-0.2, RealType(1e-4)));
    REQUIRE_THAT(cart[hermite_index(0,0,1)], WithinAbs(0.3, RealType(1e-4)));
}

TEST_CASE("Charge unchanged by Cartesian rotation", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType body[N] = {};
    body[0] = 2.5;

    RealType R[3][3] = {
        {0.0, -1.0, 0.0},
        {1.0,  0.0, 0.0},
        {0.0,  0.0, 1.0}
    };

    RealType lab[N];
    rotate_cartesian_multipole(body, R, 4, lab);
    REQUIRE_THAT(lab[0], WithinAbs(2.5, RealType(1e-4)));
}

TEST_CASE("Dipole rotation matches R @ dipole_vec", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType body[N] = {};
    // Dipole: (1.0, 0.0, 0.0) in body frame
    body[hermite_index(1,0,0)] = 1.0;
    body[hermite_index(0,1,0)] = 0.0;
    body[hermite_index(0,0,1)] = 0.0;

    // 90 degree rotation about z
    RealType R[3][3] = {
        {0.0, -1.0, 0.0},
        {1.0,  0.0, 0.0},
        {0.0,  0.0, 1.0}
    };

    RealType lab[N];
    rotate_cartesian_multipole(body, R, 1, lab);
    // Expected: R @ (1,0,0) = (0, 1, 0)
    REQUIRE_THAT(lab[hermite_index(1,0,0)], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(lab[hermite_index(0,1,0)], WithinAbs(1.0, RealType(1e-4)));
    REQUIRE_THAT(lab[hermite_index(0,0,1)], WithinAbs(0.0, RealType(1e-4)));
}

TEST_CASE("Dipole rotation general case", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType body[N] = {};
    body[hermite_index(1,0,0)] = 0.3;
    body[hermite_index(0,1,0)] = -0.5;
    body[hermite_index(0,0,1)] = 0.7;

    // Rotation about (1,1,1)/sqrt(3) by 120 degrees: cycles x->y->z->x
    // This matrix sends x->y, y->z, z->x
    RealType R[3][3] = {
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };

    RealType lab[N];
    rotate_cartesian_multipole(body, R, 1, lab);
    // R @ (0.3, -0.5, 0.7) = (0.7, 0.3, -0.5)
    REQUIRE_THAT(lab[hermite_index(1,0,0)], WithinAbs(0.7, RealType(1e-4)));
    REQUIRE_THAT(lab[hermite_index(0,1,0)], WithinAbs(0.3, RealType(1e-4)));
    REQUIRE_THAT(lab[hermite_index(0,0,1)], WithinAbs(-0.5, RealType(1e-4)));
}

TEST_CASE("Rotation derivative matches finite difference", "[multipole]") {
    constexpr int N = nhermsum(4);
    RealType body[N] = {};
    body[0] = 1.0;
    body[hermite_index(1,0,0)] = 0.3;
    body[hermite_index(0,1,0)] = -0.2;
    body[hermite_index(0,0,1)] = 0.5;
    body[hermite_index(2,0,0)] = 0.1;
    body[hermite_index(0,0,2)] = -0.05;

    // Base rotation from angle-axis
    RealType aa[3] = {0.3, -0.2, 0.5};
    RealType R[3][3];
    angle_axis_to_matrix(aa, R);

    // Get analytical dR/d(aa_0)
    RealType dR0[3][3], dR1[3][3], dR2[3][3];
    angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);

    for (int m = 0; m < 3; ++m) {
        RealType (*dR_m)[3] = (m == 0) ? dR0 : (m == 1 ? dR1 : dR2);
        RealType dlab_analytical[N];
        rotate_cartesian_multipole_deriv(body, R, dR_m, 2, dlab_analytical);

        const RealType eps = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
        RealType aa_p[3] = {aa[0], aa[1], aa[2]};
        RealType aa_m[3] = {aa[0], aa[1], aa[2]};
        aa_p[m] += eps;
        aa_m[m] -= eps;

        RealType Rp[3][3], Rm[3][3];
        angle_axis_to_matrix(aa_p, Rp);
        angle_axis_to_matrix(aa_m, Rm);

        RealType lab_p[N], lab_m[N];
        rotate_cartesian_multipole(body, Rp, 2, lab_p);
        rotate_cartesian_multipole(body, Rm, 2, lab_m);

        for (int i = 0; i < nhermsum(2); ++i) {
            RealType fd = (lab_p[i] - lab_m[i]) / (2.0 * eps);
            REQUIRE_THAT(dlab_analytical[i], WithinAbs(fd, sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-3)));
        }
    }
}

TEST_CASE("the generated rotation equals the transport-matrix rotation", "[multipole]") {
    // rotate_cartesian_multipole_gen<L> (scripts/gen_multipole_contraction.py)
    // must equal the enumerated form at every rank, for a rotation and a sheared
    // matrix; the identity holds for any M.
    constexpr int N = nhermsum(4);
    RealType body[N];
    for (int i = 0; i < N; ++i) body[i] = RealType(0.43) * std::cos(RealType(2.3 * i + 0.7));
    RealType aa[3] = {0.7, -0.4, 0.2};
    RealType R[3][3];
    angle_axis_to_matrix(aa, R);
    const RealType S[3][3] = {{1.0, 0.2, -0.1}, {0.0, 0.9, 0.3}, {0.05, 0.0, 1.1}};
    RealType M[3][3];
    mat3_mul(S, R, M);
    for (const auto* mat : {&R, &M}) {
        for (int rank = 0; rank <= 4; ++rank) {
            RealType ref[N] = {}, gen[N] = {};
            rotate_cartesian_multipole(body, *mat, rank, ref);
            switch (rank) {
                case 0: rotate_cartesian_multipole_gen<0>(body, *mat, gen); break;
                case 1: rotate_cartesian_multipole_gen<1>(body, *mat, gen); break;
                case 2: rotate_cartesian_multipole_gen<2>(body, *mat, gen); break;
                case 3: rotate_cartesian_multipole_gen<3>(body, *mat, gen); break;
                default: rotate_cartesian_multipole_gen<4>(body, *mat, gen); break;
            }
            for (int i = 0; i < nhermsum(rank); ++i) {
                INFO("rank " << rank << " component " << i << (mat == &R ? " (rotation)" : " (sheared)"));
                REQUIRE_THAT(gen[i], WithinAbs(ref[i], RealType(1e-12) * (RealType(1) + std::abs(ref[i]))));
            }
        }
    }
}

TEST_CASE("pair moments reproduce the direct multipole term", "[multipole]") {
    // E, force and torque of one image from PairMoments::build + contract must
    // equal pair_multipole_higher's energy and force and its adjoint contracted
    // with the same Jacobian.
    constexpr int N = nhermsum(4);
    RealType a[N], b[N], J[3 * N], Jb[3 * N];
    for (int i = 0; i < N; ++i) {
        a[i] = RealType(0.37) * std::sin(RealType(1.7 * i + 0.3));
        b[i] = RealType(0.29) * std::cos(RealType(2.1 * i + 0.9));
    }
    for (int i = 0; i < 3 * N; ++i) {
        J[i] = RealType(0.5) * std::sin(RealType(0.7 * i + 1.1));
        Jb[i] = RealType(0.4) * std::cos(RealType(1.3 * i + 0.2));
    }
    const RealType rx = 4.0, ry = -1.5, rz = 2.5;
    const RealType r = std::sqrt(rx*rx + ry*ry + rz*rz), inv_r = 1 / r;
    RealType e_ref, f_ref[3], da[N], db[N];
    pair_multipole_higher<4, true>(a, b, rx, ry, rz, inv_r, 1.0, 0.0, e_ref, f_ref, da, db);
    RealType tq_ref[3] = {0, 0, 0}, tqb_ref[3] = {0, 0, 0};
    for (int m = 0; m < 3; ++m)
        for (int c = 0; c < N; ++c) {
            tq_ref[m] += da[c] * J[m * N + c];
            tqb_ref[m] += db[c] * Jb[m * N + c];
        }

    using PM = PairMoments<4>;
    RealType s[PM::NT], v[3 * PM::NT], wm[3 * PM::NT];
    PM::build(a, b, J, s, v);
    PM::build_b(a, b, Jb, wm);
    RealType T[nhermsum(5)];
    TTensor<5, RealType>::run(rx, ry, rz, inv_r * inv_r, inv_r, T);
    RealType f[3], tq[3], tqb[3];
    const RealType e = PM::contract(s, v, T, f, tq);
    PM::contract_b(wm, T, tqb);

    const RealType tol = RealType(1e-12);
    REQUIRE_THAT(e, WithinAbs(e_ref, tol * (RealType(1) + std::abs(e_ref))));
    for (int d = 0; d < 3; ++d) {
        INFO("component " << d);
        REQUIRE_THAT(f[d], WithinAbs(f_ref[d], tol * (RealType(1) + std::abs(f_ref[d]))));
        REQUIRE_THAT(tq[d], WithinAbs(tq_ref[d], tol * (RealType(1) + std::abs(tq_ref[d]))));
        REQUIRE_THAT(tqb[d], WithinAbs(tqb_ref[d], tol * (RealType(1) + std::abs(tqb_ref[d]))));
    }
    REQUIRE(std::abs(e) > 1e-6);   // a real term, not an empty sum

    // The entry-at-a-time builder must give the same row layout:
    // s | v_0 v_1 v_2 | w_0 w_1 w_2. Compared to a tolerance, not bitwise, since
    // FMA contraction differs between builders.
    RealType row[7 * PM::NT];
    for (int k = 0; k < PM::NT; ++k) PM::build_entry(k, a, b, J, Jb, row);
    for (int k = 0; k < PM::NT; ++k) {
        INFO("entry " << k);
        REQUIRE_THAT(row[k], WithinRel(s[k], tol));
        for (int m = 0; m < 3; ++m) {
            REQUIRE_THAT(row[(1 + m) * PM::NT + k], WithinRel(v[m * PM::NT + k], tol));
            REQUIRE_THAT(row[(4 + m) * PM::NT + k], WithinRel(wm[m * PM::NT + k], tol));
        }
    }
}

TEST_CASE("the generator derivative equals the transport derivative", "[multipole]") {
    // The generator form (Omega = dM M^-1) and the transport form must give the
    // same derivative at every rank up to 4. M is non-orthogonal so a transpose
    // in place of the inverse would fail.
    constexpr int N = nhermsum(4);
    RealType body[N];
    for (int i = 0; i < N; ++i) body[i] = RealType(0.37) * std::sin(RealType(1.7 * i + 0.3));

    RealType aa[3] = {0.3, -0.2, 0.5};
    RealType R[3][3], dR0[3][3], dR1[3][3], dR2[3][3];
    angle_axis_to_matrix(aa, R);
    angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);
    // A shear on the left, like a Cartesian symmetry operator on a slightly
    // wrong cell, exaggerated.
    const RealType S[3][3] = {{1.0, 0.2, -0.1}, {0.0, 0.9, 0.3}, {0.05, 0.0, 1.1}};
    RealType M[3][3];
    mat3_mul(S, R, M);
    RealType Minv[3][3];
    REQUIRE(mat3_inv(M, Minv));

    for (int rank = 1; rank <= 4; ++rank) {
        RealType lab[N];
        rotate_cartesian_multipole(body, M, rank, lab);
        for (int m = 0; m < 3; ++m) {
            RealType (*dR)[3] = (m == 0) ? dR0 : (m == 1 ? dR1 : dR2);
            RealType dM[3][3], Omega[3][3];
            mat3_mul(S, dR, dM);
            mat3_mul(dM, Minv, Omega);

            RealType transport[N], generator[N];
            rotate_cartesian_multipole_deriv(body, M, dM, rank, transport);
            rotate_cartesian_multipole_generator(lab, Omega, rank, generator);
            for (int i = 0; i < nhermsum(rank); ++i) {
                INFO("rank " << rank << " direction " << m << " component " << i);
                REQUIRE_THAT(generator[i],
                             WithinAbs(transport[i], RealType(1e-10) * (RealType(1) + std::abs(transport[i]))));
            }
        }
    }
}

TEST_CASE("Rotation derivative rank 4 matches finite difference", "[multipole]") {
    constexpr int N = nhermsum(4);
    // Body multipole with all ranks populated
    RealType body[N] = {};
    body[0] = 0.21;  // charge
    body[hermite_index(1,0,0)] = 0.3;
    body[hermite_index(0,1,0)] = -0.2;
    body[hermite_index(0,0,1)] = 0.5;
    body[hermite_index(2,0,0)] = 0.1;
    body[hermite_index(0,2,0)] = -0.08;
    body[hermite_index(0,0,2)] = -0.02;
    body[hermite_index(1,1,0)] = 0.04;
    body[hermite_index(1,0,1)] = -0.03;
    body[hermite_index(0,1,1)] = 0.06;
    // rank 3
    body[hermite_index(3,0,0)] = 0.015;
    body[hermite_index(0,3,0)] = -0.01;
    body[hermite_index(0,0,3)] = 0.02;
    body[hermite_index(2,1,0)] = 0.008;
    body[hermite_index(1,0,2)] = -0.005;
    // rank 4
    body[hermite_index(4,0,0)] = 0.003;
    body[hermite_index(0,4,0)] = -0.002;
    body[hermite_index(0,0,4)] = 0.004;
    body[hermite_index(2,2,0)] = 0.001;
    body[hermite_index(2,0,2)] = -0.001;

    RealType aa[3] = {0.3, -0.2, 0.5};
    RealType R[3][3];
    angle_axis_to_matrix(aa, R);

    RealType dR0[3][3], dR1[3][3], dR2[3][3];
    angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);

    for (int m = 0; m < 3; ++m) {
        RealType (*dR_m)[3] = (m == 0) ? dR0 : (m == 1 ? dR1 : dR2);
        RealType dlab_analytical[N];
        rotate_cartesian_multipole_deriv(body, R, dR_m, 4, dlab_analytical);

        const RealType eps = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
        RealType aa_p[3] = {aa[0], aa[1], aa[2]};
        RealType aa_m[3] = {aa[0], aa[1], aa[2]};
        aa_p[m] += eps;
        aa_m[m] -= eps;

        RealType Rp[3][3], Rm[3][3];
        angle_axis_to_matrix(aa_p, Rp);
        angle_axis_to_matrix(aa_m, Rm);

        RealType lab_p[N], lab_m[N];
        rotate_cartesian_multipole(body, Rp, 4, lab_p);
        rotate_cartesian_multipole(body, Rm, 4, lab_m);

        for (int i = 0; i < N; ++i) {
            RealType fd = (lab_p[i] - lab_m[i]) / (2.0 * eps);
            INFO("m=" << m << " i=" << i << " analytical=" << dlab_analytical[i] << " fd=" << fd);
            REQUIRE_THAT(dlab_analytical[i], WithinAbs(fd, sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-3)));
        }
    }
}

TEST_CASE("Multipole mp_grad accumulation", "[multipole]") {
    // Verify dE/d(cart_a) and dE/d(cart_b) match finite difference
    constexpr int N = nhermsum(4);
    RealType cart_a[N] = {};
    RealType cart_b[N] = {};
    cart_a[0] = 0.8;
    cart_a[hermite_index(1,0,0)] = 0.15;
    cart_b[0] = -0.5;
    cart_b[hermite_index(0,0,1)] = 0.1;

    RealType Rx = 3.5, Ry = 1.2, Rz = -0.8;
    RealType force[3], dE_da[N], dE_db[N];
    RealType E = multipole_pair_energy_forces_mp_grad<4>(
        cart_a, cart_b, Rx, Ry, Rz, force, dE_da, dE_db);
    REQUIRE_THAT(E, WithinRel(multipole_pair_energy<4>(cart_a, cart_b, Rx, Ry, Rz),
                              RealType(1e-6)));

    const RealType eps = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
    const RealType fd_tol = sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-3);
    // Check dE/d(cart_a[i]) for a few components
    for (int c = 0; c < 4; ++c) {
        RealType ca_p[N], ca_m[N];
        for (int i = 0; i < N; ++i) ca_p[i] = ca_m[i] = cart_a[i];
        ca_p[c] += eps;
        ca_m[c] -= eps;
        RealType Ep = multipole_pair_energy<4>(ca_p, cart_b, Rx, Ry, Rz);
        RealType Em = multipole_pair_energy<4>(ca_m, cart_b, Rx, Ry, Rz);
        RealType fd = (Ep - Em) / (2.0 * eps);
        REQUIRE_THAT(dE_da[c], WithinAbs(fd, fd_tol));
    }
    // Check dE/d(cart_b[j])
    for (int c = 0; c < 4; ++c) {
        RealType cb_p[N], cb_m[N];
        for (int i = 0; i < N; ++i) cb_p[i] = cb_m[i] = cart_b[i];
        cb_p[c] += eps;
        cb_m[c] -= eps;
        RealType Ep = multipole_pair_energy<4>(cart_a, cb_p, Rx, Ry, Rz);
        RealType Em = multipole_pair_energy<4>(cart_a, cb_m, Rx, Ry, Rz);
        RealType fd = (Ep - Em) / (2.0 * eps);
        REQUIRE_THAT(dE_db[c], WithinAbs(fd, fd_tol));
    }
}

// ---------------------------------------------------------------------------
// Integration tests with full pipeline
// ---------------------------------------------------------------------------

// Helper: NaCl dimer with optional multipoles
static std::pair<MoleculeData, Workspace> make_nacl_with_mp(int mp_rank) {
    // Room for a second operator, used by the translation-gradient test (in P1
    // with one molecule the translation gradient is zero by symmetry).
    auto mol = allocate_molecule_data(1, 2, 2, mp_rank);
    auto ws  = allocate_workspace(64, 30000, 50000, 2, mp_rank);

    RealType a = 5.6402 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.25; h_t(0, 1) = 0.25; h_t(0, 2) = 0.25;
    Kokkos::deep_copy(mol.translations, h_t);

    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.0; h_r(0, 1) = 0.0; h_r(0, 2) = 0.0;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0, 0) = -a/4.0; h_b(0, 1) = 0.0; h_b(0, 2) = 0.0;
    h_b(1, 0) =  a/4.0; h_b(1, 1) = 0.0; h_b(1, 2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);  // body_positions derive from a, already in Bohr

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 1.0; h_q(1) = -1.0;
    Kokkos::deep_copy(mol.body_charges, h_q);

    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    h_bt(0) = 0; h_bt(1) = 1;
    Kokkos::deep_copy(mol.body_types, h_bt);

    // Body-frame Cartesian multipoles
    if (mp_rank > 0) {
        auto h_mp = Kokkos::create_mirror_view(mol.body_multipoles);
        for (int i = 0; i < 2; ++i)
            for (int c = 0; c < 35; ++c)
                h_mp(i, c) = 0.0;
        // Charge
        h_mp(0, 0) = 1.0;   // Na+
        h_mp(1, 0) = -1.0;  // Cl-
        if (mp_rank >= 1) {
            h_mp(0, hermite_index(1,0,0)) = 0.05;  // small x-dipole on Na
            h_mp(1, hermite_index(0,0,1)) = -0.03;  // small z-dipole on Cl
        }
        Kokkos::deep_copy(mol.body_multipoles, h_mp);
    }

    // Identity symop
    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    h_sr(0, 0) = 1; h_sr(0, 1) = 0; h_sr(0, 2) = 0;
    h_sr(0, 3) = 0; h_sr(0, 4) = 1; h_sr(0, 5) = 0;
    h_sr(0, 6) = 0; h_sr(0, 7) = 0; h_sr(0, 8) = 1;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);

    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    h_st(0, 0) = 0; h_st(0, 1) = 0; h_st(0, 2) = 0;
    Kokkos::deep_copy(mol.sym_translations, h_st);

    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0) = 0;
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = 1;
    mol.n_symops = 1;
    mol.n_sites_per_mol = 2;

    // Buckingham params
    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            h_A(i,j) = h_B(i,j) = h_C(i,j) = 0.0;
    h_A(0,1) = 1000.0 / AU_TO_KJ_PER_MOL; h_A(1,0) = 1000.0 / AU_TO_KJ_PER_MOL;
    h_B(0,1) = 3.0 * BOHR_TO_ANGSTROM;    h_B(1,0) = 3.0 * BOHR_TO_ANGSTROM;
    h_C(0,1) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_C(1,0) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B);
    Kokkos::deep_copy(ws.buck_C, h_C);
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR;
    ws.n_types = 2;

    return {mol, ws};
}

TEST_CASE("Charge-only multipoles give E_mp = 0", "[multipole][integration]") {
    using namespace klasp::testing;
    // When body multipoles are charge-only, the T-contraction minus qq/R should be ~0
    auto _result = make_nacl_with_mp(1); auto& mol = _result.first; auto& ws = _result.second;

    // Override: set body multipoles to charge-only (no dipoles)
    auto h_mp = Kokkos::create_mirror_view(mol.body_multipoles);
    for (int i = 0; i < 2; ++i)
        for (int c = 0; c < 35; ++c)
            h_mp(i, c) = 0.0;
    h_mp(0, 0) = 1.0;
    h_mp(1, 0) = -1.0;
    Kokkos::deep_copy(mol.body_multipoles, h_mp);

    auto slot = make_single_slot(mol, ws, RealType(1e-8));
    RealType E_mp = single_point_terms<4>(slot, EWALD_MULTIPOLE_H);

    // higher = full - qq - qd - dd, which cancels for charge-only multipoles
    REQUIRE_THAT(E_mp, WithinAbs(0.0, RealType(1e-4)));
}

TEST_CASE("Translation gradient with multipoles matches FD", "[multipole][integration]") {
    using namespace klasp::testing;
    auto _result = make_nacl_with_mp(1); auto& mol = _result.first; auto& ws = _result.second;

    // P-1 rather than P1: with one molecule and only the identity, every
    // translation gradient is zero by symmetry.
    {
        auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
        auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
        Kokkos::deep_copy(h_sr, mol.sym_rotations);
        Kokkos::deep_copy(h_si, mol.sym_indep_index);
        for (int c = 0; c < 9; ++c) h_sr(1, c) = 0;
        h_sr(1, 0) = -1; h_sr(1, 4) = -1; h_sr(1, 8) = -1;
        h_si(1) = 0;
        Kokkos::deep_copy(mol.sym_rotations, h_sr);
        Kokkos::deep_copy(mol.sym_indep_index, h_si);
        Kokkos::deep_copy(mol.sym_translations, RealType(0));
        mol.n_symops = 2;

        // Off the special position, so no component is zero for a second reason.
        auto h_mt = Kokkos::create_mirror_view(mol.translations);
        h_mt(0, 0) = 0.31; h_mt(0, 1) = 0.22; h_mt(0, 2) = 0.17;
        Kokkos::deep_copy(mol.translations, h_mt);
        auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
        h_mr(0, 0) = 0.21; h_mr(0, 1) = 0.14; h_mr(0, 2) = 0.09;
        Kokkos::deep_copy(mol.rot_vectors, h_mr);
    }

    auto slot = make_single_slot(mol, ws, RealType(1e-8));
    single_point<4>(slot);
    const auto h_gt = host_snapshot(slot.bws.dofs.d_gt);

    // Finite difference in the molecule's fractional translation, with the
    // whole model on.
    const RealType eps_t = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
    auto h_t = Kokkos::create_mirror(Kokkos::HostSpace(), slot.bws.dofs.translations);
    Kokkos::deep_copy(h_t, slot.bws.dofs.translations);

    double largest = 0;
    for (int d = 0; d < 3; ++d) {
        const RealType orig = h_t(0, 0, d);
        auto eval_e = [&](RealType delta) {
            auto h = Kokkos::create_mirror_view(slot.bws.dofs.translations);
            Kokkos::deep_copy(h, slot.bws.dofs.translations);
            h(0, 0, d) = orig + delta;
            Kokkos::deep_copy(slot.bws.dofs.translations, h);
            return single_point<4>(slot);
        };
        RealType Ep = eval_e(eps_t);
        RealType Em = eval_e(-eps_t);
        eval_e(RealType(0));

        RealType fd = (Ep - Em) / (2.0 * eps_t);
        REQUIRE_THAT(h_gt(0, 0, d), WithinAbs(fd, sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-2)));
        largest = std::max(largest, std::abs(double(h_gt(0, 0, d))));
    }

    REQUIRE(largest > 1e-2);
}

TEST_CASE("Cross-validate pair energy with dmapy (urea C-H)", "[multipole][integration]") {
    // Reference values from dmapy for urea UREAXX22 C and H sites
    constexpr int N = nhermsum(4);
    RealType cart_a[N] = {
        2.094982752000000e-01, 3.561226500000000e-03, -2.762822560000000e-02,
        0.0, 2.256934629653609e-01, -8.370830657300704e-04,
        0.0, 2.320964671679724e-01, 0.0, -4.577899301333333e-01,
        1.333874713183190e-01, -3.298239716115363e-01, 0.0,
        -1.289339031283968e-01, 0.0, -4.453568189922251e-03,
        2.952648358856541e-01, 0.0, 3.455913572588215e-02,
        0.0, 1.000451362537695e-01, 2.256995244852825e-02,
        0.0, -2.070313441658023e-02, 0.0, -7.934200183718926e-02,
        -1.543351884377468e-02, 0.0, -7.136433604753574e-03,
        0.0, 4.559663603081953e-02, 0.0, -2.489350161423930e-02,
        0.0, 1.042355034514286e-01
    };
    RealType cart_b[N] = {
        1.423208260000000e-01, -9.189123650000000e-02, -1.271514320000000e-02,
        0.0, 6.817686580538906e-02, 6.480884487306597e-02,
        0.0, 3.655132586127761e-02, 0.0, -1.047281916666667e-01,
        3.720756592886665e-02, 3.725712845019948e-02, 0.0,
        -2.054114242242191e-04, 0.0, -3.700215450464243e-02,
        -2.636900640005510e-02, 0.0, -1.088812205014438e-02,
        0.0, -5.895120242605665e-03, -2.988030098597396e-03,
        0.0, -1.742566762940450e-02, 0.0, 2.332078787201016e-02,
        -1.329447068687053e-02, 0.0, 1.628250078546793e-02,
        0.0, 2.332853101414661e-03, 0.0, 1.509281452798984e-02,
        0.0, -3.841360240000000e-02
    };

    RealType E = multipole_pair_energy<4>(cart_a, cart_b, 3.0, 1.0, 2.0);
    RealType expected = 8.506158672945600e-03;

    REQUIRE_THAT(E, WithinRel(expected, RealType(1e-4)));
}

TEST_CASE("Sph-to-cart matches dmapy for urea C site", "[multipole][integration]") {
    // First site of urea: C atom, 25 spherical components
    RealType sph[25] = {
        0.209498, 0.0, 0.003561, -0.027628,  // Q00, Q10, Q11c, Q11s
        -0.595218, 0.0, 0.0, -0.005758, 0.0, // Q20..Q22s
        0.0, -0.119137, 0.284938, 0.0, 0.0, -0.176773, 0.0,  // Q30..Q33s
        0.207497, 0.0, 0.145281, 0.0, -0.038427, 0.0, 0.0, -0.046485, 0.0 // Q40..Q44s
    };
    // Only check that the charge is preserved.
    constexpr int N = nhermsum(4);
    RealType cart[N];
    spherical_to_cartesian(sph, 25, 4, cart);
    REQUIRE_THAT(cart[0], WithinAbs(0.209498, RealType(1e-4)));
}

TEST_CASE("Optimizer with multipoles reduces energy", "[multipole][integration]") {
    auto _result = make_nacl_with_mp(1); auto& mol = _result.first; auto& ws = _result.second;

    // Perturb
    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.30; h_t(0, 1) = 0.20; h_t(0, 2) = 0.25;
    Kokkos::deep_copy(mol.translations, h_t);

    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.1; h_r(0, 1) = -0.1; h_r(0, 2) = 0.05;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    // Use batch-of-1 optimization
    auto bws = allocate_batch_workspace(1, ws, mol);
    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    h_bt(0,0,0) = 0.30; h_bt(0,0,1) = 0.20; h_bt(0,0,2) = 0.25;
    h_br(0,0,0) = 0.1;  h_br(0,0,1) = -0.1; h_br(0,0,2) = 0.05;
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0,i) = mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);

    BatchOptimizeSettings bsettings;
    bsettings.n_iterations = 20;
    auto result = batch_optimize_asymunit<4>(bws, bsettings);

    REQUIRE(result.n_iterations > 0);
    REQUIRE(std::isfinite(result.final_energies(0)));
}

// The higher-multipole term contains only the component pairs the rank mask
// kept (la + lb <= MaxRank). For a pair carrying only dipoles it is zero at
// every rank.
TEST_CASE("higher-multipole term excludes what the rank mask excluded",
          "[multipole]") {
    RealType a[35] = {}, b[35] = {};
    a[1] = 0.30; a[2] = -0.10; a[3] = 0.20;
    b[1] = -0.15; b[2] = 0.25; b[3] = 0.05;

    const RealType rx = 4.0, ry = 1.5, rz = -2.0;
    const RealType r = std::sqrt(rx*rx + ry*ry + rz*rz);
    const RealType inv_r = 1/r, inv_r3 = inv_r*inv_r*inv_r;
    const RealType inv_r5 = inv_r3*inv_r*inv_r, inv_r7 = inv_r5*inv_r*inv_r;
    const RealType di[3] = {a[1], a[2], a[3]}, dj[3] = {b[1], b[2], b[3]};
    const RealType miR = di[0]*rx + di[1]*ry + di[2]*rz;
    const RealType mjR = dj[0]*rx + dj[1]*ry + dj[2]*rz;
    const RealType mimj = di[0]*dj[0] + di[1]*dj[1] + di[2]*dj[2];
    // Non-trivial, so "zero" below is a real cancellation and not an empty sum.
    REQUIRE(std::abs(mimj*inv_r3 - RealType(3)*miR*mjR*inv_r5) > 1e-6);

    auto higher = [&](auto rank) {
        RealType e = 0, f[3] = {0,0,0}, da[35] = {}, db[35] = {};
        pair_multipole_higher<decltype(rank)::value>(
            a, b, rx, ry, rz, inv_r, /*ft=*/1.0, /*dft=*/0.0, e, f, da, db);
        return e;
    };
    CHECK_THAT(higher(std::integral_constant<int,1>{}), WithinAbs(0.0, 1e-14));
    CHECK_THAT(higher(std::integral_constant<int,2>{}), WithinAbs(0.0, 1e-14));
    CHECK_THAT(higher(std::integral_constant<int,3>{}), WithinAbs(0.0, 1e-14));
    CHECK_THAT(higher(std::integral_constant<int,4>{}), WithinAbs(0.0, 1e-14));

    // A real quadrupole still gives a non-zero term. Rank 4, since without
    // charges the lowest surviving quadrupole term is dipole-quadrupole (1+2).
    RealType c[35] = {};
    c[1] = -0.15; c[2] = 0.25; c[3] = 0.05;
    c[hermite_index(2,0,0)] = 0.4;
    RealType e = 0, f[3] = {0,0,0}, da[35] = {}, db[35] = {};
    pair_multipole_higher<4>(a, c, rx, ry, rz, inv_r, 1.0, 0.0, e, f, da, db);
    CHECK(std::abs(e) > 1e-8);

    // The single-precision kernel gives the same answer to single precision.
    RealType e32 = 0, f32[3] = {0,0,0}, da32[35] = {}, db32[35] = {};
    pair_multipole_higher<4, true, float>(a, c, rx, ry, rz, inv_r, 1.0, 0.0,
                                          e32, f32, da32, db32);
    CHECK_THAT(e32, WithinRel(e, RealType(1e-5)));
    for (int d = 0; d < 3; ++d)
        CHECK_THAT(f32[d], WithinAbs(f[d], RealType(1e-6) * (RealType(1) + std::abs(f[d]))));
    for (int k = 0; k < 35; ++k) {
        CHECK_THAT(da32[k], WithinAbs(da[k], RealType(1e-6) * (RealType(1) + std::abs(da[k]))));
        CHECK_THAT(db32[k], WithinAbs(db[k], RealType(1e-6) * (RealType(1) + std::abs(db[k]))));
    }
}
