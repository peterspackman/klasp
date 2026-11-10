#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/rotation.hpp"
#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Zero angle-axis gives identity", "[rotation]") {
    RealType aa[3] = {0.0, 0.0, 0.0};
    RealType R[3][3];
    angle_axis_to_matrix(aa, R);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            RealType expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(R[i][j], WithinAbs(expected, RealType(1e-4)));
        }
}

TEST_CASE("90 degrees about z", "[rotation]") {
    RealType aa[3] = {0.0, 0.0, M_PI / 2.0};
    RealType R[3][3];
    angle_axis_to_matrix(aa, R);

    // Expected: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    REQUIRE_THAT(R[0][0], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(R[0][1], WithinAbs(-1.0, RealType(1e-4)));
    REQUIRE_THAT(R[0][2], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(R[1][0], WithinAbs(1.0, RealType(1e-4)));
    REQUIRE_THAT(R[1][1], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(R[1][2], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(R[2][0], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(R[2][1], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(R[2][2], WithinAbs(1.0, RealType(1e-4)));
}

TEST_CASE("R * R^T = I for arbitrary angle-axis", "[rotation]") {
    RealType aa[3] = {0.3, -0.7, 1.2};
    RealType R[3][3], Rt[3][3], prod[3][3];

    angle_axis_to_matrix(aa, R);
    mat3_transpose(R, Rt);
    mat3_mul(R, Rt, prod);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            RealType expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(prod[i][j], WithinAbs(expected, RealType(1e-4)));
        }
}

TEST_CASE("Derivative matches finite difference", "[rotation]") {
    RealType aa[3] = {0.3, -0.7, 1.2};
    RealType dR0[3][3], dR1[3][3], dR2[3][3];
    angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);

    RealType* dR_all[3] = {&dR0[0][0], &dR1[0][0], &dR2[0][0]};

    const RealType eps = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
    const RealType fd_tol = sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-5);

    for (int m = 0; m < 3; ++m) {
        RealType aa_plus[3] = {aa[0], aa[1], aa[2]};
        RealType aa_minus[3] = {aa[0], aa[1], aa[2]};
        aa_plus[m] += eps;
        aa_minus[m] -= eps;

        RealType Rp[3][3], Rm[3][3];
        angle_axis_to_matrix(aa_plus, Rp);
        angle_axis_to_matrix(aa_minus, Rm);

        RealType (*dR_m)[3] = reinterpret_cast<RealType(*)[3]>(dR_all[m]);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                RealType fd = (Rp[i][j] - Rm[i][j]) / (2.0 * eps);
                REQUIRE_THAT(dR_m[i][j], WithinAbs(fd, fd_tol));
            }
    }
}

TEST_CASE("Derivative at zero matches finite difference", "[rotation]") {
    RealType aa[3] = {0.0, 0.0, 0.0};
    RealType dR0[3][3], dR1[3][3], dR2[3][3];
    angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);

    RealType* dR_all[3] = {&dR0[0][0], &dR1[0][0], &dR2[0][0]};

    const RealType eps = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
    const RealType fd_tol = sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-5);

    for (int m = 0; m < 3; ++m) {
        RealType aa_plus[3] = {0.0, 0.0, 0.0};
        RealType aa_minus[3] = {0.0, 0.0, 0.0};
        aa_plus[m] += eps;
        aa_minus[m] -= eps;

        RealType Rp[3][3], Rm[3][3];
        angle_axis_to_matrix(aa_plus, Rp);
        angle_axis_to_matrix(aa_minus, Rm);

        RealType (*dR_m)[3] = reinterpret_cast<RealType(*)[3]>(dR_all[m]);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                RealType fd = (Rp[i][j] - Rm[i][j]) / (2.0 * eps);
                REQUIRE_THAT(dR_m[i][j], WithinAbs(fd, fd_tol));
            }
    }
}

TEST_CASE("mat3_vec correctness", "[rotation]") {
    RealType R[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    RealType v[3] = {1, 0, 0};
    RealType out[3];
    mat3_vec(R, v, out);

    REQUIRE_THAT(out[0], WithinAbs(1.0, RealType(1e-4)));
    REQUIRE_THAT(out[1], WithinAbs(4.0, RealType(1e-4)));
    REQUIRE_THAT(out[2], WithinAbs(7.0, RealType(1e-4)));
}
