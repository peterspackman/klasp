#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/cell.hpp"
#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Cubic cell is diagonal", "[cell]") {
    RealType params[6] = {5.0, 5.0, 5.0, 90.0, 90.0, 90.0};
    RealType cell[3][3];
    cell_params_to_matrix(params, cell);

    REQUIRE_THAT(cell[0][0], WithinAbs(5.0, RealType(1e-4)));
    REQUIRE_THAT(cell[1][1], WithinAbs(5.0, RealType(1e-4)));
    REQUIRE_THAT(cell[2][2], WithinAbs(5.0, RealType(1e-4)));

    // Off-diagonal should be zero
    REQUIRE_THAT(cell[0][1], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(cell[0][2], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(cell[1][0], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(cell[1][2], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(cell[2][0], WithinAbs(0.0, RealType(1e-4)));
    REQUIRE_THAT(cell[2][1], WithinAbs(0.0, RealType(1e-4)));
}

TEST_CASE("Orthorhombic cell", "[cell]") {
    RealType params[6] = {3.0, 4.0, 5.0, 90.0, 90.0, 90.0};
    RealType cell[3][3];
    cell_params_to_matrix(params, cell);

    REQUIRE_THAT(cell[0][0], WithinAbs(3.0, RealType(1e-4)));
    REQUIRE_THAT(cell[1][1], WithinAbs(4.0, RealType(1e-4)));
    REQUIRE_THAT(cell[2][2], WithinAbs(5.0, RealType(1e-4)));
}

TEST_CASE("Cell volume cubic", "[cell]") {
    RealType params[6] = {5.0, 5.0, 5.0, 90.0, 90.0, 90.0};
    RealType cell[3][3];
    cell_params_to_matrix(params, cell);

    REQUIRE_THAT(cell_volume(cell), WithinAbs(125.0, RealType(1e-4)));
}

TEST_CASE("Cell volume monoclinic", "[cell]") {
    RealType params[6] = {4.0, 5.0, 6.0, 90.0, 100.0, 90.0};
    RealType cell[3][3];
    cell_params_to_matrix(params, cell);

    // V = a*b*c*sin(beta)
    RealType expected = 4.0 * 5.0 * 6.0 * std::sin(100.0 * M_PI / 180.0);
    REQUIRE_THAT(cell_volume(cell), WithinRel(expected, RealType(1e-4)));
}

TEST_CASE("Invert 3x3 roundtrip", "[cell]") {
    RealType params[6] = {5.0, 6.0, 7.0, 80.0, 95.0, 100.0};
    RealType cell[3][3], inv[3][3];
    cell_params_to_matrix(params, cell);
    REQUIRE(mat3_inv(cell, inv));

    // cell * inv should be identity
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            RealType sum = 0.0;
            for (int k = 0; k < 3; ++k)
                sum += cell[i][k] * inv[k][j];
            RealType expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(sum, WithinAbs(expected, RealType(1e-4)));
        }
    }
}

TEST_CASE("Fill shifts produces correct count", "[cell]") {
    RealType params[6] = {5.0, 5.0, 5.0, 90.0, 90.0, 90.0};
    RealType cell[3][3];
    cell_params_to_matrix(params, cell);

    auto ws = allocate_workspace(8, 20000, 1, 1);
    int n = fill_shifts(cell, 10.0, ws);

    // Cubic 5 A cell, cutoff 10 A: ceil(10/5)+1 = 3, so (2*3+1)^3 = 343
    REQUIRE(n == 343);
    REQUIRE(ws.n_shifts == 343);
}
