// The 3x3 helpers' contract (singular matrices, [3][3] vs [9] layouts) and the
// host/device marshalling helpers.
#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/mat3.hpp"
#include "klasp/types.hpp"

#include <array>
#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;

namespace {

// An arbitrary well-conditioned matrix, in both layouts.
constexpr double kM33[3][3] = {{2.0, -1.0, 0.5}, {0.25, 3.0, 1.5}, {-0.75, 0.5, 4.0}};
constexpr double kM9[9] = {2.0, -1.0, 0.5, 0.25, 3.0, 1.5, -0.75, 0.5, 4.0};

}  // namespace

TEST_CASE("the two layouts agree, term for term where they must", "[mat3]") {
    REQUIRE(mat3_det(kM33) == mat3_det(kM9));

    double a33[3][3], a9[9];
    REQUIRE(mat3_inv(kM33, a33));
    REQUIRE(mat3_inv(kM9, a9));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(a33[i][j] == a9[i * 3 + j]);  // bit-for-bit, not approximately

    // mat3_mul's [3][3] and [9] bodies agree only to rounding (different FMA
    // contraction), so callers keep the overload they use.
    double p33[3][3], p9[9];
    mat3_mul(kM33, a33, p33);
    mat3_mul(kM9, a9, p9);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE_THAT(p33[i][j], WithinAbs(p9[i * 3 + j], 1e-15));

    double t33[3][3], t9[9];
    mat3_transpose(kM33, t33);
    mat3_transpose(kM9, t9);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(t33[i][j] == t9[i * 3 + j]);
}

TEST_CASE("the inverse inverts", "[mat3]") {
    double inv[3][3], prod[3][3];
    REQUIRE(mat3_inv(kM33, inv));
    mat3_mul(kM33, inv, prod);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE_THAT(prod[i][j], WithinAbs((i == j) ? 1.0 : 0.0, 1e-12));
}

TEST_CASE("a singular matrix leaves the output alone", "[mat3]") {
    const double sing[3][3] = {{1.0, 0.0, 2.0}, {2.0, 1.0, 4.0}, {3.0, 1.0, 6.0}};
    double out[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i][j] = -7.0;  // a marker the call must not touch

    REQUIRE_FALSE(mat3_inv(sing, out));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) REQUIRE(out[i][j] == -7.0);

    const double sing9[9] = {1.0, 0.0, 2.0, 2.0, 1.0, 4.0, 3.0, 1.0, 6.0};
    double out9[9];
    for (int k = 0; k < 9; ++k) out9[k] = -7.0;
    REQUIRE_FALSE(mat3_inv(sing9, out9));
    for (int k = 0; k < 9; ++k) REQUIRE(out9[k] == -7.0);
}

TEST_CASE("the caller writes the fallback, so a zero fill still works", "[mat3]") {
    const double sing[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    double inv[3][3] = {};
    REQUIRE_FALSE(mat3_inv(sing, inv));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) REQUIRE(inv[i][j] == 0.0);
}

TEST_CASE("a NaN determinant is not singular", "[mat3]") {
    // Only an exact zero determinant counts as singular; callers guard NaN themselves.
    double nan_cell[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, std::nan("")}};
    double inv[3][3] = {};
    REQUIRE(mat3_inv(nan_cell, inv));
    REQUIRE(std::isnan(inv[2][2]));
}

TEST_CASE("float and double callers both work", "[mat3]") {
    const float m[3][3] = {{2.0f, 0.0f, 0.0f}, {0.0f, 4.0f, 0.0f}, {0.0f, 0.0f, 8.0f}};
    float inv[3][3];
    REQUIRE(mat3_inv(m, inv));
    REQUIRE(inv[0][0] == 0.5f);
    REQUIRE(inv[1][1] == 0.25f);
    REQUIRE(inv[2][2] == 0.125f);
    REQUIRE(mat3_det(m) == 64.0f);
}

TEST_CASE("to_view and to_host round-trip a vector", "[mat3][marshal]") {
    const std::vector<double> v{1.5, -2.5, 3.25, 4.0};
    auto d = to_view("round_trip", v);
    REQUIRE(d.extent(0) == v.size());
    REQUIRE(to_host_vector(d) == v);

    const std::vector<std::array<double, 3>> rows{{1, 2, 3}, {4, 5, 6}};
    auto d2 = to_view("round_trip_2d", rows);
    REQUIRE(d2.extent(0) == 2);
    REQUIRE(d2.extent(1) == 3);
    auto h2 = to_host(d2);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (int c = 0; c < 3; ++c) REQUIRE(h2(i, c) == rows[i][c]);
}

TEST_CASE("copy_to_view leaves the padded tail alone", "[mat3][marshal]") {
    // copy_to_view must copy a subview: the view's padded tail must stay untouched.
    View2D<double> v("padded", 4, 3);           // Kokkos zero-initialises
    const std::vector<std::array<double, 3>> rows{{1, 2, 3}, {4, 5, 6}};
    copy_to_view(v, rows);
    auto h = to_host(v);
    REQUIRE(h(1, 2) == 6.0);
    for (int c = 0; c < 3; ++c) {
        REQUIRE(h(2, c) == 0.0);
        REQUIRE(h(3, c) == 0.0);
    }

    View1D<int> w("padded1d", 3);
    copy_to_view(w, std::vector<int>{7, 8});
    auto hw = to_host(w);
    REQUIRE(hw(0) == 7);
    REQUIRE(hw(1) == 8);
    REQUIRE(hw(2) == 0);
}

TEST_CASE("copy_to_view refuses a source that does not fit", "[mat3][marshal]") {
    View1D<double> v("small", 2);
    REQUIRE_THROWS(copy_to_view(v, std::vector<double>{1, 2, 3}));

    View2D<double> v2("wrong_width", 2, 4);
    REQUIRE_THROWS(copy_to_view(v2, std::vector<std::array<double, 3>>{{1, 2, 3}}));
}
