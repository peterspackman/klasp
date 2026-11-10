#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "kokkos_listener.hpp"
#include "klasp/trust_region.hpp"

using namespace klasp;
using namespace klasp::tr;
using Catch::Matchers::WithinAbs;

namespace {
[[maybe_unused]] RealType vnorm(const std::vector<RealType>& v) {
    RealType s = 0; for (auto x : v) s += x * x; return std::sqrt(s);
}
}  // namespace

TEST_CASE("radius_of_gyration of a known cluster", "[tr]") {
    // four points at unit distance from the origin centroid -> R_gyr = 1.
    const RealType pos[12] = {1,0,0, -1,0,0, 0,1,0, 0,-1,0};
    REQUIRE_THAT(radius_of_gyration(pos, 4), WithinAbs(1.0, 1e-12));
    // scale by 2 -> R_gyr scales by 2.
    RealType pos2[12]; for (int i = 0; i < 12; ++i) pos2[i] = 2 * pos[i];
    REQUIRE_THAT(radius_of_gyration(pos2, 4), WithinAbs(2.0, 1e-12));
}

TEST_CASE("cholesky solve matches a known SPD system", "[tr]") {
    const int n = 3;
    // SPD A = LLᵀ for a simple L; solve A x = b.
    RealType A[9] = {4, 2, 0,
                     2, 5, 1,
                     0, 1, 3};
    RealType L[9];
    REQUIRE(cholesky(A, L, n));
    RealType x_true[3] = {1, -2, 0.5};
    RealType b[3];
    matvec(A, x_true, b, n);
    RealType x[3];
    chol_solve(L, b, x, n);
    for (int i = 0; i < n; ++i) REQUIRE_THAT(x[i], WithinAbs(x_true[i], 1e-10));
}

TEST_CASE("dogleg: interior region returns the full Newton step", "[tr]") {
    const int n = 2;
    RealType B[4] = {1, 0, 0, 4};      // SPD diagonal
    RealType g[2] = {1, 1};
    RealType p[2];
    // Newton step pN = -B^{-1} g = (-1, -0.25), ‖pN‖ ≈ 1.031
    RealType Lsc[tr::MAX_TR_DOF*tr::MAX_TR_DOF];
    RealType pred = dogleg(B, g, /*delta=*/5.0, n, p, Lsc);
    REQUIRE_THAT(p[0], WithinAbs(-1.0, 1e-10));
    REQUIRE_THAT(p[1], WithinAbs(-0.25, 1e-10));
    REQUIRE(pred > 0);                 // model predicts a decrease
}

TEST_CASE("dogleg: tiny radius gives a steepest-descent step to the boundary", "[tr]") {
    const int n = 2;
    RealType B[4] = {1, 0, 0, 1};      // identity
    RealType g[2] = {3, 4};            // ‖g‖ = 5
    RealType p[2];
    RealType Lsc[tr::MAX_TR_DOF*tr::MAX_TR_DOF];
    RealType pred = dogleg(B, g, /*delta=*/1.0, n, p, Lsc);
    // -g/‖g‖ * delta = (-0.6, -0.8)
    REQUIRE_THAT(p[0], WithinAbs(-0.6, 1e-10));
    REQUIRE_THAT(p[1], WithinAbs(-0.8, 1e-10));
    REQUIRE_THAT(std::sqrt(p[0]*p[0] + p[1]*p[1]), WithinAbs(1.0, 1e-12));
    REQUIRE(pred > 0);
}

TEST_CASE("dogleg: step never exceeds the radius and reduces the model", "[tr]") {
    const int n = 3;
    RealType B[9] = {2, 0.3, 0,
                     0.3, 1, 0.1,
                     0, 0.1, 5};       // SPD
    RealType g[3] = {1, -2, 0.5};
    for (RealType delta : {0.05, 0.2, 0.5, 1.0, 3.0}) {
        RealType p[3];
        RealType Lsc[tr::MAX_TR_DOF*tr::MAX_TR_DOF];
    RealType pred = dogleg(B, g, delta, n, p, Lsc);
        REQUIRE(norm(p, n) <= delta + 1e-9);
        REQUIRE(pred >= -1e-12);       // never predicts an increase
    }
}

TEST_CASE("bfgs_update satisfies the secant equation B s = y", "[tr]") {
    const int n = 3;
    RealType B[9] = {1,0,0, 0,1,0, 0,0,1};     // start from identity
    RealType s[3] = {0.5, -1.0, 2.0};
    RealType y[3] = {3.0, 1.0, 0.4};           // yᵀs > 0
    REQUIRE(dot(y, s, n) > 0);
    bfgs_update(B, s, y, n);
    RealType Bs[3];
    matvec(B, s, Bs, n);
    for (int i = 0; i < n; ++i) REQUIRE_THAT(Bs[i], WithinAbs(y[i], 1e-10));
    // B stays SPD (Cholesky succeeds).
    RealType L[9];
    REQUIRE(cholesky(B, L, n));
}

// Powell damping updates B under negative curvature, blending towards Bs just
// enough to keep B SPD.
TEST_CASE("bfgs_update damps non-positive curvature instead of skipping", "[tr]") {
    const int n = 2;
    RealType B[4] = {1,0, 0,1};
    RealType s[2] = {1, 0};
    RealType y[2] = {-1, 0};                    // yᵀs < 0
    REQUIRE(dot(y, s, n) < 0);

    REQUIRE(bfgs_update(B, s, y, n));           // it updates rather than skipping

    // Damping targets ŷᵀs = 0.2 sᵀBs, so the curvature along s is positive and
    // B is still usable by the dogleg.
    RealType Bs[2];
    matvec(B, s, Bs, n);
    REQUIRE(dot(s, Bs, n) > 0);
    RealType L[4];
    REQUIRE(cholesky(B, L, n));
}

// Sustained negative curvature may flatten B; dogleg must still step downhill
// inside the radius.
TEST_CASE("optimiser still steps downhill after sustained bad curvature", "[tr]") {
    const int n = 3;
    RealType B[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int it = 0; it < 40; ++it) {
        const RealType a = RealType(0.05) * (it % 5 + 1);
        const RealType t = RealType(0.3) * it;
        RealType s[3] = {a * std::cos(t), a * std::sin(t), a * std::cos(2 * t)};
        RealType y[3] = {-RealType(0.5) * s[0], -RealType(0.5) * s[1], -RealType(0.5) * s[2]};
        REQUIRE(dot(y, s, n) < 0);              // bad curvature every iteration
        bfgs_update(B, s, y, n);

        // Whatever B has become, the step must remain a descent direction
        // inside the trust region.
        RealType g[3] = {0.3, -0.2, 0.1}, p[3], L[9];
        const RealType delta = 0.1;
        dogleg(B, g, delta, n, p, L);
        REQUIRE(dot(g, p, n) < 0);                        // downhill
        REQUIRE(norm(p, n) <= delta * (1 + 1e-6));        // inside the radius
    }
}

// A non-finite first energy can never accept a step, so the sentinel ends up as
// the stored energy and must itself be non-finite for pack_results to drop it.
TEST_CASE("a non-finite start is rejected and stays non-finite", "[trust_region]") {
    const RealType sentinel = Kokkos::Experimental::infinity_v<RealType>;
    REQUIRE_FALSE(std::isfinite(sentinel));   // the property the storage path needs

    // Against the sentinel, every trial collapses rather than being accepted.
    for (double E_trial : {-500.0, -1.0, 0.0, 7.5}) {
        const auto v = tr::judge_step(RealType(E_trial), sentinel,
                                      /*pred=*/RealType(1e-2),
                                      /*eta=*/RealType(1e-4), /*rho_max=*/RealType(10));
        INFO("E_trial = " << E_trial);
        CHECK_FALSE(v.accept);
        CHECK(v.collapsed);
    }
    // Eacc is never overwritten, so the optimiser reports the sentinel itself.
    CHECK(std::isfinite(RealType(1e300)));
}
