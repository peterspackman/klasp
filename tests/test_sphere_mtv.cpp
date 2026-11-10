#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "kokkos_listener.hpp"
#include "klasp/sphere_mtv.hpp"
#include "hull/kokkos_sphereclash.hpp"

using namespace klasp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct Cloud {
    std::vector<double> c, r;  // c interleaved x,y,z; r one per sphere
    size_t n() const { return r.size(); }
};

Cloud randomCloud(size_t n, double R, double rmin, double rmax, unsigned seed,
                  double sx = 0, double sy = 0, double sz = 0) {
    Cloud cl;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1, 1), Ur(rmin, rmax);
    for (size_t i = 0; i < n; ++i) {
        double x, y, z;
        do { x = U(rng); y = U(rng); z = U(rng); } while (x*x + y*y + z*z > 1);
        cl.c.insert(cl.c.end(), {x*R + sx, y*R + sy, z*R + sz});
        cl.r.push_back(Ur(rng));
    }
    return cl;
}

// Reference MTV computed directly.
MTVResult mtv(const Cloud& A, const Cloud& B) {
    return sphere_set_mtv(A.c.data(), A.r.data(), (int)A.n(),
                          B.c.data(), B.r.data(), (int)B.n());
}

double refDepth(const Cloud& A, const Cloud& B) {
    kokkos_quickhull::KokkosSphereClash<double> clash;
    auto dA = clash.upload(A.c.data(), A.r.data(), A.n());
    auto dB = clash.upload(B.c.data(), B.r.data(), B.n());
    return clash.hullMTV(dA, dB).signedDepth;
}

Cloud shifted(const Cloud& b, const RealType d[3], double s) {
    Cloud out = b;
    for (size_t j = 0; j < out.n(); ++j) {
        out.c[3*j+0] += s * d[0];
        out.c[3*j+1] += s * d[1];
        out.c[3*j+2] += s * d[2];
    }
    return out;
}

}  // namespace

TEST_CASE("sphere_set_mtv: analytic two-sphere case", "[mtv]") {
    Cloud A, B;
    A.c = {0, 0, 0};       A.r = {2.0};
    B.c = {1, 2, 2};       B.r = {1.5};  // |p|=3, R=3.5 -> depth 0.5 along p^
    auto m = mtv(A, B);
    REQUIRE_THAT(m.depth, WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(m.dir[0], WithinAbs(1.0/3.0, 1e-7));
    REQUIRE_THAT(m.dir[1], WithinAbs(2.0/3.0, 1e-7));
    REQUIRE_THAT(m.dir[2], WithinAbs(2.0/3.0, 1e-7));
}

TEST_CASE("sphere_set_mtv: overlapping clouds match exact hull MTV", "[mtv]") {
    Cloud A = randomCloud(40, 6, 0.5, 2.0, 1);
    Cloud B = randomCloud(30, 6, 0.5, 2.0, 2, 4, 1, -1);  // overlapping shift
    const double ref = refDepth(A, B);
    REQUIRE(ref > 0);  // they overlap
    auto m = mtv(A, B);
    // Support-min must agree closely with the exact convex MTV.
    REQUIRE_THAT(m.depth, WithinAbs(ref, 1e-3 + 5e-3 * std::abs(ref)));
    // Shifting B by depth*dir leaves negligible overlap (support-min is accurate to ~1e-4 A).
    Cloud Bs = shifted(B, m.dir, m.depth);
    auto m2 = mtv(A, Bs);
    REQUIRE(m2.depth < 1e-3);
}

TEST_CASE("sphere_set_mtv: separated clouds give clearance", "[mtv]") {
    Cloud A = randomCloud(30, 5, 0.5, 1.5, 3);
    Cloud B = randomCloud(30, 5, 0.5, 1.5, 4, 30, 0, 0);  // far apart
    const double ref = refDepth(A, B);
    REQUIRE(ref < 0);  // clearance
    auto m = mtv(A, B);
    REQUIRE_THAT(m.depth, WithinAbs(ref, 1e-3 + 5e-3 * std::abs(ref)));
    // Shifting B by depth*dir (depth < 0) brings them to touching.
    Cloud Bs = shifted(B, m.dir, m.depth);
    auto m2 = mtv(A, Bs);
    REQUIRE_THAT(m2.depth, WithinAbs(0.0, 1e-3));
}

TEST_CASE("sphere_set_mtv: equal radii reduce to point-hull distance", "[mtv]") {
    // Two single-point sets: exact distance minus radii.
    Cloud A, B;
    A.c = {0, 0, 0}; A.r = {1.0};
    B.c = {5, 0, 0}; B.r = {1.0};
    auto m = mtv(A, B);
    REQUIRE_THAT(m.depth, WithinAbs(2.0 - 5.0, 1e-9));  // -3 clearance
}
