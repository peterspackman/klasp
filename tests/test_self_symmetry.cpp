#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include "klasp/compack.hpp"

#include <cmath>
#include <vector>

using namespace klasp;

namespace {

int n_ops(const std::vector<double>& pos, const std::vector<int>& z, double tol,
          int* improper = nullptr) {
    RigidBody b = make_rigid_body(pos.data(), z.data(), static_cast<int>(z.size()),
                                  /*heavy_only=*/false);
    detect_self_symmetry(b, tol);
    if (improper) {
        *improper = 0;
        for (int o = 0; o < b.n_self_ops(); ++o) {
            const double* m = &b.self_ops[o * 9];
            const double d = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6])
                           + m[2]*(m[3]*m[7]-m[4]*m[6]);
            if (d < 0) ++(*improper);
        }
    }
    return b.n_self_ops();
}

std::vector<double> ideal_ammonia() {
    const double rh = 0.9377, zh = -0.3263, zn = 0.0704;
    std::vector<double> p = {0, 0, zn};
    for (int k = 0; k < 3; ++k) {
        const double a = 2.0 * M_PI * k / 3.0;
        p.push_back(rh * std::cos(a));
        p.push_back(rh * std::sin(a));
        p.push_back(zh);
    }
    return p;
}

std::vector<double> ideal_benzene() {
    std::vector<double> p;
    for (double r : {1.39, 2.48})
        for (int k = 0; k < 6; ++k) {
            const double a = 2.0 * M_PI * k / 6.0;
            p.push_back(r * std::cos(a));
            p.push_back(r * std::sin(a));
            p.push_back(0.0);
        }
    return p;
}

}  // namespace

TEST_CASE("a pyramidal molecule keeps its mirrors", "[compack][symmetry]") {
    // Ammonia is C3v. The reference triple is the hydrogen base, coplanar with its
    // own centroid, so the fit alone cannot decide the mirrors.
    const std::vector<int> z = {7, 1, 1, 1};
    int improper = 0;
    REQUIRE(n_ops(ideal_ammonia(), z, 0.05, &improper) == 6);
    REQUIRE(improper == 3);
}

TEST_CASE("a real ammonia geometry keeps its mirrors too", "[compack][symmetry]") {
    // The geometry a basis file carries, off the ideal one in the fifth decimal.
    const std::vector<double> real = {
        0.0000004222, -0.0000122437,  0.0704446530,
       -0.7983338918, -0.4929496307, -0.3263319239,
        0.8261647624, -0.4447434241, -0.3263323227,
       -0.0278367377,  0.9378631980, -0.3262601315};
    const std::vector<int> z = {7, 1, 1, 1};
    int improper = 0;
    REQUIRE(n_ops(real, z, 0.05, &improper) == 6);
    REQUIRE(improper == 3);
}

TEST_CASE("a planar ring keeps its full point group", "[compack][symmetry]") {
    // Benzene is D6h, 24 operations. Its reference triple is coplanar with the
    // centroid, and here the reflection in that plane is the molecular plane.
    std::vector<int> z(12, 6);
    for (int k = 6; k < 12; ++k) z[k] = 1;
    REQUIRE(n_ops(ideal_benzene(), z, 0.05) == 24);
}

TEST_CASE("an unsymmetric molecule gets only the identity", "[compack][symmetry]") {
    const std::vector<double> p = {
        0.00, 0.00, 0.00,
        1.43, 0.00, 0.00,
       -0.53, 1.31, 0.00,
       -0.41, -0.72, 1.09};
    const std::vector<int> z = {6, 8, 7, 1};
    REQUIRE(n_ops(p, z, 0.05) == 1);
}
