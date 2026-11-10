#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "klasp/sobol.hpp"
#include "klasp/kgf.hpp"
#include "klasp/rotation.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/csp.hpp"

using namespace klasp;
using Catch::Matchers::WithinAbs;

// Reference values from csp-jax reference_data.json
TEST_CASE("Sobol matches csp-jax reference", "[csp]") {
    SECTION("seed=1 D=3") {
        double pts[3];
        sobol_vector(1, 3, pts);
        REQUIRE_THAT(pts[0], WithinAbs(0.0, 1e-15));
        REQUIRE_THAT(pts[1], WithinAbs(0.0, 1e-15));
        REQUIRE_THAT(pts[2], WithinAbs(0.0, 1e-15));
    }

    SECTION("seed=2 D=3") {
        double pts[3];
        sobol_vector(2, 3, pts);
        REQUIRE_THAT(pts[0], WithinAbs(0.5, 1e-15));
        REQUIRE_THAT(pts[1], WithinAbs(0.5, 1e-15));
        REQUIRE_THAT(pts[2], WithinAbs(0.5, 1e-15));
    }

    SECTION("seed=5 D=6") {
        double ref[] = {0.375, 0.375, 0.625, 0.875, 0.375, 0.125};
        double pts[6];
        sobol_vector(5, 6, pts);
        for (int i = 0; i < 6; ++i)
            REQUIRE_THAT(pts[i], WithinAbs(ref[i], 1e-15));
    }

    SECTION("seed=10 D=10") {
        double ref[] = {0.6875, 0.8125, 0.4375, 0.9375, 0.0625,
                        0.8125, 0.9375, 0.4375, 0.4375, 0.8125};
        double pts[10];
        sobol_vector(10, 10, pts);
        for (int i = 0; i < 10; ++i)
            REQUIRE_THAT(pts[i], WithinAbs(ref[i], 1e-15));
    }

    SECTION("seed=100 D=3") {
        double ref[] = {0.2890625, 0.8828125, 0.6484375};
        double pts[3];
        sobol_vector(100, 3, pts);
        for (int i = 0; i < 3; ++i)
            REQUIRE_THAT(pts[i], WithinAbs(ref[i], 1e-15));
    }

    SECTION("seed=1000 D=10") {
        double ref[] = {0.1572265625, 0.9091796875, 0.0810546875,
                        0.9892578125, 0.9677734375, 0.8447265625,
                        0.8583984375, 0.7119140625, 0.8134765625,
                        0.6318359375};
        double pts[10];
        sobol_vector(1000, 10, pts);
        for (int i = 0; i < 10; ++i)
            REQUIRE_THAT(pts[i], WithinAbs(ref[i], 1e-15));
    }

    SECTION("seed=42 D=15") {
        double ref[] = {0.734375, 0.578125, 0.765625, 0.484375, 0.203125,
                        0.234375, 0.234375, 0.859375, 0.546875, 0.859375,
                        0.328125, 0.390625, 0.015625, 0.546875, 0.671875};
        double pts[15];
        sobol_vector(42, 15, pts);
        for (int i = 0; i < 15; ++i)
            REQUIRE_THAT(pts[i], WithinAbs(ref[i], 1e-15));
    }
}

TEST_CASE("Space group symops from gemmi", "[csp]") {
    SECTION("P-1 has identity and inversion") {
        auto& sg = get_space_group(2);
        REQUIRE(sg.n_symops == 2);
        // Identity
        int id = find_identity_symop(sg);
        REQUIRE_THAT(sg.symops[id].rot[0][0], WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(sg.symops[id].rot[1][1], WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(sg.symops[id].rot[2][2], WithinAbs(1.0, 1e-10));
        // Inversion
        int inv = 1 - id;
        REQUIRE_THAT(sg.symops[inv].rot[0][0], WithinAbs(-1.0, 1e-10));
        REQUIRE_THAT(sg.symops[inv].rot[1][1], WithinAbs(-1.0, 1e-10));
        REQUIRE_THAT(sg.symops[inv].rot[2][2], WithinAbs(-1.0, 1e-10));
    }

    SECTION("P21/c has screw axis") {
        auto& sg = get_space_group(14);
        REQUIRE(sg.n_symops == 4);
        // Find the (-x, y+1/2, -z) operation
        bool found = false;
        for (int i = 0; i < sg.n_symops; ++i) {
            if (std::abs(sg.symops[i].rot[0][0] + 1.0) < 1e-10 &&
                std::abs(sg.symops[i].rot[1][1] - 1.0) < 1e-10 &&
                std::abs(sg.symops[i].rot[2][2] + 1.0) < 1e-10 &&
                std::abs(sg.symops[i].trans[1] - 0.5) < 1e-10) {
                found = true;
            }
        }
        REQUIRE(found);
    }

    SECTION("R-3 uses rhombohedral setting") {
        auto& sg = get_space_group(148);
        REQUIRE(sg.lattice == RHOMBOHEDRAL);
        REQUIRE(sg.n_symops == 6);  // not 18 (hexagonal)
    }

    SECTION("R-3m uses rhombohedral setting") {
        auto& sg = get_space_group(166);
        REQUIRE(sg.lattice == RHOMBOHEDRAL);
        REQUIRE(sg.n_symops == 12);  // not 36 (hexagonal)
    }
}

TEST_CASE("vec2rot produces orthogonal matrix", "[csp]") {
    RealType v[3] = {0.3, 0.7, 0.2};
    RealType R[3][3];
    vec2rot(v, R);

    // R^T @ R = I
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            RealType dot = 0;
            for (int k = 0; k < 3; ++k)
                dot += R[k][i] * R[k][j];
            REQUIRE_THAT(dot, WithinAbs(i == j ? 1.0 : 0.0, RealType(1e-4)));
        }

    // det(R) = 1
    RealType det = R[0][0]*(R[1][1]*R[2][2] - R[1][2]*R[2][1])
               - R[0][1]*(R[1][0]*R[2][2] - R[1][2]*R[2][0])
               + R[0][2]*(R[1][0]*R[2][1] - R[1][1]*R[2][0]);
    REQUIRE_THAT(det, WithinAbs(1.0, RealType(1e-4)));
}

TEST_CASE("angle-axis roundtrip", "[csp]") {
    RealType v[3] = {0.5, 0.3, 0.8};
    RealType R[3][3];
    vec2rot(v, R);

    RealType aa[3];
    rotation_matrix_to_angle_axis(R, aa);

    RealType R2[3][3];
    angle_axis_to_matrix(aa, R2);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE_THAT(R2[i][j], WithinAbs(R[i][j], RealType(1e-4)));
}

TEST_CASE("Space group P-1 has 2 symops", "[csp]") {
    auto& sg = get_space_group(2);
    REQUIRE(sg.n_symops == 2);
    REQUIRE(sg.lattice == TRICLINIC);
}

TEST_CASE("Space group P21/c has 4 symops", "[csp]") {
    auto& sg = get_space_group(14);
    REQUIRE(sg.n_symops == 4);
    REQUIRE(sg.lattice == MONOCLINIC);
}

TEST_CASE("Space group P212121 has 4 symops", "[csp]") {
    auto& sg = get_space_group(19);
    REQUIRE(sg.n_symops == 4);
    REQUIRE(sg.lattice == ORTHORHOMBIC);
}


// The generator's quasi-random sequence, pinned against csp-jax. P-1 draws 12
// dimensions (3 translation, 3 rotation, 3 angles, 3 lengths), translation first.
TEST_CASE("the Sobol sequence matches csp-jax at seed 3", "[csp]") {
    double vector[12];
    sobol_vector(3, 12, vector);
    REQUIRE_THAT(vector[0], WithinAbs(0.75, RealType(1e-4)));
    REQUIRE_THAT(vector[1], WithinAbs(0.25, RealType(1e-4)));
    REQUIRE_THAT(vector[2], WithinAbs(0.25, RealType(1e-4)));
}
