// Finite-difference checks on the force field's strain (cell) gradient. The
// trust region needs the gradient to match the energy, or steps are rejected and
// the structure stalls.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "test_support.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/sgdata.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace klasp;

namespace {
using klasp::testing::Fixture;
using klasp::testing::make_fixture;
using klasp::testing::energy_at_strain;

// Central-difference dE/d(eps_v) against the analytic d_g_strain, for every
// Voigt component. Returns the worst relative disagreement.
template <int RANK>
double worst_strain_mismatch(Fixture& f, double h, bool verbose = true) {
    RealType eps0[6] = {0, 0, 0, 0, 0, 0};
    energy_at_strain<RANK>(f, eps0);
    // Snapshot by value: on a host build create_mirror_view aliases the view,
    // and later evaluations in this loop would overwrite it.
    RealType an_g[6];
    {
        auto h_g = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       f.bws.dofs.d_g_strain);
        for (int v = 0; v < 6; ++v) an_g[v] = h_g(0, v);
    }

    double worst = 0.0;
    for (int v = 0; v < 6; ++v) {
        RealType ep[6] = {0, 0, 0, 0, 0, 0}, em[6] = {0, 0, 0, 0, 0, 0};
        ep[v] = static_cast<RealType>(h);
        em[v] = static_cast<RealType>(-h);
        const double Ep = energy_at_strain<RANK>(f, ep);
        const double Em = energy_at_strain<RANK>(f, em);
        const double fd = (Ep - Em) / (2.0 * h);
        const double an = an_g[v];
        // Absolute floor, so a component that is zero by symmetry is not judged by
        // a ratio against FD noise.
        const double scale = std::max({std::abs(an), std::abs(fd), 1e-10});
        const double rel = klasp::testing::fd_agrees(an, fd)
                         ? 0.0 : std::abs(an - fd) / scale;
        if (verbose)
            std::printf("    eps_%d: analytic=% .8e  fd=% .8e  rel=%.2e\n", v, an, fd, rel);
        worst = std::max(worst, rel);
    }
    return worst;
}


// As above, but without re-anchoring: both sides are taken with respect to the
// cumulative strain from cell_ref, the quantity the optimiser varies.
template <int RANK>
double worst_cumulative_mismatch(Fixture& f, const RealType base[6], double h,
                                 bool verbose = true) {
    energy_at_strain<RANK>(f, base);
    RealType an_g[6];
    {
        auto h_g = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       f.bws.dofs.d_g_strain);
        for (int v = 0; v < 6; ++v) an_g[v] = h_g(0, v);
    }
    double worst = 0.0;
    for (int v = 0; v < 6; ++v) {
        RealType ep[6], em[6];
        for (int i = 0; i < 6; ++i) { ep[i] = base[i]; em[i] = base[i]; }
        ep[v] += static_cast<RealType>(h);
        em[v] -= static_cast<RealType>(h);
        const double Ep = energy_at_strain<RANK>(f, ep);
        const double Em = energy_at_strain<RANK>(f, em);
        const double fd = (Ep - Em) / (2.0 * h);
        const double an = an_g[v];
        // Absolute floor, as above.
        const double scale = std::max({std::abs(an), std::abs(fd), 1e-10});
        const double rel = klasp::testing::fd_agrees(an, fd)
                         ? 0.0 : std::abs(an - fd) / scale;
        if (verbose)
            std::printf("    eps_%d: analytic=% .8e  fd=% .8e  rel=%.2e\n", v, an, fd, rel);
        worst = std::max(worst, rel);
    }
    return worst;
}

}  // namespace

TEST_CASE("FF strain gradient matches finite difference (charges only)",
          "[gradients][strain]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    std::printf("  charge-only (rank 0), triclinic:\n");
    const double worst = worst_strain_mismatch<0>(f, 1e-5);
    REQUIRE(worst < 1e-6);
}

TEST_CASE("FF strain gradient matches finite difference (multipoles)",
          "[gradients][strain]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    std::printf("  multipole (rank 1), triclinic:\n");
    const double worst = worst_strain_mismatch<1>(f, 1e-5);
    REQUIRE(worst < 1e-6);
}

// Full Ewald at a deformed working point: exercises the cumulative-strain fold
// and the Dual6 reciprocal strain gradient (including the k-sphere window),
// which Wolf DSF never touches.
TEST_CASE("FF strain gradient matches finite difference (full Ewald)",
          "[gradients][strain]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    f.settings.model.wolf = false;
    std::printf("  multipole (rank 1), triclinic, Ewald reciprocal:\n");
    REQUIRE(worst_strain_mismatch<1>(f, 1e-5) < 1e-6);
}

TEST_CASE("FF strain gradient matches finite difference at nonzero strain",
          "[gradients][strain]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);

    // Move to a deformed cell and make THAT the working point.
    RealType pre[6] = {0.03, -0.02, 0.015, 0.01, -0.008, 0.012};
    energy_at_strain<1>(f, pre);
    Kokkos::deep_copy(f.bws.dofs.cell_ref, f.bws.geom.cell);
    Kokkos::deep_copy(f.bws.dofs.strain, RealType(0));

    std::printf("  multipole (rank 1) at a deformed reference cell:\n");
    const double worst = worst_strain_mismatch<1>(f, 1e-5);
    REQUIRE(worst < 1e-6);
}

// The rotation (angle-axis) gradient against finite differences, including near
// |r| -> pi where angle-axis is singular.
namespace {
template <int RANK>
double worst_rotation_mismatch(Fixture& f, const RealType rot[3], double h) {
    auto set_rot = [&](const RealType r[3]) {
        auto h_br = Kokkos::create_mirror_view(f.bws.dofs.rot_vectors);
        Kokkos::deep_copy(h_br, f.bws.dofs.rot_vectors);
        for (int d = 0; d < 3; ++d) h_br(0, 0, d) = r[d];
        Kokkos::deep_copy(f.bws.dofs.rot_vectors, h_br);
    };
    auto energy = [&]() -> double {
        batch_expand_asymmetric_unit<RANK>(f.bws.tables, f.bws.geom, f.bws.dofs);
        Kokkos::fence();
        batch_energy_gradient_asymunit<RANK>(f.bws.tables, f.bws.geom, f.bws.dofs, f.settings, 0);
        Kokkos::fence();
        auto hE = Kokkos::create_mirror_view(f.bws.geom.energies);
        Kokkos::deep_copy(hE, f.bws.geom.energies);
        return hE(0);
    };

    set_rot(rot);
    energy();
    RealType an_gr[3];
    {
        auto h_gr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_gr);
        for (int d = 0; d < 3; ++d) an_gr[d] = h_gr(0, 0, d);
    }

    double worst = 0.0;
    for (int d = 0; d < 3; ++d) {
        RealType rp[3] = {rot[0], rot[1], rot[2]};
        RealType rm[3] = {rot[0], rot[1], rot[2]};
        rp[d] += static_cast<RealType>(h);
        rm[d] -= static_cast<RealType>(h);
        set_rot(rp); const double Ep = energy();
        set_rot(rm); const double Em = energy();
        const double fd = (Ep - Em) / (2.0 * h);
        const double an = an_gr[d];
        // Absolute floor, as above.
        const double scale = std::max({std::abs(an), std::abs(fd), 1e-10});
        const double rel = klasp::testing::fd_agrees(an, fd)
                         ? 0.0 : std::abs(an - fd) / scale;
        std::printf("    rot_%d: analytic=% .8e  fd=% .8e  rel=%.2e\n", d, an, fd, rel);
        worst = std::max(worst, rel);
    }
    set_rot(rot);
    return worst;
}
}  // namespace

TEST_CASE("FF rotation gradient matches finite difference (small rotation)",
          "[gradients][rotation]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    const RealType r[3] = {0.21, 0.14, 0.09};       // |r| ~ 0.27 rad
    std::printf("  rank 1, |r| = 0.27 rad:\n");
    REQUIRE(worst_rotation_mismatch<1>(f, r, 1e-5) < 1e-6);
}

TEST_CASE("FF rotation gradient matches finite difference near the angle-axis pole",
          "[gradients][rotation]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    // |r| = 3.0 rad, just inside pi, where the angle-axis Jacobian is worst
    // conditioned; uniform orientation sampling puts many structures here.
    const RealType r[3] = {1.8, 2.1, 0.9};
    std::printf("  rank 1, |r| = %.2f rad (near pi):\n",
                std::sqrt(1.8 * 1.8 + 2.1 * 2.1 + 0.9 * 0.9));
    REQUIRE(worst_rotation_mismatch<1>(f, r, 1e-5) < 1e-6);
}

namespace {
// Largest second difference of E(strain) along a fine scan. A discontinuity
// from a pair crossing a hard cutoff stands out far above the curvature scale.
double roughness(Fixture& f, int N = 1200, double lo = -0.05, double hi = 0.05) {
    std::vector<double> E(N);
    for (int i = 0; i < N; ++i) {
        const double x = lo + (hi - lo) * i / (N - 1);
        RealType eps[6] = {static_cast<RealType>(x), 0, 0, 0, 0, 0};
        E[i] = energy_at_strain<1>(f, eps);
    }
    double worst = 0.0;
    for (int i = 2; i < N; ++i)
        worst = std::max(worst, std::abs((E[i] - E[i-1]) - (E[i-1] - E[i-2])));
    return worst;
}
}  // namespace

// The optimiser needs a smooth objective: a step in the energy makes the
// trust-region ratio noise near convergence. For smooth f the second difference
// scales as h^2, so quartering h must divide it by ~16; a floor that does not
// move with h is a step (Ewald real-space cutoff, reciprocal sphere, pair taper).
TEST_CASE("energy under strain is smooth, not stepped",
          "[gradients][strain][smooth]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    f.settings.model.wolf = false;                       // production path: Ewald

    const int N1 = 601, N2 = 2401;                 // h and h/4
    const double r1 = roughness(f, N1);
    const double r2 = roughness(f, N2);
    const double p = std::log(r1 / r2) / std::log(4.0);
    std::printf("  second difference: %.3e at N=%d, %.3e at N=%d -> h^%.2f\n",
                r1, N1, r2, N2, p);

    REQUIRE(p > 1.5);                              // h^2 is smooth, h^0 is a step
    REQUIRE(r2 < 1e-9);                            // and the grain is small
}

// The strain gradient at the cell the optimiser is at, not only at its
// reference. Isotropic deformation cannot see a misordered fold (F^-T is then
// scalar), so it is the control and must pass at a larger magnitude.
TEST_CASE("FF strain gradient at nonzero cumulative strain (isotropic control)",
          "[gradients][strain]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    RealType iso[6] = {0.12, 0.12, 0.12, 0, 0, 0};
    std::printf("  isotropic cumulative strain 0.12 (control):\n");
    REQUIRE(worst_cumulative_mismatch<1>(f, iso, 1e-5) < 1e-6);
}

// Anisotropic deformations (monoclinic, triclinic), which the symmetric
// nonzero-strain tests in test_symmetry_gradient.cpp cannot see.
TEST_CASE("FF strain gradient at nonzero cumulative strain (anisotropic)",
          "[gradients][strain]") {
    SECTION("diagonal, no shear") {
        auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
        RealType d[6] = {0.03, -0.02, 0.015, 0, 0, 0};
        std::printf("  diagonal anisotropic cumulative strain:\n");
        REQUIRE(worst_cumulative_mismatch<1>(f, d, 1e-5) < 1e-6);
    }
    SECTION("monoclinic-allowed, with a beta shear") {
        auto f = make_fixture(7.0, 8.2, 9.5, 90.0, 97.0, 90.0);
        RealType m[6] = {0.03, -0.02, 0.015, 0, 0.02, 0};
        std::printf("  monoclinic cumulative strain with beta shear:\n");
        REQUIRE(worst_cumulative_mismatch<1>(f, m, 1e-5) < 1e-6);
    }
    SECTION("full triclinic") {
        auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
        RealType t[6] = {0.03, -0.02, 0.015, 0.01, -0.008, 0.012};
        std::printf("  triclinic cumulative strain:\n");
        REQUIRE(worst_cumulative_mismatch<1>(f, t, 1e-5) < 1e-6);
    }
}

// The fixture defaults to Wolf DSF, so this covers the reciprocal half of the
// cumulative-strain gradient (batch_recip_strain_ad), where the duals are seeded
// against the reference frame: dG/deps carries an F^-T, d(1/V)/deps a tr(F^-1 E_v).
TEST_CASE("Ewald reciprocal strain gradient at nonzero cumulative strain",
          "[gradients][strain]") {
    SECTION("isotropic control") {
        auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
        f.settings.model.wolf = false;
        RealType iso[6] = {0.12, 0.12, 0.12, 0, 0, 0};
        std::printf("  reciprocal, isotropic cumulative strain 0.12:\n");
        REQUIRE(worst_cumulative_mismatch<1>(f, iso, 1e-5) < 1e-6);
    }
    SECTION("diagonal anisotropic") {
        auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
        f.settings.model.wolf = false;
        RealType d[6] = {0.03, -0.02, 0.015, 0, 0, 0};
        std::printf("  reciprocal, diagonal anisotropic cumulative strain:\n");
        REQUIRE(worst_cumulative_mismatch<1>(f, d, 1e-5) < 1e-6);
    }
    SECTION("full triclinic") {
        auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
        f.settings.model.wolf = false;
        RealType t[6] = {0.03, -0.02, 0.015, 0.01, -0.008, 0.012};
        std::printf("  reciprocal, triclinic cumulative strain:\n");
        REQUIRE(worst_cumulative_mismatch<1>(f, t, 1e-5) < 1e-6);
    }
}
