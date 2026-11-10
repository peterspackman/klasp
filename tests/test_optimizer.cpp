// End-to-end behaviour of the trust-region optimiser loop: convergence, the
// returned state matching the evaluated one, and lattice-type preservation.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "test_support.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/sgdata.hpp"

#include <cmath>
#include <cstdio>

using namespace klasp;
using klasp::testing::make_fixture;
using klasp::testing::host_snapshot;
using Catch::Matchers::WithinAbs;

namespace {

// a, b, c and the three angles of the current cell (lattice vectors are columns).
struct CellMetric { double a, b, c, alpha, beta, gamma; };

CellMetric metric_of(const BatchWorkspace& bws, int b) {
    auto h = host_snapshot(bws.geom.cell);
    auto col = [&](int j, double v[3]) { for (int i = 0; i < 3; ++i) v[i] = h(b, i, j); };
    auto nrm = [](const double v[3]) { return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); };
    auto ang = [&](const double u[3], const double v[3]) {
        return std::acos((u[0]*v[0]+u[1]*v[1]+u[2]*v[2]) / (nrm(u)*nrm(v))) * 180.0 / M_PI;
    };
    double A[3], B[3], C[3];
    col(0, A); col(1, B); col(2, C);
    return {nrm(A), nrm(B), nrm(C), ang(B, C), ang(A, C), ang(A, B)};
}

BatchOptimizeSettings tr_settings(int iters) {
    BatchOptimizeSettings s;
    s.n_iterations = iters;
    s.grad_tol = 1e-6;
    s.store_diagnostics = true;
    s.model.taper_on = 0;
    s.model.taper_off = 0;
    return s;
}

}  // namespace

TEST_CASE("trust region drives the gradient down from a perturbed start",
          "[optimiser]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));

    const auto r = batch_optimize_asymunit<1>(f.bws, tr_settings(60));
    REQUIRE(r.diagnostics.grad_max.extent(0) > 0);
    const double g = r.diagnostics.grad_max(0);
    std::printf("  final |grad|_max = %.3e (scaled, Hartree/Bohr)\n", g);
    REQUIRE(std::isfinite(g));
    REQUIRE(g < 1e-4);
}

// The workspace must hold the accepted geometry, not the last trial, since
// energies and descriptors are read from it.
TEST_CASE("the energy returned belongs to the geometry returned", "[optimiser]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));

    const auto r = batch_optimize_asymunit<1>(f.bws, tr_settings(40));
    const double reported = r.final_energies(0);

    // Re-evaluate at whatever geometry the workspace was left holding.
    BatchOptimizeSettings eo = tr_settings(1);
    eo.energy_only = true;
    batch_expand_asymmetric_unit<1>(f.bws.tables, f.bws.geom, f.bws.dofs);
    Kokkos::fence();
    batch_energy_gradient_asymunit<1>(f.bws.tables, f.bws.geom, f.bws.dofs, eo, 0);
    Kokkos::fence();
    const double recomputed = host_snapshot(f.bws.geom.energies)(0);

    std::printf("  reported=%.10f  recomputed=%.10f  diff=%.2e\n",
                reported, recomputed, reported - recomputed);
    REQUIRE_THAT(recomputed, WithinAbs(reported, 1e-9));
}

// A cell given a lattice type must still satisfy that type's metric constraints
// after optimising.
TEST_CASE("cell optimisation keeps the cell in its lattice type", "[optimiser]") {
    SECTION("tetragonal keeps a == b and all angles 90") {
        auto f = make_fixture(7.0, 7.0, 9.5, 90.0, 90.0, 90.0);
        batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TETRAGONAL));
        REQUIRE(f.bws.dofs.n_cell_dof == 2);

        const auto before = metric_of(f.bws, 0);
        batch_optimize_asymunit<1>(f.bws, tr_settings(40));
        const auto after = metric_of(f.bws, 0);
        std::printf("  a %.4f -> %.4f, b %.4f -> %.4f, c %.4f -> %.4f\n",
                    before.a, after.a, before.b, after.b, before.c, after.c);

        REQUIRE_THAT(after.a, WithinAbs(after.b, 1e-9));
        REQUIRE_THAT(after.alpha, WithinAbs(90.0, 1e-9));
        REQUIRE_THAT(after.beta,  WithinAbs(90.0, 1e-9));
        REQUIRE_THAT(after.gamma, WithinAbs(90.0, 1e-9));
        REQUIRE(std::abs(after.a - before.a) > 1e-6);   // it did move
    }

    SECTION("cubic keeps a == b == c") {
        auto f = make_fixture(8.0, 8.0, 8.0, 90.0, 90.0, 90.0);
        batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(CUBIC));
        REQUIRE(f.bws.dofs.n_cell_dof == 1);

        const auto before = metric_of(f.bws, 0);
        batch_optimize_asymunit<1>(f.bws, tr_settings(40));
        const auto after = metric_of(f.bws, 0);
        REQUIRE_THAT(after.a, WithinAbs(after.b, 1e-9));
        REQUIRE_THAT(after.b, WithinAbs(after.c, 1e-9));
        REQUIRE(std::abs(after.a - before.a) > 1e-6);
    }
}

// Angle-axis is singular at |r| = 2*pi, where the rotational gradient vanishes;
// the optimiser must bring |r| back.
TEST_CASE("rotations come back to the principal branch", "[optimiser]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));

    // Start well past a full turn: the same orientation as |r| - 2*pi, but in
    // the part of the parameterisation where the Jacobian is nearly singular.
    {
        auto h = host_snapshot(f.bws.dofs.rot_vectors);
        h(0,0,0) = 4.0; h(0,0,1) = 3.0; h(0,0,2) = 2.0;   // |r| = 5.39 rad
        Kokkos::deep_copy(f.bws.dofs.rot_vectors, h);
    }
    const auto r = batch_optimize_asymunit<1>(f.bws, tr_settings(40));

    double n = 0;
    for (int d = 0; d < 3; ++d) n += double(r.final_rotations(0,0,d)) * r.final_rotations(0,0,d);
    n = std::sqrt(n);
    std::printf("  |r| 5.385 -> %.4f rad\n", n);
    REQUIRE(n <= M_PI + 1e-9);
}

// A cell at the strain bound must not stall on zero-length steps.
TEST_CASE("a cell pushed to the strain bound keeps making progress",
          "[optimiser]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));

    // Park the reference cell far from where the energy wants it, so the first
    // steps run hard into the bound.
    {
        auto h = host_snapshot(f.bws.dofs.strain);
        for (int v = 0; v < 3; ++v) h(0, v) = RealType(0.39);   // just inside 0.4
        Kokkos::deep_copy(f.bws.dofs.strain, h);
        batch_update_cell_from_strain(make_force_field(f.bws.tables, f.settings.model),
                                      f.bws.geom, f.bws.dofs);
    }
    const auto before = metric_of(f.bws, 0);
    const auto r = batch_optimize_asymunit<1>(f.bws, tr_settings(60));
    const auto after = metric_of(f.bws, 0);
    std::printf("  a %.4f -> %.4f, final |grad|_max = %.3e, restarts = %d\n",
                before.a, after.a, r.diagnostics.grad_max(0), r.diagnostics.restarts(0));

    REQUIRE(std::isfinite(r.final_energies(0)));
    REQUIRE(std::abs(after.a - before.a) > 1e-3);   // it got past the bound
}
