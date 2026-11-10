// The streaming optimiser and the stage-1 -> stage-2 handoff.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "test_support.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/batch_stream.hpp"
#include "klasp/sgdata.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace klasp;
using klasp::testing::make_fixture;
using klasp::testing::host_snapshot;
using Catch::Matchers::WithinAbs;

namespace {

BatchOptimizeSettings stream_settings(int iters) {
    BatchOptimizeSettings s;
    s.n_iterations = iters;
    s.grad_tol = 1e-6;
    s.store_diagnostics = true;
    s.model.taper_on = 0;
    s.model.taper_off = 0;
    s.refill_interval = 2;
    s.reuse_hessian = false;   // as stage 1 runs: nothing upstream to warm-start
    return s;
}

// Park `n` copies of the fixture's single structure in a pool, so any difference
// between harvested entries is the scheduler, not the physics.
StreamPool pool_of(testing::Fixture& f, int n) {
    StreamPool pool = allocate_stream_pool(n, f.bws.tables.n_indep);
    std::vector<int> seeds(1);
    for (int i = 0; i < n; ++i) {
        seeds[0] = 1000 + i;
        stream_pool_append(pool, f.bws, 1, seeds);
    }
    return pool;
}

}  // namespace

TEST_CASE("streaming and lockstep reach the same minimum", "[stream]") {
    // Same start, same tolerance, same step: the only difference is that
    // streaming stops iterating a slot once it is done.
    auto lock = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(lock.bws.geom, lock.bws.dofs, static_cast<int>(TRICLINIC));
    const auto rl = batch_optimize_asymunit<1>(lock.bws, stream_settings(120));
    const double E_lock = rl.final_energies(0);

    auto strm = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(strm.bws.geom, strm.bws.dofs, static_cast<int>(TRICLINIC));
    StreamPool pool = pool_of(strm, 1);
    const auto rs = batch_optimize_stream_tr<1>(strm.bws, pool, stream_settings(120));

    REQUIRE(rs.n == 1);
    const double E_strm = rs.energies(0);
    std::printf("  lockstep E = %.10f  streaming E = %.10f  diff = %.2e\n",
                E_lock, E_strm, std::abs(E_lock - E_strm));
    REQUIRE(std::isfinite(E_strm));
    // Same well, not bitwise agreement: step counts differ.
    REQUIRE_THAT(E_strm, WithinAbs(E_lock, 1e-6));
}

TEST_CASE("a slot refilled with no warm Hessian still converges", "[stream]") {
    // No Hessian in the pool (stage 1's situation). Three entries through one
    // slot means entries 2 and 3 arrive by refill.
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));
    StreamPool pool = pool_of(f, 3);

    const auto r = batch_optimize_stream_tr<1>(f.bws, pool, stream_settings(120));
    REQUIRE(r.n == 3);
    REQUIRE(r.diagnostics.valid());
    for (int i = 0; i < r.n; ++i) {
        const double g = r.diagnostics.grad_max(i);
        std::printf("  entry %d: |grad|_max = %.3e, converged at iter %d\n",
                    i, g, r.diagnostics.iters_to_converge(i));
        REQUIRE(std::isfinite(g));
        REQUIRE(g < 1e-4);
    }
    // Identical inputs: the refilled entries must land where the first one did.
    REQUIRE_THAT(r.energies(1), WithinAbs(r.energies(0), 1e-8));
    REQUIRE_THAT(r.energies(2), WithinAbs(r.energies(0), 1e-8));
}

TEST_CASE("stage 1 hands the relaxed structure to stage 2, not the starting one",
          "[stream]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));
    StreamPool raw = pool_of(f, 2);
    StreamPool out = allocate_stream_pool(2, f.bws.tables.n_indep);

    const auto start_cell = host_snapshot(raw.cells.cell);
    const auto start_tr = host_snapshot(raw.translations);

    const auto r = batch_optimize_stream_tr<1>(f.bws, raw, stream_settings(120), &out);
    REQUIRE(r.n == 2);
    REQUIRE(out.filled == 2);        // the pool, not the host, received them
    REQUIRE(r.energies.extent(0) == 0);   // and nothing crossed to the host

    const auto got_cell = host_snapshot(out.cells.cell);
    const auto got_tr = host_snapshot(out.translations);
    const auto got_H = host_snapshot(out.hess_mol);
    const auto got_seed = host_snapshot(out.seeds);
    const auto got_grad = host_snapshot(out.s1_grad);

    for (int e = 0; e < 2; ++e) {
        // Relaxed, i.e. moved from the pool entry.
        double moved = 0;
        for (int k = 0; k < f.bws.tables.n_indep; ++k)
            for (int d = 0; d < 3; ++d)
                moved = std::max(moved, std::abs(got_tr(e,k,d) - start_tr(e,k,d)));
        std::printf("  entry %d: |dtranslation| = %.3e, |grad| = %.3e, seed = %d\n",
                    e, moved, got_grad(e), got_seed(e));
        REQUIRE(moved > 1e-6);

        // A usable cell: right-handed and not collapsed, so stage 2 can start.
        double det = 0;
        {
            const auto& c = got_cell;
            det = c(e,0,0)*(c(e,1,1)*c(e,2,2) - c(e,1,2)*c(e,2,1))
                - c(e,0,1)*(c(e,1,0)*c(e,2,2) - c(e,1,2)*c(e,2,0))
                + c(e,0,2)*(c(e,1,0)*c(e,2,1) - c(e,1,1)*c(e,2,0));
        }
        REQUIRE(det > 0.0);
        REQUIRE(std::abs(det) > 1e-3);

        // Seeds travel with the structure, or results attach to the wrong crystal.
        REQUIRE(got_seed(e) == 1000 + e);
        // Convergence travels too, so the stage-2 row can report it.
        REQUIRE(got_grad(e) >= 0.0);
        REQUIRE(got_grad(e) < 1e-4);

        // The warm-start Hessian block must be a real curvature estimate, not zeros.
        double hmax = 0;
        for (int i = 0; i < out.nmd; ++i) hmax = std::max(hmax, std::abs(got_H(e,i,i)));
        REQUIRE(hmax > 0.0);
    }

    // Identical inputs relax to identical outputs.
    for (int k = 0; k < f.bws.tables.n_indep; ++k)
        for (int d = 0; d < 3; ++d)
            REQUIRE_THAT(got_tr(1,k,d), WithinAbs(got_tr(0,k,d), 1e-8));
    REQUIRE(start_cell(0,0,0) == start_cell(1,0,0));
}

// Iterating past convergence must not wreck the curvature stage 2 warm-starts
// from: the L-BFGS model restarts to its seeded diagonal, and the early stop at
// grad_tol avoids even that.
TEST_CASE("the Hessian handed to stage 2 survives post-convergence iterations",
          "[stream][hessian]") {
    // grad_tol = 0 disables the convergence test and early stop, so the probe
    // burns the whole budget.
    auto probe = [](int iters, double grad_tol) {
        auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
        batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));
        auto s = stream_settings(iters);
        s.reuse_hessian = false;
        s.grad_tol = grad_tol;
        const auto r = batch_optimize_asymunit<1>(f.bws, s);
        const auto H = host_snapshot(f.bws.tr.tr_hessian_mol);
        const int nmd = f.bws.tables.n_indep * 6;
        double diag = 0, offdiag = 0;
        for (int i = 0; i < nmd; ++i)
            for (int j = 0; j < nmd; ++j)
                (i == j ? diag : offdiag) += std::abs(H(0,i,j));
        // How far from the identity the block sits: identity has mean diagonal 1
        // and zero off-diagonal, so a reset block scores ~1.0 / 0.0.
        const double mean_diag = diag / nmd;
        const double mean_off  = offdiag / (nmd*nmd - nmd);
        std::printf("  %3d iters (grad_tol %.0e): ran %d, converged at %d, "
                    "mean|diag| = %.4f, mean|offdiag| = %.4f\n",
                    iters, grad_tol, r.n_iterations,
                    r.diagnostics.valid() ? r.diagnostics.iters_to_converge(0) : -1,
                    mean_diag, mean_off);
        return std::pair<double,double>{mean_diag, mean_off};
    };

    // The fixture converges at ~55 iterations, so 150 is far past it.
    const auto at_convergence = probe(55, 1e-6);
    const auto unprotected    = probe(150, 0.0);
    const auto protected_run  = probe(150, 1e-6);

    // Ratios only: tr_hessian_mol holds unscaled curvature.
    REQUIRE(at_convergence.first > 0.0);
    REQUIRE(unprotected.first < 10.0 * at_convergence.first);
    std::printf("  iterating past convergence changes the curvature %.2fx\n",
                unprotected.first / at_convergence.first);

    // With grad_tol set, the same budget stops at convergence.
    REQUIRE(protected_run.first < 2.0 * at_convergence.first);
}
