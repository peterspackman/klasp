// The analytic strain second derivative, against finite differences.
//
// Molecules are rigid, so only the COM separation responds to strain; with the
// exact map cell = (I + eps) cell_ref the second derivative is closed form. The
// Voigt shear half-factors are the easy part to get wrong.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "test_support.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/sgdata.hpp"

#include <cmath>
#include <vector>
#include <cstdio>

using namespace klasp;
using klasp::testing::make_fixture;
using klasp::testing::make_fixture_p21c;
using klasp::testing::host_snapshot;
using klasp::testing::energy_at_strain;

namespace {

// The terms the analytic block covers: real-space Buckingham and the erfc
// charge-charge term. Compared at rank 0, since EWALD_PAIR_ERFC also carries
// dipole terms the block does not have.
constexpr uint32_t ANALYTIC_TERMS = EWALD_PAIR_ERFC | EWALD_BUCKINGHAM;

// Central second difference of the energy with respect to two strain components,
// through the real parameterisation (batch_update_cell_from_strain + expand).
double fd_second(testing::Fixture& f, int v, int w, double h) {
    auto E = [&](double sv, double sw) {
        RealType eps[6] = {0,0,0,0,0,0};
        eps[v] += static_cast<RealType>(sv);
        eps[w] += static_cast<RealType>(sw);
        return static_cast<double>(energy_at_strain<0>(f, eps));
    };
    if (v == w) return (E(h,0) - 2.0*E(0,0) + E(-h,0)) / (h*h);
    // mixed: [E(+,+) - E(+,-) - E(-,+) + E(-,-)] / 4h^2
    return (E(h,h) - E(h,-h) - E(-h,h) + E(-h,-h)) / (4.0*h*h);
}

// Returns the worst relative disagreement over the lower triangle.
double worst_hessian_mismatch(testing::Fixture& f, double h, bool verbose = true,
                              const std::vector<int>* only = nullptr) {
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));   // identity basis, 6 DOFs
    f.settings.model.wolf = false;                 // plain Ewald erfc, matching the kernel
    f.settings.model.ewald_components = ANALYTIC_TERMS;

    // The strain block needs com_positions, which the expansion fills.
    batch_expand_asymmetric_unit<0>(f.bws.tables, f.bws.geom, f.bws.dofs);
    batch_strain_hessian(f.bws.tables, f.bws.geom, f.bws.tr, f.settings);
    const auto H = host_snapshot(f.bws.tr.strain_hess);

    double worst = 0.0;
    if (verbose) std::printf("  %3s %3s %16s %16s %10s\n", "v", "w", "analytic", "fd", "rel");
    auto wanted = [&](int c) {
        if (!only) return true;
        for (int x : *only) if (x == c) return true;
        return false;
    };
    for (int v = 0; v < 6; ++v)
        for (int w = 0; w <= v; ++w) {
            if (!wanted(v) || !wanted(w)) continue;
            const double an = H(0, v, w);
            const double fd = fd_second(f, v, w, h);
            const double mag = std::max(std::abs(an), std::abs(fd));
            const double rel = mag < 1e-10 ? 0.0 : std::abs(an - fd) / mag;
            if (verbose)
                std::printf("  %3d %3d %16.8e %16.8e %10.2e\n", v, w, an, fd, rel);
            worst = std::max(worst, rel);
        }
    return worst;
}

}  // namespace

TEST_CASE("the analytic strain Hessian matches finite differences",
          "[strain][hessian]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    const double worst = worst_hessian_mismatch(f, 2e-4);
    std::printf("  worst relative disagreement: %.2e\n", worst);

    // The remaining disagreement is finite-difference truncation at h = 2e-4.
    // A triclinic P-1 fixture cannot see the n_symops/(2 n_indep) prefactor (it
    // is 1); the P2_1/c case below covers it.
    REQUIRE(worst < 1e-3);
}

TEST_CASE("the analytic strain Hessian is symmetric", "[strain][hessian]") {
    auto f = make_fixture(7.0, 7.0, 9.5, 90.0, 90.0, 90.0);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));
    f.settings.model.wolf = false;
    f.settings.model.ewald_components = ANALYTIC_TERMS;
    batch_expand_asymmetric_unit<0>(f.bws.tables, f.bws.geom, f.bws.dofs);
    batch_strain_hessian(f.bws.tables, f.bws.geom, f.bws.tr, f.settings);
    const auto H = host_snapshot(f.bws.tr.strain_hess);
    for (int v = 0; v < 6; ++v)
        for (int w = 0; w < 6; ++w)
            REQUIRE_THAT(H(0,v,w), Catch::Matchers::WithinAbs(H(0,w,v), 1e-12));
}

// The same check on P2_1/c, where the prefactor n_symops / (2 n_indep) is 2.
//
// Asserted on the symmetry-allowed components {xx, yy, zz, xz} only. The
// forbidden shears disagree by ~50% because symmetry copies are treated as
// frozen under a symmetry-breaking strain; batch_set_strain_dofs projects
// those rows out before use. The limitation is pinned below.
TEST_CASE("the analytic strain Hessian matches finite differences (P2_1/c)",
          "[strain][hessian]") {
    auto f = make_fixture_p21c(7.0, 8.2, 9.5, 97.0);
    REQUIRE(f.mol.n_symops == 4);
    REQUIRE(f.mol.n_indep == 1);          // so the prefactor is 2, not 1

    const std::vector<int> allowed{0, 1, 2, 4};   // monoclinic: xx, yy, zz, xz
    const double worst_allowed = worst_hessian_mismatch(f, 2e-4, false, &allowed);
    std::printf("  P2_1/c allowed-component worst disagreement: %.2e\n", worst_allowed);
    REQUIRE(worst_allowed < 2e-2);

    // And the limitation itself, so it cannot silently get worse.
    auto g = make_fixture_p21c(7.0, 8.2, 9.5, 97.0);
    const std::vector<int> forbidden{3, 5};
    const double worst_forbidden = worst_hessian_mismatch(g, 2e-4, false, &forbidden);
    std::printf("  P2_1/c forbidden-component disagreement:     %.2e (frozen copies)\n",
                worst_forbidden);
    REQUIRE(worst_forbidden < 1.0);
}
