// A tabulated pair potential must stand in for the closed form in energy,
// gradient and curvature: the end-to-end case relaxes the same structure under
// both models and compares the results.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/batch_kernels.hpp"
#include "klasp/interp.hpp"
#include "klasp/pair_model.hpp"
#include "klasp/potentials.hpp"
#include "kokkos_listener.hpp"
#include "test_support.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace klasp;
using namespace klasp::testing;

namespace {

// The fixture's own Buckingham, as a PotentialTable. One type; the ghost row is
// added by tabulate_potentials and stays empty.
PotentialTable fixture_table() {
    PotentialTable t;
    t.type_names = {"X"};
    t.element_of = {"C"};
    t.A = {60.0};
    t.B = {1.9};
    t.C = {40.0};
    return t;
}

// Pinned taper: the default window follows the Ewald cutoff, which moves with
// kappa, and the table must know its range in advance.
constexpr double kTaperOnBohr  = 6.0 * ANGSTROM_TO_BOHR;
constexpr double kTaperOffBohr = 8.0 * ANGSTROM_TO_BOHR;

TabulatedSpec fine_spec() {
    TabulatedSpec spec;
    spec.enabled = true;
    // Enough knots for the relaxed cell to match the closed form within 1e-5
    // Bohr; slope error scales with the cube of the knot spacing.
    spec.n_knots = 16000;
    spec.r_min = 0.5;
    spec.r_max = kTaperOffBohr;
    return spec;
}

BatchOptimizeSettings relax_settings(int iters) {
    BatchOptimizeSettings s;
    s.n_iterations = iters;
    s.grad_tol = 1e-8;
    s.store_diagnostics = true;
    s.model.taper_on = RealType(kTaperOnBohr);
    s.model.taper_off = RealType(kTaperOffBohr);
    return s;
}

// Install the tabulated model on a fixture's ModelTables, leaving Buckingham
// loaded; EnergyModelParams::pair_model selects between them.
std::shared_ptr<TabulatedPotentials> install_tables(Fixture& f, const TangToenniesDamping& damping) {
    auto tabs = tabulate_potentials(fixture_table(), damping, fine_spec(),
                                    kTaperOffBohr);
    REQUIRE(tabs);
    f.bws.tables.pair_table_coeffs = tabs->coeffs;
    f.bws.tables.pair_tables = tabs->splines;
    return tabs;
}

struct Landing {
    double energy;
    double dof[6];      // translation then rotation, for the one molecule
    double cell[9];
};

Landing relax(Fixture& f, PairModelKind kind, int iters) {
    BatchOptimizeSettings s = relax_settings(iters);
    s.model.pair_model = kind;
    const auto r = batch_optimize_asymunit<1>(f.bws, s);
    Landing out{};
    out.energy = r.final_energies(0);
    for (int d = 0; d < 3; ++d) {
        out.dof[d] = r.final_translations(0, 0, d);
        out.dof[3 + d] = r.final_rotations(0, 0, d);
    }
    auto h_cell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      f.bws.geom.cell);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out.cell[i * 3 + j] = h_cell(0, i, j);
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// The scalar layer: value, slope and curvature.
// ---------------------------------------------------------------------------
TEST_CASE("a tabulated pair model reproduces the closed form it tabulates",
          "[pair-model]") {
    const TangToenniesDamping damping{};   // the production default
    auto tabs = tabulate_potentials(fixture_table(), damping, fine_spec(),
                                    kTaperOffBohr);
    REQUIRE(tabs);
    REQUIRE(tabs->n_types == 2);            // one real type plus the ghost

    // The builder's own measure, taken at interval midpoints.
    std::printf("  builder-reported worst relative error = %.2e\n",
                tabs->worst_rel_error);
    REQUIRE(tabs->worst_rel_error < 1e-8);

    auto h_spl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                     tabs->splines);
    auto h_coeff = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       tabs->coeffs);
    RadialSpline<> spl = h_spl(0);
    spl.c = h_coeff.data();   // host coefficients for a host evaluation

    // Scale errors by the largest |V| on the grid, since V passes through zero.
    double vmax = 0, dmax = 0, d2max = 0;
    double e_err = 0, d_err = 0, d2_err = 0;
    const auto& t = fixture_table();
    const RealType rmin = buckingham_r_min(RealType(t.a(0, 0)), RealType(t.b(0, 0)),
                                           RealType(t.c(0, 0)));
    for (int k = 0; k <= 2000; ++k) {
        const double r = 0.7 + (kTaperOffBohr - 0.9) * k / 2000.0;
        RealType e_ref, d_ref, d2_ref;
        buckingham_untapered(RealType(r), RealType(1.0 / r), RealType(t.a(0, 0)),
                             RealType(t.b(0, 0)), RealType(t.c(0, 0)), damping,
                             rmin, e_ref, d_ref, &d2_ref);
        RealType e, d, d2;
        spl.eval(RealType(r), RealType(1.0 / r), e, d, d2);
        vmax = std::max(vmax, std::abs(double(e_ref)));
        dmax = std::max(dmax, std::abs(double(d_ref)));
        d2max = std::max(d2max, std::abs(double(d2_ref)));
        e_err = std::max(e_err, std::abs(double(e - e_ref)));
        d_err = std::max(d_err, std::abs(double(d - d_ref)));
        d2_err = std::max(d2_err, std::abs(double(d2 - d2_ref)));
    }
    std::printf("  value %.2e  slope %.2e  curvature %.2e (relative to peak)\n",
                e_err / vmax, d_err / dmax, d2_err / d2max);
    // The second derivative matters: seed_strain_hessian starts every trust
    // region from it.
    REQUIRE(e_err / vmax < 1e-8);
    REQUIRE(d_err / dmax < 1e-7);
    REQUIRE(d2_err / d2max < 1e-5);
}

// ---------------------------------------------------------------------------
// End to end: the same structure, relaxed under both models.
// ---------------------------------------------------------------------------
TEST_CASE("the tabulated and closed-form models relax to the same structure",
          "[pair-model]") {
    // P2_1/c, so the asymmetric-unit prefactor is 2.
    auto fb = make_fixture_p21c(6.2, 7.1, 8.3, 96.0);
    auto ft = make_fixture_p21c(6.2, 7.1, 8.3, 96.0);
    batch_set_strain_dofs(fb.bws.geom, fb.bws.dofs, static_cast<int>(MONOCLINIC));
    batch_set_strain_dofs(ft.bws.geom, ft.bws.dofs, static_cast<int>(MONOCLINIC));
    auto tabs = install_tables(ft, relax_settings(1).model.buck_damping);

    const Landing buck = relax(fb, PairModelKind::Buckingham, 200);
    const Landing tab  = relax(ft, PairModelKind::Tabulated, 200);

    // Both runs must have moved.
    REQUIRE(std::abs(buck.dof[0]) + std::abs(buck.dof[1]) > 1e-6);
    REQUIRE(std::isfinite(buck.energy));
    REQUIRE(std::isfinite(tab.energy));

    double dof_err = 0, cell_err = 0;
    for (int i = 0; i < 6; ++i)
        dof_err = std::max(dof_err, std::abs(buck.dof[i] - tab.dof[i]));
    for (int i = 0; i < 9; ++i)
        cell_err = std::max(cell_err, std::abs(buck.cell[i] - tab.cell[i]));
    std::printf("  E: %.12f vs %.12f (diff %.2e Hartree)\n",
                buck.energy, tab.energy, buck.energy - tab.energy);
    std::printf("  max DOF difference %.2e, max cell difference %.2e Bohr\n",
                dof_err, cell_err);

    // Bounds set by the table's resolution: far inside 1e-8 relative at 16000
    // knots, on a lattice energy of order 1e-2 Hartree.
    REQUIRE_THAT(tab.energy, Catch::Matchers::WithinAbs(buck.energy, 1e-9));
    REQUIRE(dof_err < 1e-5);
    REQUIRE(cell_err < 1e-5);
}

// ---------------------------------------------------------------------------
// The selector.
// ---------------------------------------------------------------------------
TEST_CASE("forcing a model that was never built is refused, not silently empty",
          "[pair-model]") {
    auto f = make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);

    // Auto with no tables is the closed form.
    EnergyModelParams m;
    REQUIRE(m.pair_model == PairModelKind::Auto);
    REQUIRE_FALSE(use_tabulated_pair_model(m, f.bws.tables));

    // Forcing a missing table would give every pair exactly zero.
    m.pair_model = PairModelKind::Tabulated;
    REQUIRE_THROWS(use_tabulated_pair_model(m, f.bws.tables));

    // Once built, Auto picks it up and Buckingham still forces the closed form.
    auto tabs = install_tables(f, TangToenniesDamping{});
    m.pair_model = PairModelKind::Auto;
    REQUIRE(use_tabulated_pair_model(m, f.bws.tables));
    m.pair_model = PairModelKind::Buckingham;
    REQUIRE_FALSE(use_tabulated_pair_model(m, f.bws.tables));
}
