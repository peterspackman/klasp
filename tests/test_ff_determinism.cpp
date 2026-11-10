// The same geometry must give bit-identical energies and gradients regardless
// of batch company: K identical copies in one batch must agree with each other,
// and one structure must agree between a narrow and a wide batch.
#include "kokkos_listener.hpp"
#include "klasp/batch_kernels.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace klasp;

namespace {

// One structure, replicated into every slot of an n-wide batch. 24 sites over
// two symmetry operations (48 atoms), so per-force accumulation order matters.
constexpr int kSites = 24;

testing::Fixture wide_fixture(int n) {
    auto f = testing::make_fixture_wide(kSites, 6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    // Re-allocate at the requested width and copy slot 0's state into all of it,
    // through the same setup calls as the fixture.
    auto bws = allocate_batch_workspace(n, f.ws, f.mol);

    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    auto h_t  = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_r  = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto s_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.geom.cell_params);
    auto s_t  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.translations);
    auto s_r  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.rot_vectors);
    for (int b = 0; b < n; ++b) {
        for (int j = 0; j < 6; ++j) h_cp(b, j) = s_cp(0, j);
        for (int k = 0; k < f.mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_t(b, k, d) = s_t(0, k, d);
                h_r(b, k, d) = s_r(0, k, d);
            }
    }
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    Kokkos::deep_copy(bws.dofs.translations, h_t);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_r);

    // Copy the potentials across, then the same cell/strain setup the fixture does.
    Kokkos::deep_copy(bws.tables.buck_A, f.bws.tables.buck_A);
    Kokkos::deep_copy(bws.tables.buck_B, f.bws.tables.buck_B);
    Kokkos::deep_copy(bws.tables.buck_C, f.bws.tables.buck_C);
    Kokkos::deep_copy(bws.tables.buck_Rmin, f.bws.tables.buck_Rmin);
    batch_setup_cells(bws.tables, bws.geom);
    bws.dofs.cell_ref   = decltype(bws.dofs.cell_ref)("cell_ref", n, 3, 3);
    bws.dofs.strain     = decltype(bws.dofs.strain)("strain", n, 6);
    bws.dofs.d_g_strain = decltype(bws.dofs.d_g_strain)("d_g_strain", n, 6);
    batch_set_strain_dofs(bws.geom, bws.dofs, static_cast<int>(TRICLINIC));
    Kokkos::deep_copy(bws.dofs.cell_ref, bws.geom.cell);
    Kokkos::deep_copy(bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(bws.dofs.d_g_strain, RealType(0));

    f.bws = bws;
    return f;
}

// Everything one evaluation produces, for one slot, as raw bits.
struct Slot {
    double energy;
    std::vector<double> gt, gr, gs;
};

Slot evaluate(testing::Fixture& f, int slot) {
    batch_expand_asymmetric_unit<1>(f.bws.tables, f.bws.geom, f.bws.dofs);
    batch_energy_gradient_asymunit<1>(f.bws.tables, f.bws.geom, f.bws.dofs, f.settings, 0);
    auto hE  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.geom.energies);
    auto hgt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_gt);
    auto hgr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_gr);
    auto hgs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_g_strain);
    Slot s;
    s.energy = double(hE(slot));
    for (int k = 0; k < f.mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            s.gt.push_back(double(hgt(slot, k, d)));
            s.gr.push_back(double(hgr(slot, k, d)));
        }
    for (int j = 0; j < 6; ++j) s.gs.push_back(double(hgs(slot, j)));
    return s;
}

// Bitwise: last-bit differences become kJ/mol after relaxation.
void require_identical(const Slot& a, const Slot& b, const char* what) {
    INFO(what);
    REQUIRE(a.energy == b.energy);
    REQUIRE(a.gt == b.gt);
    REQUIRE(a.gr == b.gr);
    REQUIRE(a.gs == b.gs);
}

}  // namespace

TEST_CASE("identical structures in one batch get identical answers",
          "[determinism][batch]") {
    constexpr int K = 32;
    auto f = wide_fixture(K);
    const Slot ref = evaluate(f, 0);
    REQUIRE(std::isfinite(ref.energy));
    REQUIRE(ref.energy != 0.0);          // the fixture is doing real work

    for (int b = 1; b < K; ++b)
        require_identical(ref, evaluate(f, b), "slot differs from slot 0");
}

TEST_CASE("a structure's answer does not depend on how wide its batch is",
          "[determinism][batch]") {
    auto narrow = wide_fixture(8);
    auto wide   = wide_fixture(512);
    require_identical(evaluate(narrow, 0), evaluate(wide, 0),
                      "batch of 8 vs batch of 512");
}

TEST_CASE("repeating an evaluation gives the same answer", "[determinism][batch]") {
    // Run-to-run order dependence within a single evaluation.
    auto f = wide_fixture(64);
    const Slot first = evaluate(f, 0);
    for (int i = 0; i < 3; ++i)
        require_identical(first, evaluate(f, 0), "repeat evaluation");
}
