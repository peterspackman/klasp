// A charge-only pass must not read the dipole array, which is never re-zeroed
// and may hold a previous rank>0 batch's values.
#include "kokkos_listener.hpp"
#include "klasp/batch_kernels.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace klasp;
using Catch::Matchers::WithinAbs;

namespace {

// Energy of one rank-0 (charge-only) evaluation on this fixture.
double charge_only_energy(testing::Fixture& f) {
    batch_expand_asymmetric_unit<0>(f.bws.tables, f.bws.geom, f.bws.dofs);
    batch_energy_gradient_asymunit<0>(f.bws.tables, f.bws.geom, f.bws.dofs, f.settings, 0);
    auto h_E = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.geom.energies);
    return double(h_E(0));
}

}  // namespace

TEST_CASE("a charge-only pass ignores whatever is left in the dipole array",
          "[batch][dipole]") {
    auto clean = testing::make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    const double e_clean = charge_only_energy(clean);
    REQUIRE(std::isfinite(e_clean));

    // The same structure, with the dipole array holding stale non-zero values.
    auto dirty = testing::make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    Kokkos::deep_copy(dirty.bws.geom.dipoles, RealType(0.37));
    const double e_dirty = charge_only_energy(dirty);

    REQUIRE_THAT(e_dirty, WithinAbs(e_clean, 1e-12));
}

TEST_CASE("a multipole pass does read the dipole array", "[batch][dipole]") {
    // The guard must not become "never read dipoles".
    auto f = testing::make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    batch_expand_asymmetric_unit<1>(f.bws.tables, f.bws.geom, f.bws.dofs);
    batch_energy_gradient_asymunit<1>(f.bws.tables, f.bws.geom, f.bws.dofs, f.settings, 0);
    auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.geom.energies);
    const double e_rank1 = double(h1(0));

    auto g = testing::make_fixture(6.0, 6.8, 7.4, 88.0, 95.0, 103.0);
    const double e_rank0 = charge_only_energy(g);

    REQUIRE(std::isfinite(e_rank1));
    // The molecule in the fixture carries dipoles, so the two ranks must disagree.
    REQUIRE(std::abs(e_rank1 - e_rank0) > 1e-9);
}
