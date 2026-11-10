#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "test_support.hpp"
#include "klasp/batch.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/molecule.hpp"
#include <cmath>

using namespace klasp;
using namespace klasp::testing;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Finite-difference checks on the force the production kernel produces, term by
// term. Rock salt as eight single-atom molecules: each rigid-body translation is
// one ion's position, so the force is -d_gt / a on a cubic cell.

static const RealType kNaClA = RealType(5.6402) * ANGSTROM_TO_BOHR;

// dE/d(fractional translation) -> Cartesian force on the ion.
static RealType force_from_gradient(RealType d_gt) { return -d_gt / kNaClA; }

// Push every ion off its rock-salt site, where every force is zero by symmetry.
static void displace_ions(MoleculeData& mol) {
    auto h_t = Kokkos::create_mirror_view(mol.translations);
    Kokkos::deep_copy(h_t, mol.translations);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d)
            h_t(k, d) += RealType(0.055) * RealType((k * 3 + d) % 5 - 2);
    Kokkos::deep_copy(mol.translations, h_t);
}

// Central difference of the energy with respect to one ion's Cartesian
// coordinate, done by moving its fractional translation by eps/a.
static RealType fd_force(SingleSlot& slot, int k, int d, RealType eps) {
    auto h_t = Kokkos::create_mirror(Kokkos::HostSpace(), slot.bws.dofs.translations);
    Kokkos::deep_copy(h_t, slot.bws.dofs.translations);
    const RealType orig = h_t(0, k, d);
    const RealType step = eps / kNaClA;

    auto at = [&](RealType delta) {
        auto h = Kokkos::create_mirror_view(slot.bws.dofs.translations);
        Kokkos::deep_copy(h, slot.bws.dofs.translations);
        h(0, k, d) = orig + delta;
        Kokkos::deep_copy(slot.bws.dofs.translations, h);
        return single_point<0>(slot);
    };
    const RealType Ep = at(step);
    const RealType Em = at(-step);
    at(RealType(0));
    return -(Ep - Em) / (RealType(2) * eps);
}

// Every ion, every direction, one term of the model.
static void check_forces_against_fd(uint32_t components) {
    auto nacl = make_nacl_ions(kNaClA, /*buckingham=*/true);
    displace_ions(nacl.first);
    auto slot = make_single_slot(nacl.first, nacl.second, RealType(1e-8));
    slot.settings.model.ewald_components = components;

    single_point<0>(slot);
    const auto h_g = host_snapshot(slot.bws.dofs.d_gt);

    const RealType eps = sizeof(RealType) < 8 ? RealType(1e-3) : RealType(1e-5);
    const RealType tol = sizeof(RealType) < 8 ? RealType(0.05) : RealType(1e-3);

    RealType largest = 0;
    for (int k = 0; k < 8; ++k)
        for (int d = 0; d < 3; ++d) {
            const RealType f = force_from_gradient(h_g(0, k, d));
            largest = std::max(largest, std::abs(f));
            REQUIRE_THAT(f, WithinAbs(fd_force(slot, k, d, eps), tol));
        }

    REQUIRE(largest > tol);
}

TEST_CASE("Buckingham forces match finite difference", "[gradients]") {
    check_forces_against_fd(EWALD_BUCKINGHAM);
}

TEST_CASE("Ewald real forces match finite difference", "[gradients]") {
    check_forces_against_fd(EWALD_PAIR_ERFC);
}

TEST_CASE("Ewald recip forces match finite difference", "[gradients]") {
    check_forces_against_fd(EWALD_RECIP);
}

// The single-point entry point and the optimiser's gradient kernel must
// describe the same energy.
TEST_CASE("Total energy+forces consistent with energy-only", "[gradients]") {
    auto nacl = make_nacl_ions(kNaClA, /*buckingham=*/true);
    displace_ions(nacl.first);
    auto& mol = nacl.first;
    auto& ws  = nacl.second;
    ws.ewald_accuracy = RealType(1e-8);

    auto slot = make_single_slot(mol, ws, RealType(1e-8));
    const RealType E_ef = single_point<0>(slot);

    const RealType E_eo = batch_evaluate_energy(mol, ws, /*max_rank=*/0,
                                                /*taper_on=*/0, /*taper_off=*/0,
                                                slot.settings.model.buck_damping);

    REQUIRE_THAT(E_ef, WithinAbs(E_eo, RealType(1e-4)));
}

TEST_CASE("NaCl equilibrium forces near zero", "[gradients]") {
    // Pure electrostatics on the ideal rock-salt geometry: every ion sits on a
    // site whose point group forbids a net force.
    auto nacl = make_nacl_ions(kNaClA, /*buckingham=*/false);
    auto slot = make_single_slot(nacl.first, nacl.second, RealType(1e-8));

    single_point<0>(slot);
    const auto h_g = host_snapshot(slot.bws.dofs.d_gt);

    for (int k = 0; k < 8; ++k)
        for (int d = 0; d < 3; ++d)
            REQUIRE_THAT(force_from_gradient(h_g(0, k, d)),
                         WithinAbs(RealType(0.0), RealType(1e-4)));
}
