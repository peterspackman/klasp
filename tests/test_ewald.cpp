#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "test_support.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/molecule.hpp"
#include <cmath>

using namespace klasp;
using namespace klasp::testing;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// Rock-salt energy: real + reciprocal + self (no Buckingham parameters,
// single-atom molecules).
static RealType nacl_energy(RealType a, RealType accuracy = RealType(1e-10),
                            RealType charge_scale = RealType(1)) {
    auto nacl = make_nacl_ions(a);
    auto& mol = nacl.first;
    auto& ws  = nacl.second;
    if (charge_scale != RealType(1)) {
        auto h_q = Kokkos::create_mirror_view(mol.body_charges);
        Kokkos::deep_copy(h_q, mol.body_charges);
        for (int i = 0; i < mol.n_indep; ++i) h_q(i) *= charge_scale;
        Kokkos::deep_copy(mol.body_charges, h_q);
    }
    auto slot = make_single_slot(mol, ws, accuracy);
    return single_point<0>(slot);
}

TEST_CASE("NaCl Madelung constant", "[ewald]") {
    RealType a = 5.6402 * ANGSTROM_TO_BOHR;
    RealType E = nacl_energy(a);

    // M = -E * r_nn / n_formula_units, r_nn = a/2, 4 formula units
    RealType r_nn = a / RealType(2.0);
    RealType M = -E * r_nn / RealType(4.0);

    REQUIRE_THAT(M, WithinRel(RealType(1.747565), RealType(1e-4)));
}

TEST_CASE("Ewald self energy is negative", "[ewald]") {
    auto nacl = make_nacl_ions();
    auto slot = make_single_slot(nacl.first, nacl.second, RealType(1e-10));
    RealType E_self = single_point_terms<0>(slot, EWALD_SELF);
    REQUIRE(E_self < 0.0);
}

TEST_CASE("Ewald charge scaling", "[ewald]") {
    RealType a = 5.6402 * ANGSTROM_TO_BOHR;
    RealType E1 = nacl_energy(a);

    RealType E2 = nacl_energy(a, RealType(1e-10), RealType(2.0));
    REQUIRE_THAT(E2, WithinRel(RealType(4.0) * E1, RealType(1e-5)));
}

TEST_CASE("Ewald convergence with accuracy", "[ewald]") {
    RealType a = 5.6402 * ANGSTROM_TO_BOHR;

    RealType energies[3];
    RealType accuracies[3] = {RealType(1e-4), RealType(1e-6), RealType(1e-8)};
    for (int ai = 0; ai < 3; ++ai)
        energies[ai] = nacl_energy(a, accuracies[ai]);

    RealType d0 = std::abs(energies[0] - energies[2]);
    RealType d1 = std::abs(energies[1] - energies[2]);
    REQUIRE((d0 >= d1 || std::abs(d0 - d1) < RealType(1e-4)));
}

TEST_CASE("Ewald components sum to total", "[ewald]") {
    auto nacl = make_nacl_ions();
    auto slot = make_single_slot(nacl.first, nacl.second, RealType(1e-10));

    RealType E_real  = single_point_terms<0>(slot, EWALD_PAIR_ERFC);
    RealType E_recip = single_point_terms<0>(slot, EWALD_RECIP);
    RealType E_self  = single_point_terms<0>(slot, EWALD_SELF);
    RealType E_intra = single_point_terms<0>(slot, EWALD_INTRA);
    RealType E_buck  = single_point_terms<0>(slot, EWALD_BUCKINGHAM);
    RealType E_total = single_point<0>(slot);

    // Both zero for this cell; asserted so the sum below cannot hide them.
    REQUIRE_THAT(E_intra, WithinAbs(RealType(0), RealType(1e-12)));
    REQUIRE_THAT(E_buck,  WithinAbs(RealType(0), RealType(1e-12)));

    REQUIRE_THAT(E_total, WithinAbs(E_real + E_recip + E_self, RealType(1e-5)));
}
