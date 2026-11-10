#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "test_support.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/molecule.hpp"
#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// Two independent single-atom molecules, so the pair at zero shift is not
// dropped as intramolecular.
TEST_CASE("Single pair Buckingham", "[buckingham]") {
    using namespace klasp::testing;

    // Two atoms at distance 3.0 Angstrom in a large cubic cell
    const RealType r_bohr = 3.0 * ANGSTROM_TO_BOHR;
    const RealType cell_len = 50.0 * ANGSTROM_TO_BOHR;

    auto mol = allocate_molecule_data(2, 1, 2);
    auto ws  = allocate_workspace(2, 30000, 1, 1);

    mol.cell_params[0] = cell_len; mol.cell_params[1] = cell_len;
    mol.cell_params[2] = cell_len;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    // One atom at the origin, one 3 A along a. Fractional COMs; the single site
    // of each molecule sits on its COM.
    auto h_t = Kokkos::create_mirror_view(mol.translations);
    Kokkos::deep_copy(h_t, RealType(0));
    h_t(1, 0) = r_bohr / cell_len;
    Kokkos::deep_copy(mol.translations, h_t);
    Kokkos::deep_copy(mol.rot_vectors, RealType(0));
    Kokkos::deep_copy(mol.body_positions, RealType(0));
    Kokkos::deep_copy(mol.body_charges, RealType(0));
    Kokkos::deep_copy(mol.body_types, 0);

    // P1. The pair kernel reads molecule k's identity copy at symop slot k.
    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    Kokkos::deep_copy(h_sr, RealType(0));
    for (int op = 0; op < 2; ++op) {
        h_sr(op, 0) = 1; h_sr(op, 4) = 1; h_sr(op, 8) = 1;
        h_si(op) = op;
    }
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    Kokkos::deep_copy(mol.sym_translations, RealType(0));
    Kokkos::deep_copy(mol.sym_indep_index, h_si);
    mol.n_indep = 2;
    mol.n_symops = 2;
    mol.n_sites_per_mol = 1;

    // Buck params: A=1000 kJ/mol, B=3.0 1/Ang, C=10.0 kJ/mol*Ang^6 -> convert to AU
    const RealType A_au = 1000.0 / AU_TO_KJ_PER_MOL;
    const RealType B_au = 3.0 * BOHR_TO_ANGSTROM;
    const RealType C_au = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    h_A(0, 0) = A_au;
    h_B(0, 0) = B_au;
    h_C(0, 0) = C_au;
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B);
    Kokkos::deep_copy(ws.buck_C, h_C);
    ws.n_types = 1;
    ws.buck_cutoff = 15.0 * ANGSTROM_TO_BOHR;

    auto slot = make_single_slot(mol, ws);
    const TangToenniesDamping damping{};   // the default: on, range 1.5
    slot.settings.model.buck_damping = damping;
    const RealType E = single_point_terms<0>(slot, EWALD_BUCKINGHAM);

    // Hand-calc for nearest image: A*exp(-B*r) - C/r^6 * f6(range*B*r)
    // where f6 is the Tang-Toennies damping function
    RealType Br = damping.range * B_au * r_bohr;
    RealType expBr = std::exp(-Br);
    RealType Br2 = Br*Br, Br3 = Br2*Br, Br4 = Br3*Br, Br5 = Br4*Br, Br6 = Br5*Br;
    RealType S6 = 1.0 + Br + Br2/2.0 + Br3/6.0 + Br4/24.0 + Br5/120.0 + Br6/720.0;
    RealType f6 = 1.0 - expBr * S6;
    RealType expected = A_au * std::exp(-B_au * r_bohr) - C_au / std::pow(r_bohr, 6) * f6;

    // Energy should be dominated by the nearest pair
    // (periodic images at ~47, 50, etc contribute negligibly)
    REQUIRE_THAT(E, WithinRel(expected, RealType(1e-4)));
}

// Undamped, -C/r^6 outruns A e^{-Br} below the inner turning point; the
// buckingham_r_min guard must stop the dive.
TEST_CASE("Undamped Buckingham does not dive below the inner turning point",
          "[buckingham]") {
    // A typical W99 C...C pair, atomic units.
    const RealType A = 1.2e5, B = 3.6, C = 1.5e3;
    const RealType r_min = buckingham_r_min(A, B, C);
    REQUIRE(r_min > 0.0);
    REQUIRE(r_min < 4.0);   // well inside any real contact

    auto eval = [&](RealType r, RealType rm) {
        // Untapered (ft = 1, dft = 0 inside the taper window).
        RealType e = 0, de_dr = 0;
        buckingham_untapered(r, RealType(1) / r,A, B, C, TangToenniesDamping::off(), rm, e, de_dr);
        return std::make_pair(e, de_dr / r);
    };

    SECTION("unguarded, the energy runs away") {
        // r_min = 0 disables the guard.
        const auto [e_half, fs_half] = eval(r_min * RealType(0.5), 0);
        const auto [e_top, fs_top] = eval(r_min, 0);
        REQUIRE(e_half < e_top);            // deeper than the barrier top
        REQUIRE(e_half < RealType(0));      // ... and attractive, not repulsive
    }

    SECTION("guarded, it keeps rising inward") {
        const auto [e_top, fs_top] = eval(r_min, r_min);
        const auto [e_mid, fs_mid] = eval(r_min * RealType(0.8), r_min);
        const auto [e_in, fs_in] = eval(r_min * RealType(0.5), r_min);
        REQUIRE(e_mid > e_top);
        REQUIRE(e_in > e_mid);
        // fs is (dV/dr)/r; pushing the pair apart means dV/dr < 0.
        REQUIRE(fs_in < RealType(0));
    }

    SECTION("continuous in value and slope at the turning point") {
        // Value: both branches evaluate to V(r_min) there.
        const RealType h = 1e-6 * r_min;
        const auto [e_out, fs_out] = eval(r_min + h, r_min);
        const auto [e_inn, fs_inn] = eval(r_min - h, r_min);
        REQUIRE_THAT(double(e_inn), WithinRel(double(e_out), 1e-6));

        // Slope: V'(r_min) = 0 on both sides, but r_min is a maximum of the true
        // form, so the join is C1 not C2 and the slope gap vanishes linearly in h.
        auto gap = [&](RealType step) {
            const auto [eo, fo] = eval(r_min + step, r_min);
            const auto [ei, fi] = eval(r_min - step, r_min);
            return std::abs(double(fi) - double(fo));
        };
        const double g1 = gap(1e-4 * r_min);
        const double g2 = gap(5e-5 * r_min);
        REQUIRE(g2 < 0.6 * g1);            // halving h roughly halves the gap
        REQUIRE(gap(1e-8 * r_min) < 1e-3); // and it goes to zero
    }

    SECTION("leaves the physical region untouched") {
        for (RealType r : {RealType(5), RealType(7), RealType(9)}) {
            const auto [e_guard, fs_guard] = eval(r, r_min);
            const auto [e_plain, fs_plain] = eval(r, 0);
            REQUIRE(e_guard == e_plain);
            REQUIRE(fs_guard == fs_plain);
        }
    }
}

// Bd = range * B must enter V, V' and V'' consistently; checked at the default
// and textbook ranges against a closed form.
TEST_CASE("The damping range reaches the energy, its slope and its curvature",
          "[buckingham][damping]") {
    // W99 H...H, atomic units.
    const RealType A = 131.42 * EV_TO_AU;
    const RealType B = BOHR_TO_ANGSTROM / 0.280899;
    const RealType C = 2.885328 * EV_TO_AU * std::pow(ANGSTROM_TO_BOHR, 6);
    const auto close = [](RealType got, RealType want, RealType rel, RealType abs) {
        return std::abs(got - want) <= rel * std::abs(want) + abs;
    };

    for (const RealType range : {RealType(1.0), RealType(1.5)}) {
        TangToenniesDamping damping;
        damping.range = range;
        const auto V = [&](RealType r, RealType& de, RealType& d2e) {
            RealType e = 0;
            buckingham_untapered(r, RealType(1) / r,A, B, C, damping, RealType(0), e, de, &d2e);
            return e;
        };
        for (RealType r = 3.0; r <= 9.0; r += 0.25) {
            INFO("range " << range << ", r " << r << " Bohr");
            const RealType x = range * B * r;
            RealType s6 = 0, term = 1;
            for (int k = 0; k <= 6; ++k) { s6 += term; term *= x / RealType(k + 1); }
            const RealType f6 = 1 - std::exp(-x) * s6;
            const RealType expected = A * std::exp(-B * r) - C / std::pow(r, 6) * f6;

            RealType de = 0, d2e = 0, dp = 0, d2p = 0, dm = 0, d2m = 0;
            const RealType e = V(r, de, d2e);
            const RealType h = 1e-4;
            const RealType ep = V(r + h, dp, d2p);
            const RealType em = V(r - h, dm, d2m);
            CHECK(close(e, expected, 1e-12, 1e-15));
            CHECK(close(de, (ep - em) / (2 * h), 1e-6, 1e-11));
            CHECK(close(d2e, (dp - dm) / (2 * h), 1e-6, 1e-11));
        }
    }
}

// Range 1.5 keeps the W99 wells within half a percent and still removes the
// inner dive; range 1 makes H...H over 10% shallower, and range 2.5 dives again.
TEST_CASE("The default damping range leaves W99 wells alone and still stops the dive",
          "[buckingham][damping]") {
    struct Pair { const char* name; double A_ev, rho_ang, C6; };
    const Pair pairs[] = {
        {"H_W1-H_W1", 131.42, 0.280899, 2.885328},
        {"C_W3-C_W3", 2802.12, 0.277778, 17.638572},
        {"N_W3-N_W3", 1989.27, 0.287356, 24.633137},
        {"O_W1-O_W1", 2498.22, 0.252525, 13.067571},
        {"H_W1-O_W1", 572.9887192606849, 0.2659573621546837, 6.140376901973363},
    };
    const TangToenniesDamping standard{};
    REQUIRE(standard.enabled);
    REQUIRE(standard.range == RealType(1.5));

    struct Well { RealType r, depth; };
    for (const Pair& p : pairs) {
        INFO(p.name);
        const RealType A = p.A_ev * EV_TO_AU;
        const RealType B = BOHR_TO_ANGSTROM / p.rho_ang;
        const RealType C = p.C6 * EV_TO_AU * std::pow(ANGSTROM_TO_BOHR, 6);
        const auto V = [&](RealType r, const TangToenniesDamping& d) {
            RealType e = 0, de = 0;
            buckingham_untapered(r, RealType(1) / r,A, B, C, d, RealType(0), e, de);
            return e;
        };
        const auto well = [&](const TangToenniesDamping& d) {
            Well w{0, 1e30};
            for (RealType r = 4.0; r < 12.0; r += 1e-3)
                if (const RealType v = V(r, d); v < w.depth) w = {r, v};
            return w;
        };
        // True when V only falls from 0.1 Bohr out to the well: no inner
        // maximum for a structure to tip over.
        const auto climbs_all_the_way_in = [&](const TangToenniesDamping& d, RealType r_well) {
            RealType prev = V(0.1, d);
            for (RealType r = 0.1 + 1e-3; r < r_well; r += 1e-3) {
                const RealType v = V(r, d);
                if (v > prev) return false;
                prev = v;
            }
            return true;
        };

        const Well undamped = well(TangToenniesDamping::off());
        const Well damped = well(standard);
        CHECK(std::abs(damped.depth / undamped.depth - 1) < 0.005);
        CHECK(climbs_all_the_way_in(standard, damped.r));
    }

    // H...H at both ends, for scale.
    const Pair& hh = pairs[0];
    const RealType A = hh.A_ev * EV_TO_AU;
    const RealType B = BOHR_TO_ANGSTROM / hh.rho_ang;
    const RealType C = hh.C6 * EV_TO_AU * std::pow(ANGSTROM_TO_BOHR, 6);
    const auto V = [&](RealType r, RealType range) {
        TangToenniesDamping d;
        d.range = range;
        RealType e = 0, de = 0;
        buckingham_untapered(r, RealType(1) / r,A, B, C, d, RealType(0), e, de);
        return e;
    };
    const auto depth = [&](RealType range) {
        RealType lo = 1e30;
        for (RealType r = 4.0; r < 12.0; r += 1e-3) lo = std::min(lo, V(r, range));
        return lo;
    };
    RealType undamped_depth = 1e30;
    for (RealType r = 4.0; r < 12.0; r += 1e-3) {
        RealType e = 0, de = 0;
        buckingham_untapered(r, RealType(1) / r,A, B, C, TangToenniesDamping::off(), RealType(0), e, de);
        undamped_depth = std::min(undamped_depth, e);
    }
    CHECK(1 - depth(1.0) / undamped_depth > 0.10);   // the textbook range intrudes

    bool falls_somewhere = false;   // the stretched range lets the dive back in
    RealType prev = V(0.1, 2.5);
    for (RealType r = 0.1 + 1e-3; r < 6.0; r += 1e-3) {
        const RealType v = V(r, 2.5);
        if (v > prev) { falls_somewhere = true; break; }
        prev = v;
    }
    CHECK(falls_somewhere);
}
