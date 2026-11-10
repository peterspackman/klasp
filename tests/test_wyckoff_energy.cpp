#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/wyckoff.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/molecule.hpp"
#include "klasp/batch.hpp"
#include "klasp/cell.hpp"
#include "klasp/ewald.hpp"
#include "klasp/rotation.hpp"
#include "klasp/constants.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinRel;

// A molecule on a special position: its copies are the coset representatives,
// not the full operation list, so the crystal holds `multiplicity` molecules.
// Checks the structure has the claimed symmetry and that its energy matches the
// same atoms written out in P1.

namespace {

constexpr int N_SITES = 4;             // ammonia
constexpr double A_ANG = 6.4;          // hexagonal cell, loose enough not to clash
constexpr double C_ANG = 7.1;

// Ideal C3v ammonia, 3-fold along z, in Bohr.
std::vector<double> ammonia_body() {
    const double rh = 0.9377, zh = -0.3263, zn = 0.0704;
    std::vector<double> p = {0, 0, zn};
    for (int k = 0; k < 3; ++k) {
        const double a = 2.0 * M_PI * k / 3.0;
        p.push_back(rh * std::cos(a));
        p.push_back(rh * std::sin(a));
        p.push_back(zh);
    }
    for (double& v : p) v *= ANGSTROM_TO_BOHR;
    return p;
}

std::vector<double> ammonia_ops() {
    std::vector<double> ops;
    auto push = [&](double m[9]) { ops.insert(ops.end(), m, m + 9); };
    double I[9] = {1,0,0, 0,1,0, 0,0,1};
    push(I);
    for (int k = 1; k <= 2; ++k) {
        const double a = 2.0 * M_PI * k / 3.0, c = std::cos(a), s = std::sin(a);
        double r[9] = {c,-s,0, s,c,0, 0,0,1};
        push(r);
    }
    for (int k = 0; k < 3; ++k) {
        const double a = M_PI * k / 3.0;
        const double n[3] = {std::cos(a), std::sin(a), 0};
        double m[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i*3+j] = (i == j ? 1.0 : 0.0) - 2.0 * n[i] * n[j];
        push(m);
    }
    return ops;
}

void fill_body(MoleculeData& mol, int n_indep) {
    const auto body = ammonia_body();
    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    auto h_t = Kokkos::create_mirror_view(mol.body_types);
    for (int m = 0; m < n_indep; ++m)
        for (int s = 0; s < N_SITES; ++s) {
            const int row = m * N_SITES + s;
            for (int d = 0; d < 3; ++d) h_b(row, d) = body[s * 3 + d];
            h_q(row) = (s == 0) ? -0.9 : 0.3;        // N negative, H positive
            h_t(row) = (s == 0) ? 0 : 1;
        }
    Kokkos::deep_copy(mol.body_positions, h_b);
    Kokkos::deep_copy(mol.body_charges, h_q);
    Kokkos::deep_copy(mol.body_types, h_t);
}

void set_cell(MoleculeData& mol) {
    mol.cell_params[0] = A_ANG * ANGSTROM_TO_BOHR;
    mol.cell_params[1] = A_ANG * ANGSTROM_TO_BOHR;
    mol.cell_params[2] = C_ANG * ANGSTROM_TO_BOHR;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 120.0;
}

Workspace make_ws() {
    Workspace ws = allocate_workspace(256, 40000, 60000, 2);
    ws.n_types = 2;
    ws.ghost_type = -1;
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR;
    // Charges only: Buckingham A is zeroed; B stays non-zero because it divides.
    auto hA = Kokkos::create_mirror_view(ws.buck_A);
    auto hB = Kokkos::create_mirror_view(ws.buck_B);
    auto hC = Kokkos::create_mirror_view(ws.buck_C);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) { hA(i,j) = 0; hB(i,j) = 1; hC(i,j) = 0; }
    Kokkos::deep_copy(ws.buck_A, hA);
    Kokkos::deep_copy(ws.buck_B, hB);
    Kokkos::deep_copy(ws.buck_C, hC);
    ws.ewald_accuracy = 1e-10;
    return ws;
}

// The site klasp derives, its coset representatives, and an orientation the
// molecule is invariant in.
struct Placement {
    WyckoffSite site;
    std::vector<double> rot_vec;   // angle-axis for klasp's rigid-body DOF
};

Placement place_on_site(int sg_num, int want_multiplicity) {
    const SpaceGroupInfo& sg = get_space_group(sg_num);
    auto sites = special_positions(sg);
    auto it = std::find_if(sites.begin(), sites.end(),
        [&](const WyckoffSite& s) { return s.multiplicity == want_multiplicity; });
    REQUIRE(it != sites.end());

    double cell[3][3];
    canonical_cell(sg.lattice, cell);
    const auto w = site_ops_cartesian(sg, *it, cell);
    const auto oris = orientations_on_site(w, ammonia_ops());
    REQUIRE(!oris.empty());

    RealType R[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R[i][j] = static_cast<RealType>(oris[i * 3 + j]);
    RealType aa[3];
    rotation_matrix_to_angle_axis(R, aa);
    return {*it, {aa[0], aa[1], aa[2]}};
}

}  // namespace

TEST_CASE("a molecule on a special position is a crystal with the full group",
          "[wyckoff][energy]") {
    // The copies the coset representatives generate must be carried onto
    // themselves by every operation of the space group.
    const SpaceGroupInfo& sg = get_space_group(194);
    const Placement p = place_on_site(194, 4);
    REQUIRE(p.site.multiplicity == 4);
    REQUIRE(static_cast<int>(p.site.coset_reps.size()) == 4);

    auto apply = [&](int op, const double x[3], double out[3]) {
        for (int i = 0; i < 3; ++i) {
            out[i] = sg.symops[op].trans[i];
            for (int j = 0; j < 3; ++j) out[i] += sg.symops[op].rot[i][j] * x[j];
        }
    };
    // The four molecular centres.
    std::vector<std::array<double, 3>> centres;
    for (int r : p.site.coset_reps) {
        double y[3];
        apply(r, p.site.rep, y);
        centres.push_back({y[0], y[1], y[2]});
    }
    auto same = [](const double a[3], const double b[3]) {
        for (int i = 0; i < 3; ++i) {
            const double d = a[i] - b[i];
            if (std::abs(d - std::round(d)) > 1e-6) return false;
        }
        return true;
    };
    for (int op = 0; op < sg.n_symops; ++op)
        for (const auto& c : centres) {
            double y[3];
            apply(op, c.data(), y);
            bool found = false;
            for (const auto& d : centres) if (same(y, d.data())) { found = true; break; }
            INFO("operation " << op << " took a centre off the set");
            REQUIRE(found);
        }
}

TEST_CASE("a special-position structure has the energy of its P1 expansion",
          "[wyckoff][energy]") {
    const SpaceGroupInfo& sg = get_space_group(194);
    const Placement p = place_on_site(194, 4);
    const int mult = p.site.multiplicity;

    // --- A: one molecule, the coset representatives as its copies ------------
    MoleculeData a = allocate_molecule_data(1, N_SITES, mult);
    Workspace wsa = make_ws();
    set_cell(a);
    fill_body(a, 1);
    a.n_indep = 1;
    a.n_symops = mult;
    a.n_sites_per_mol = N_SITES;
    {
        auto h_t = Kokkos::create_mirror_view(a.translations);
        for (int d = 0; d < 3; ++d) h_t(0, d) = static_cast<RealType>(p.site.rep[d]);
        Kokkos::deep_copy(a.translations, h_t);
        auto h_r = Kokkos::create_mirror_view(a.rot_vectors);
        for (int d = 0; d < 3; ++d) h_r(0, d) = static_cast<RealType>(p.rot_vec[d]);
        Kokkos::deep_copy(a.rot_vectors, h_r);

        auto h_sr = Kokkos::create_mirror_view(a.sym_rotations);
        auto h_st = Kokkos::create_mirror_view(a.sym_translations);
        auto h_si = Kokkos::create_mirror_view(a.sym_indep_index);
        auto h_sc = Kokkos::create_mirror_view(a.mol_site_count);
        for (int s = 0; s < mult; ++s) {
            const int op = p.site.coset_reps[s];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    h_sr(s, i * 3 + j) = static_cast<RealType>(sg.symops[op].rot[i][j]);
            for (int d = 0; d < 3; ++d) {
                double t = sg.symops[op].trans[d];
                h_st(s, d) = static_cast<RealType>(t - std::floor(t));
            }
            h_si(s) = 0;
            h_sc(s) = N_SITES;
        }
        Kokkos::deep_copy(a.sym_rotations, h_sr);
        Kokkos::deep_copy(a.sym_translations, h_st);
        Kokkos::deep_copy(a.sym_indep_index, h_si);
        Kokkos::deep_copy(a.mol_site_count, h_sc);
    }
    // The identity must come first.
    validate_symmetry_multiplicity(a);
    const RealType Ea = [&] {
        RealType cell[3][3];
        cell_params_to_matrix(a.cell_params, cell);
        compute_ewald_params(cell, mult * N_SITES, wsa.ewald_accuracy, wsa);
        return batch_evaluate_energy(a, wsa, 0);
    }();

    // --- B: the same four molecules written out in P1 ------------------------
    // Positions and orientations from A's own expansion: the same atoms, only
    // the description differs.
    expand_asymmetric_unit<0>(a, wsa);
    auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wsa.positions);

    // No two molecules or atoms may coincide, or the energy comparison could
    // pass on one molecule counted four times.
    {
        std::vector<std::array<RealType, 3>> cen(mult, {0, 0, 0});
        for (int m = 0; m < mult; ++m) {
            for (int s = 0; s < N_SITES; ++s)
                for (int d = 0; d < 3; ++d) cen[m][d] += h_pos(m * N_SITES + s, d);
            for (int d = 0; d < 3; ++d) cen[m][d] /= N_SITES;
        }
        for (int i = 0; i < mult; ++i)
            for (int j = i + 1; j < mult; ++j) {
                RealType d2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const RealType e = cen[i][d] - cen[j][d];
                    d2 += e * e;
                }
                INFO("copies " << i << " and " << j << " are " << std::sqrt(d2)
                     << " Bohr apart");
                REQUIRE(d2 > RealType(1.0));
            }
        const int n_atoms = mult * N_SITES;
        for (int i = 0; i < n_atoms; ++i)
            for (int j = i + 1; j < n_atoms; ++j) {
                RealType d2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const RealType e = h_pos(i, d) - h_pos(j, d);
                    d2 += e * e;
                }
                INFO("atoms " << i << " and " << j << " coincide");
                REQUIRE(d2 > RealType(0.01));
            }
    }

    MoleculeData b = allocate_molecule_data(mult, N_SITES, mult);
    Workspace wsb = make_ws();
    set_cell(b);
    fill_body(b, mult);
    b.n_indep = mult;
    b.n_symops = mult;
    b.n_sites_per_mol = N_SITES;
    {
        // Each copy becomes an independent molecule carrying the identity, its
        // body frame its own atoms.
        auto h_b = Kokkos::create_mirror_view(b.body_positions);
        auto h_t = Kokkos::create_mirror_view(b.translations);
        auto h_r = Kokkos::create_mirror_view(b.rot_vectors);
        RealType cell[3][3], inv[3][3];
        cell_params_to_matrix(b.cell_params, cell);
        REQUIRE(mat3_inv(cell, inv));
        for (int m = 0; m < mult; ++m) {
            // Centroid of this copy, in Cartesian then fractional.
            RealType cen[3] = {0, 0, 0};
            for (int s = 0; s < N_SITES; ++s)
                for (int d = 0; d < 3; ++d) cen[d] += h_pos(m * N_SITES + s, d);
            for (int d = 0; d < 3; ++d) cen[d] /= N_SITES;
            for (int d = 0; d < 3; ++d) {
                RealType f = 0;
                for (int k = 0; k < 3; ++k) f += inv[d][k] * cen[k];
                h_t(m, d) = f;
            }
            for (int d = 0; d < 3; ++d) h_r(m, d) = 0;
            for (int s = 0; s < N_SITES; ++s)
                for (int d = 0; d < 3; ++d)
                    h_b(m * N_SITES + s, d) = h_pos(m * N_SITES + s, d) - cen[d];
        }
        Kokkos::deep_copy(b.body_positions, h_b);
        Kokkos::deep_copy(b.translations, h_t);
        Kokkos::deep_copy(b.rot_vectors, h_r);

        auto h_sr = Kokkos::create_mirror_view(b.sym_rotations);
        auto h_st = Kokkos::create_mirror_view(b.sym_translations);
        auto h_si = Kokkos::create_mirror_view(b.sym_indep_index);
        auto h_sc = Kokkos::create_mirror_view(b.mol_site_count);
        for (int s = 0; s < mult; ++s) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) h_sr(s, i * 3 + j) = (i == j) ? 1 : 0;
            for (int d = 0; d < 3; ++d) h_st(s, d) = 0;
            h_si(s) = s;                       // one copy each: P1, Z' = mult
            h_sc(s) = N_SITES;
        }
        Kokkos::deep_copy(b.sym_rotations, h_sr);
        Kokkos::deep_copy(b.sym_translations, h_st);
        Kokkos::deep_copy(b.sym_indep_index, h_si);
        Kokkos::deep_copy(b.mol_site_count, h_sc);
    }
    validate_symmetry_multiplicity(b);
    const RealType Eb = [&] {
        RealType cell[3][3];
        cell_params_to_matrix(b.cell_params, cell);
        compute_ewald_params(cell, mult * N_SITES, wsb.ewald_accuracy, wsb);
        return batch_evaluate_energy(b, wsb, 0);
    }();

    INFO("special position " << Ea << " vs P1 expansion " << Eb);
    REQUIRE(Ea != RealType(0));
    REQUIRE_THAT(static_cast<double>(Eb),
                 WithinRel(static_cast<double>(Ea), 1e-9));
}
