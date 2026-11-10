// expand_asymmetric_unit must apply symmetry operations as Cartesian rotations:
// P3's fractional 3-fold, S_frac = [[0,-1,0],[1,-1,0],[0,0,1]], shears Cartesian
// coordinates. Every symmetry copy must be the same rigid body.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/molecule.hpp"
#include "klasp/constants.hpp"

#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;

namespace {

// One three-site bent molecule on a general position, expanded by `n_symops`
// operators supplied fractionally.
struct Expanded {
    std::vector<std::array<double, 3>> pos;   // n_symops * n_sites
    int n_sites;
    int n_symops;
};

Expanded expand(const double cell_params[6],
                const std::vector<std::array<double, 9>>& sym_frac) {
    const int n_symops = static_cast<int>(sym_frac.size());
    const int n_sites = 3;
    MoleculeData mol = allocate_molecule_data(1, n_sites, n_symops,
                                              /*max_multipole_rank=*/0);
    Workspace ws = allocate_workspace(n_symops * n_sites, 30000, 50000, n_sites);

    for (int i = 0; i < 6; ++i) mol.cell_params[i] = cell_params[i];

    // A bent, asymmetric body so a shear cannot be mistaken for a rotation.
    auto h_bp = Kokkos::create_mirror_view(mol.body_positions);
    auto h_bc = Kokkos::create_mirror_view(mol.body_charges);
    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    const double body[3][3] = {{-0.75, 0.00, 0.00},
                               { 0.65, 0.42, 0.00},
                               { 0.10, -0.55, 0.38}};
    for (int i = 0; i < n_sites; ++i) {
        for (int d = 0; d < 3; ++d)
            h_bp(i, d) = static_cast<RealType>(body[i][d] * ANGSTROM_TO_BOHR);
        h_bc(i) = 0; h_bt(i) = 0;
    }
    Kokkos::deep_copy(mol.body_positions, h_bp);
    Kokkos::deep_copy(mol.body_charges, h_bc);
    Kokkos::deep_copy(mol.body_types, h_bt);
    Kokkos::deep_copy(mol.body_multipoles, RealType(0));

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    for (int s = 0; s < n_symops; ++s) {
        for (int c = 0; c < 9; ++c) h_sr(s, c) = static_cast<RealType>(sym_frac[s][c]);
        for (int d = 0; d < 3; ++d) h_st(s, d) = 0;
        h_si(s) = 0;
    }
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    Kokkos::deep_copy(mol.sym_translations, h_st);
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    // General position and a general orientation, so nothing is zero by accident.
    auto h_t = Kokkos::create_mirror_view(mol.translations);
    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_t(0,0) = 0.17; h_t(0,1) = 0.23; h_t(0,2) = 0.31;
    h_r(0,0) = 0.19; h_r(0,1) = -0.27; h_r(0,2) = 0.11;
    Kokkos::deep_copy(mol.translations, h_t);
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    mol.n_indep = 1;
    mol.n_symops = n_symops;
    mol.n_sites_per_mol = n_sites;
    ws.n_types = 1;

    expand_asymmetric_unit<0>(mol, ws);

    auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ws.positions);
    Expanded out;
    out.n_sites = n_sites;
    out.n_symops = n_symops;
    out.pos.resize(static_cast<std::size_t>(n_symops) * n_sites);
    for (int a = 0; a < n_symops * n_sites; ++a)
        for (int d = 0; d < 3; ++d) out.pos[a][d] = static_cast<double>(h_pos(a, d));
    return out;
}

// Worst deviation of any copy's internal distances from the first copy's.
double worst_internal_distortion(const Expanded& e) {
    auto dist = [&](int mol_i, int s1, int s2) {
        const auto& p = e.pos[mol_i * e.n_sites + s1];
        const auto& q = e.pos[mol_i * e.n_sites + s2];
        return std::sqrt((p[0]-q[0])*(p[0]-q[0]) + (p[1]-q[1])*(p[1]-q[1])
                       + (p[2]-q[2])*(p[2]-q[2]));
    };
    double worst = 0;
    for (int m = 1; m < e.n_symops; ++m)
        for (int i = 0; i < e.n_sites; ++i)
            for (int j = i + 1; j < e.n_sites; ++j)
                worst = std::max(worst, std::abs(dist(m,i,j) - dist(0,i,j)));
    return worst;
}

}  // namespace

// Control: signed-permutation operators, where S_frac == S_cart.
TEST_CASE("expansion keeps molecules rigid (P2_1/c, signed permutations)",
          "[molecule][symmetry]") {
    const double cp[6] = {7.0 * ANGSTROM_TO_BOHR, 8.2 * ANGSTROM_TO_BOHR,
                          9.5 * ANGSTROM_TO_BOHR, 90.0, 97.0, 90.0};
    const std::vector<std::array<double, 9>> ops = {
        { 1,0,0,  0,1,0,  0,0,1},
        {-1,0,0,  0,1,0,  0,0,-1},
        {-1,0,0,  0,-1,0, 0,0,-1},
        { 1,0,0,  0,-1,0, 0,0,1}};
    const double worst = worst_internal_distortion(expand(cp, ops));
    std::printf("  P2_1/c worst internal distortion: %.3e Bohr\n", worst);
    REQUIRE(worst < 1e-9);
}

// P3's 3-fold in a hexagonal cell (gamma = 120) is a rotation only after
// conversion through the cell metric.
TEST_CASE("expansion keeps molecules rigid (P3, hexagonal 3-fold)",
          "[molecule][symmetry]") {
    const double cp[6] = {7.4 * ANGSTROM_TO_BOHR, 7.4 * ANGSTROM_TO_BOHR,
                          9.1 * ANGSTROM_TO_BOHR, 90.0, 90.0, 120.0};
    const std::vector<std::array<double, 9>> ops = {
        { 1, 0,0,   0, 1,0,  0,0,1},
        { 0,-1,0,   1,-1,0,  0,0,1},   // 3+
        {-1, 1,0,  -1, 0,0,  0,0,1}};  // 3-
    const double worst = worst_internal_distortion(expand(cp, ops));
    std::printf("  P3 worst internal distortion:     %.3e Bohr\n", worst);
    // A sheared copy is wrong by tenths of a Bohr on a ~1 Bohr bond.
    REQUIRE(worst < 1e-9);
}
