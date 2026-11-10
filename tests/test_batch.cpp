#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "test_support.hpp"
#include "klasp/cell.hpp"
#include "klasp/descriptors.hpp"
#include "klasp/powder.hpp"
#include "klasp/ewald.hpp"
#include "klasp/constants.hpp"
#include "klasp/dedup.hpp"  // AMD_K
#include <cmath>
#include <numeric>
#include <random>

using namespace klasp;
using Catch::Matchers::WithinAbs;

static std::pair<MoleculeData, Workspace> make_nacl_dimer() {
    auto mol = allocate_molecule_data(1, 2, 1);
    auto ws  = allocate_workspace(64, 30000, 50000, 2);

    RealType a = 5.6402 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.25; h_t(0, 1) = 0.25; h_t(0, 2) = 0.25;
    Kokkos::deep_copy(mol.translations, h_t);

    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.0; h_r(0, 1) = 0.0; h_r(0, 2) = 0.0;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0, 0) = -a/4.0; h_b(0, 1) = 0.0; h_b(0, 2) = 0.0;
    h_b(1, 0) =  a/4.0; h_b(1, 1) = 0.0; h_b(1, 2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 1.0; h_q(1) = -1.0;
    Kokkos::deep_copy(mol.body_charges, h_q);

    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    h_bt(0) = 0; h_bt(1) = 1;
    Kokkos::deep_copy(mol.body_types, h_bt);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    h_sr(0, 0) = 1; h_sr(0, 1) = 0; h_sr(0, 2) = 0;
    h_sr(0, 3) = 0; h_sr(0, 4) = 1; h_sr(0, 5) = 0;
    h_sr(0, 6) = 0; h_sr(0, 7) = 0; h_sr(0, 8) = 1;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);

    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    h_st(0, 0) = 0; h_st(0, 1) = 0; h_st(0, 2) = 0;
    Kokkos::deep_copy(mol.sym_translations, h_st);

    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0) = 0;
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = 1;
    mol.n_symops = 1;
    mol.n_sites_per_mol = 2;

    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            h_A(i,j) = h_B(i,j) = h_C(i,j) = 0.0;
    h_A(0,1) = 1000.0 / AU_TO_KJ_PER_MOL; h_A(1,0) = 1000.0 / AU_TO_KJ_PER_MOL;
    h_B(0,1) = 3.0 * BOHR_TO_ANGSTROM;    h_B(1,0) = 3.0 * BOHR_TO_ANGSTROM;
    h_C(0,1) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_C(1,0) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B);
    Kokkos::deep_copy(ws.buck_C, h_C);
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR;
    ws.n_types = 2;

    return {mol, ws};
}

static void setup_ewald(MoleculeData& mol, Workspace& ws) {
    expand_asymmetric_unit(mol, ws);
    RealType cell[3][3];
    cell_params_to_matrix(mol.cell_params, cell);
    compute_ewald_params(cell, ws.n_atoms, 1e-8, ws);
    RealType cutoff = (ws.rcut > ws.buck_cutoff) ? ws.rcut : ws.buck_cutoff;
    fill_shifts(cell, cutoff, ws);
}

static void setup_batch_cells(BatchWorkspace& bws, const MoleculeData& mol,
                               RealType accuracy = 1e-8) {
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int b = 0; b < bws.geom.n_batch; ++b)
        for (int i = 0; i < 6; ++i)
            h_cp(b, i) = mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    bws.tables.ewald_accuracy = accuracy;
    batch_setup_cells(bws.tables, bws.geom);
}

TEST_CASE("Batch allocation compiles and runs", "[batch]") {
    auto _result = make_nacl_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    setup_ewald(mol, ws);

    int n_batch = 4;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);

    REQUIRE(bws.geom.n_batch == 4);
    REQUIRE(bws.geom.n_atoms == 2);
    REQUIRE(bws.tables.n_indep == 1);
    REQUIRE(bws.geom.positions.extent(0) == 4);
    REQUIRE(bws.geom.positions.extent(1) == 2);
    REQUIRE(bws.geom.positions.extent(2) == 3);
}

// The image count a cutoff needs is set by the interplanar spacing, not the
// shortest edge. Cell from a CSP run: 16 images along a, shortest edge 13 Bohr.
TEST_CASE("An oblique cell gets enough images to cover the cutoff", "[batch]") {
    auto _result = make_nacl_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    ws.buck_cutoff = 15.0 * ANGSTROM_TO_BOHR;
    setup_ewald(mol, ws);

    auto bws = allocate_batch_workspace(1, ws, mol);

    // a, b, c in Bohr; angles in degrees. Volume is a tenth of a*b*c.
    const RealType oblique[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = oblique[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);

    const RealType accuracy = 1e-6;
    bws.tables.ewald_accuracy = accuracy;
    batch_setup_cells(bws.tables, bws.geom);

    // Interplanar spacings, computed independently.
    RealType cell[3][3];
    cell_params_to_matrix(oblique, cell);
    auto col = [&](int i, int d) { return cell[d][i]; };
    auto spacing = [&](int i) {
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        const RealType cx = col(j,1)*col(k,2) - col(j,2)*col(k,1);
        const RealType cy = col(j,2)*col(k,0) - col(j,0)*col(k,2);
        const RealType cz = col(j,0)*col(k,1) - col(j,1)*col(k,0);
        RealType vol = std::abs(col(0,0)*(col(1,1)*col(2,2) - col(1,2)*col(2,1))
                              - col(1,0)*(col(0,1)*col(2,2) - col(0,2)*col(2,1))
                              + col(2,0)*(col(0,1)*col(1,2) - col(0,2)*col(1,1)));
        return vol / std::sqrt(cx*cx + cy*cy + cz*cz);
    };

    auto h_kappa = Kokkos::create_mirror_view(bws.geom.kappa);
    Kokkos::deep_copy(h_kappa, bws.geom.kappa);
    const RealType rcut = std::sqrt(-std::log(accuracy)) / h_kappa(0);
    const RealType cutoff = rcut > bws.tables.buck_cutoff ? rcut : bws.tables.buck_cutoff;

    auto h_ms = Kokkos::create_mirror_view(bws.geom.max_shift);
    Kokkos::deep_copy(h_ms, bws.geom.max_shift);

    for (int i = 0; i < 3; ++i) {
        INFO("axis " << i << " spacing " << spacing(i) << " images " << h_ms(0, i));
        REQUIRE(h_ms(0, i) * spacing(i) >= cutoff);
    }
}

// The (h,k,l) box covering |g| < gmax is bounded by the lattice vector lengths,
// not the interplanar spacings; the two agree only on an orthorhombic cell.
TEST_CASE("An oblique cell's reciprocal box covers the gmax sphere", "[batch]") {
    auto _result = make_nacl_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    ws.buck_cutoff = 15.0 * ANGSTROM_TO_BOHR;
    setup_ewald(mol, ws);

    auto bws = allocate_batch_workspace(1, ws, mol);

    // Same cell as the real-space test above.
    const RealType oblique[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = oblique[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);

    const RealType accuracy = 1e-6;
    bws.tables.ewald_accuracy = accuracy;
    batch_setup_cells(bws.tables, bws.geom);

    auto h_kappa = Kokkos::create_mirror_view(bws.geom.kappa);
    Kokkos::deep_copy(h_kappa, bws.geom.kappa);
    const RealType gmax = 2 * h_kappa(0) * std::sqrt(-std::log(accuracy));

    // The columns of `cell` are the lattice vectors, so the rows of its inverse
    // are b_i / 2 pi.
    RealType cell[3][3], inv[3][3];
    cell_params_to_matrix(oblique, cell);
    REQUIRE(mat3_inv(cell, inv));
    RealType b[3][3];
    for (int i = 0; i < 3; ++i)
        for (int d = 0; d < 3; ++d) b[i][d] = 2 * M_PI * inv[i][d];

    // Record the largest index per axis the sphere reaches within a wide sweep.
    const int SWEEP = 64;
    int needed[3] = {0, 0, 0};
    for (int h = -SWEEP; h <= SWEEP; ++h)
        for (int k = -SWEEP; k <= SWEEP; ++k)
            for (int l = -SWEEP; l <= SWEEP; ++l) {
                RealType g2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const RealType gd = h*b[0][d] + k*b[1][d] + l*b[2][d];
                    g2 += gd * gd;
                }
                if (g2 >= gmax * gmax) continue;
                const int idx[3] = {std::abs(h), std::abs(k), std::abs(l)};
                for (int d = 0; d < 3; ++d)
                    if (idx[d] > needed[d]) needed[d] = idx[d];
            }

    auto h_mhkl = Kokkos::create_mirror_view(bws.geom.max_hkl);
    Kokkos::deep_copy(h_mhkl, bws.geom.max_hkl);

    for (int i = 0; i < 3; ++i) {
        INFO("axis " << i << " needs " << needed[i] << ", box is " << h_mhkl(0, i));
        REQUIRE(needed[i] < SWEEP);          // the sweep itself has to be wide enough
        REQUIRE(h_mhkl(0, i) >= needed[i]);
    }

    // Sizing c from its interplanar spacing gives a box several times too small.
    const RealType c_len = std::sqrt(cell[0][2]*cell[0][2] + cell[1][2]*cell[1][2]
                                   + cell[2][2]*cell[2][2]);
    const RealType c_spacing = 2 * M_PI / std::sqrt(b[2][0]*b[2][0] + b[2][1]*b[2][1]
                                                  + b[2][2]*b[2][2]);
    REQUIRE(c_len > 3 * c_spacing);
}

// Every emitted k must be a reciprocal lattice vector: k . a_i is 2 pi times an
// integer, checked on an oblique cell.
TEST_CASE("Ewald k-vectors are reciprocal to the lattice", "[ewald]") {
    auto _result = make_nacl_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    (void)mol;

    const RealType oblique[6] = {14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430};
    RealType cell[3][3];
    cell_params_to_matrix(oblique, cell);   // columns are the lattice vectors
    compute_ewald_params(cell, 2, 1e-6, ws);

    REQUIRE(ws.n_kvecs > 0);
    auto h_k = Kokkos::create_mirror_view(ws.kvecs);
    Kokkos::deep_copy(h_k, ws.kvecs);

    RealType worst = 0;
    for (int i = 0; i < ws.n_kvecs; ++i)
        for (int axis = 0; axis < 3; ++axis) {
            RealType dot = 0;
            for (int d = 0; d < 3; ++d) dot += h_k(i, d) * cell[d][axis];
            const RealType index = dot / (2 * M_PI);
            const RealType off = std::abs(index - std::round(index));
            if (off > worst) worst = off;
        }
    INFO("worst departure from an integer Miller index: " << worst);
    REQUIRE(worst < 1e-9);
}

TEST_CASE("Batch expand matches single expand", "[batch]") {
    auto _result = make_nacl_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    setup_ewald(mol, ws);

    int n_batch = 3;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);

    for (int b = 0; b < n_batch; ++b)
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_bt(b, k, d) = h_mt(k, d) + b * 0.01;
                h_br(b, k, d) = h_mr(k, d) + b * 0.005;
            }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();

    auto h_bpos = Kokkos::create_mirror_view(bws.geom.positions);
    Kokkos::deep_copy(h_bpos, bws.geom.positions);

    for (int b = 0; b < n_batch; ++b) {
        auto h_t = Kokkos::create_mirror_view(mol.translations);
        auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_t(k, d) = h_bt(b, k, d);
                h_r(k, d) = h_br(b, k, d);
            }
        Kokkos::deep_copy(mol.translations, h_t);
        Kokkos::deep_copy(mol.rot_vectors, h_r);
        expand_asymmetric_unit(mol, ws);

        auto h_spos = Kokkos::create_mirror_view(ws.positions);
        Kokkos::deep_copy(h_spos, ws.positions);

        for (int a = 0; a < ws.n_atoms; ++a)
            for (int d = 0; d < 3; ++d)
                REQUIRE_THAT(h_bpos(b, a, d), WithinAbs(h_spos(a, d), RealType(1e-4)));
    }
}

TEST_CASE("Batch energy matches single-structure energy", "[batch]") {
    auto _result = make_nacl_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    setup_ewald(mol, ws);

    int n_batch = 3;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);

    for (int b = 0; b < n_batch; ++b)
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_bt(b, k, d) = h_mt(k, d) + b * 0.01;
                h_br(b, k, d) = h_mr(k, d);
            }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();

    BatchOptimizeSettings settings;
    settings.n_iterations = 1;
    auto lbfgs = allocate_batch_lbfgs_state(n_batch, mol.n_indep * 6, 5);
    batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
    Kokkos::fence();

    auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
    Kokkos::deep_copy(h_E, bws.geom.energies);

    for (int b = 0; b < n_batch; ++b) {
        auto h_t = Kokkos::create_mirror_view(mol.translations);
        auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_t(k, d) = h_bt(b, k, d);
                h_r(k, d) = h_br(b, k, d);
            }
        Kokkos::deep_copy(mol.translations, h_t);
        Kokkos::deep_copy(mol.rot_vectors, h_r);

        RealType E_single = batch_evaluate_energy(mol, ws, ws.max_multipole_rank);

        // Tolerance for FP reordering in reductions
        REQUIRE_THAT(h_E(b), WithinAbs(E_single, 0.1 / AU_TO_KJ_PER_MOL));
    }
}

TEST_CASE("Batch optimizer reduces energy", "[batch]") {
    auto _result = make_nacl_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    setup_ewald(mol, ws);

    int n_batch = 4;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);

    for (int b = 0; b < n_batch; ++b)
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_bt(b, k, d) = h_mt(k, d) + (b + 1) * 0.005;
                h_br(b, k, d) = h_mr(k, d);
            }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();
    BatchOptimizeSettings settings;
    settings.n_iterations = 1;
    batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
    Kokkos::fence();

    const auto h_E0 = testing::host_snapshot(bws.geom.energies);

    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    settings.n_iterations = 10;
    auto result = batch_optimize_asymunit<0>(bws, settings);

    auto h_Ef = Kokkos::create_mirror_view(result.final_energies);
    Kokkos::deep_copy(h_Ef, result.final_energies);

    for (int b = 0; b < n_batch; ++b) {
        REQUIRE(h_Ef(b) <= h_E0(b) + 0.1 / AU_TO_KJ_PER_MOL);  // energy should decrease (tolerance for FP reordering on GPU)
    }
}

TEST_CASE("Batch expand MaxRank4 matches single", "[batch]") {
    // Dimer with rank-4 multipoles and n_symops=2, like urea P1
    auto mol = allocate_molecule_data(2, 2, 2, 4);
    auto ws  = allocate_workspace(64, 30000, 50000, 2, 4);

    RealType a = 5.6 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90; mol.cell_params[4] = 90; mol.cell_params[5] = 90;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0,0) = 0.25; h_t(0,1) = 0.25; h_t(0,2) = 0.25;
    h_t(1,0) = 0.75; h_t(1,1) = 0.75; h_t(1,2) = 0.75;
    Kokkos::deep_copy(mol.translations, h_t);
    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0,0)=0; h_r(0,1)=0; h_r(0,2)=0;
    h_r(1,0)=0.1; h_r(1,1)=0; h_r(1,2)=0;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0,0)=-0.5 * ANGSTROM_TO_BOHR; h_b(0,1)=0; h_b(0,2)=0;
    h_b(1,0)= 0.5 * ANGSTROM_TO_BOHR; h_b(1,1)=0; h_b(1,2)=0;
    h_b(2,0)=-0.5 * ANGSTROM_TO_BOHR; h_b(2,1)=0; h_b(2,2)=0;
    h_b(3,0)= 0.5 * ANGSTROM_TO_BOHR; h_b(3,1)=0; h_b(3,2)=0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0)=1; h_q(1)=-1; h_q(2)=0.5; h_q(3)=-0.5;
    Kokkos::deep_copy(mol.body_charges, h_q);
    auto h_bt2 = Kokkos::create_mirror_view(mol.body_types);
    h_bt2(0)=0; h_bt2(1)=1; h_bt2(2)=0; h_bt2(3)=1;
    Kokkos::deep_copy(mol.body_types, h_bt2);

    auto h_bmp = Kokkos::create_mirror_view(mol.body_multipoles);
    for (int i=0; i<4; ++i) for (int c=0; c<35; ++c) h_bmp(i,c) = 0;
    h_bmp(0,0)=1; h_bmp(1,0)=-1; h_bmp(2,0)=0.5; h_bmp(3,0)=-0.5;
    h_bmp(0,1)=0.02; h_bmp(1,1)=-0.01; h_bmp(0,3)=0.01;
    h_bmp(2,1)=0.015; h_bmp(3,2)=-0.01;
    h_bmp(0,4)=0.005; h_bmp(1,7)=0.003;
    Kokkos::deep_copy(mol.body_multipoles, h_bmp);

    // 2 symops: identity for mol 0, identity for mol 1
    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    for (int s=0; s<2; ++s) {
        for (int i=0; i<9; ++i) h_sr(s,i)=0;
        h_sr(s,0)=1; h_sr(s,4)=1; h_sr(s,8)=1;
    }
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    for (int s=0; s<2; ++s) for (int d=0; d<3; ++d) h_st(s,d)=0;
    Kokkos::deep_copy(mol.sym_translations, h_st);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0)=0; h_si(1)=1;
    Kokkos::deep_copy(mol.sym_indep_index, h_si);
    mol.n_indep=2; mol.n_symops=2; mol.n_sites_per_mol=2;

    ws.buck_cutoff=12.0 * ANGSTROM_TO_BOHR; ws.n_types=2;
    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B2 = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    for (int i=0;i<2;++i) for (int j=0;j<2;++j) h_A(i,j)=h_B2(i,j)=h_C(i,j)=0;
    h_A(0,1)=1000.0 / AU_TO_KJ_PER_MOL; h_A(1,0)=1000.0 / AU_TO_KJ_PER_MOL;
    h_B2(0,1)=3.0 * BOHR_TO_ANGSTROM; h_B2(1,0)=3.0 * BOHR_TO_ANGSTROM;
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B2);
    Kokkos::deep_copy(ws.buck_C, h_C);

    expand_asymmetric_unit<4>(mol, ws);
    RealType cell[3][3];
    cell_params_to_matrix(mol.cell_params, cell);
    compute_ewald_params(cell, ws.n_atoms, 1e-8, ws);
    RealType cutoff = (ws.rcut > ws.buck_cutoff) ? ws.rcut : ws.buck_cutoff;
    fill_shifts(cell, cutoff, ws);

    RealType E_single = batch_evaluate_energy(mol, ws, ws.max_multipole_rank);

    auto bws = allocate_batch_workspace(1, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_btt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_brr = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    Kokkos::deep_copy(h_t, mol.translations);
    Kokkos::deep_copy(h_r, mol.rot_vectors);
    for (int k=0; k<mol.n_indep; ++k) for (int d=0; d<3; ++d) {
        h_btt(0,k,d)=h_t(k,d); h_brr(0,k,d)=h_r(k,d);
    }
    Kokkos::deep_copy(bws.dofs.translations, h_btt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_brr);

    batch_expand_asymmetric_unit<4>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();

    auto h_bpos = Kokkos::create_mirror_view(bws.geom.positions);
    auto h_spos = Kokkos::create_mirror_view(ws.positions);
    Kokkos::deep_copy(h_bpos, bws.geom.positions);
    // Re-expand to reset positions
    expand_asymmetric_unit<4>(mol, ws);
    Kokkos::deep_copy(h_spos, ws.positions);
    for (int a=0; a<ws.n_atoms; ++a)
        for (int d=0; d<3; ++d)
            REQUIRE_THAT(h_bpos(0,a,d), WithinAbs(h_spos(a,d), RealType(1e-4)));

    auto h_bmp2 = Kokkos::create_mirror_view(bws.geom.multipoles);
    auto h_smp = Kokkos::create_mirror_view(ws.multipoles);
    Kokkos::deep_copy(h_bmp2, bws.geom.multipoles);
    Kokkos::deep_copy(h_smp, ws.multipoles);
    for (int a=0; a<ws.n_atoms; ++a)
        for (int c=0; c<35; ++c)
            REQUIRE_THAT(h_bmp2(0,a,c), WithinAbs(h_smp(a,c), RealType(1e-4)));

    BatchOptimizeSettings bsettings;
    auto lbfgs = allocate_batch_lbfgs_state(1, mol.n_indep*6, 5);
    batch_energy_gradient_asymunit<4>(bws.tables, bws.geom, bws.dofs, bsettings, 0);
    Kokkos::fence();

    auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
    Kokkos::deep_copy(h_E, bws.geom.energies);

    printf("  MaxRank=4 n_indep=2: Single E=%.6f, Batch E=%.6f, diff=%.6f\n",
           E_single, h_E(0), h_E(0)-E_single);
    REQUIRE_THAT(h_E(0), WithinAbs(E_single, 0.5 / AU_TO_KJ_PER_MOL));
}

TEST_CASE("Batch energy with multipoles matches single", "[batch]") {
    auto mol = allocate_molecule_data(1, 2, 1, 4);  // max_rank=4
    auto ws  = allocate_workspace(64, 30000, 50000, 2, 4);  // max_rank=4

    RealType a = 5.6402 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.25; h_t(0, 1) = 0.25; h_t(0, 2) = 0.25;
    Kokkos::deep_copy(mol.translations, h_t);
    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.0; h_r(0, 1) = 0.0; h_r(0, 2) = 0.0;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0, 0) = -a/4.0; h_b(0, 1) = 0.0; h_b(0, 2) = 0.0;
    h_b(1, 0) =  a/4.0; h_b(1, 1) = 0.0; h_b(1, 2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 1.0; h_q(1) = -1.0;
    Kokkos::deep_copy(mol.body_charges, h_q);
    auto h_bt2 = Kokkos::create_mirror_view(mol.body_types);
    h_bt2(0) = 0; h_bt2(1) = 1;
    Kokkos::deep_copy(mol.body_types, h_bt2);

    // Body multipoles: charge + small dipole
    auto h_bmp = Kokkos::create_mirror_view(mol.body_multipoles);
    for (int i = 0; i < 2; ++i)
        for (int c = 0; c < 35; ++c)
            h_bmp(i, c) = 0.0;
    h_bmp(0, 0) = 1.0;   h_bmp(1, 0) = -1.0;  // charges
    h_bmp(0, 1) = 0.01;  h_bmp(1, 1) = -0.01; // x-dipole
    Kokkos::deep_copy(mol.body_multipoles, h_bmp);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    h_sr(0, 0) = 1; h_sr(0, 1) = 0; h_sr(0, 2) = 0;
    h_sr(0, 3) = 0; h_sr(0, 4) = 1; h_sr(0, 5) = 0;
    h_sr(0, 6) = 0; h_sr(0, 7) = 0; h_sr(0, 8) = 1;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    h_st(0, 0) = 0; h_st(0, 1) = 0; h_st(0, 2) = 0;
    Kokkos::deep_copy(mol.sym_translations, h_st);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0) = 0;
    Kokkos::deep_copy(mol.sym_indep_index, h_si);
    mol.n_indep = 1; mol.n_symops = 1; mol.n_sites_per_mol = 2;

    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B2 = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        h_A(i,j) = h_B2(i,j) = h_C(i,j) = 0.0;
    h_A(0,1) = 1000.0 / AU_TO_KJ_PER_MOL; h_A(1,0) = 1000.0 / AU_TO_KJ_PER_MOL;
    h_B2(0,1) = 3.0 * BOHR_TO_ANGSTROM;   h_B2(1,0) = 3.0 * BOHR_TO_ANGSTROM;
    h_C(0,1) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_C(1,0) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B2);
    Kokkos::deep_copy(ws.buck_C, h_C);
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR; ws.n_types = 2;

    setup_ewald(mol, ws);

    expand_asymmetric_unit<4>(mol, ws);

    RealType E_single = batch_evaluate_energy(mol, ws, ws.max_multipole_rank);

    int n_batch = 1;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_bt(0, k, d) = h_mt(k, d);
            h_br(0, k, d) = h_mr(k, d);
        }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<4>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();

    BatchOptimizeSettings bsettings;
    bsettings.n_iterations = 1;
    auto lbfgs = allocate_batch_lbfgs_state(n_batch, mol.n_indep * 6, 5);
    batch_energy_gradient_asymunit<4>(bws.tables, bws.geom, bws.dofs, bsettings, 0);
    Kokkos::fence();

    auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
    Kokkos::deep_copy(h_E, bws.geom.energies);

    printf("  Single E = %.6f, Batch E = %.6f, diff = %.6f\n",
           E_single, h_E(0), h_E(0) - E_single);
    REQUIRE_THAT(h_E(0), WithinAbs(E_single, 0.5 / AU_TO_KJ_PER_MOL));
}

// P-1 dimer: n_indep=1, n_symops=2 (identity + inversion), n_sites=2
static std::pair<MoleculeData, Workspace> make_p1bar_dimer() {
    auto mol = allocate_molecule_data(1, 2, 2);  // n_indep=1, n_sites=2, n_symops=2
    auto ws  = allocate_workspace(64, 30000, 50000, 2);

    RealType a = 5.6 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.15; h_t(0, 1) = 0.20; h_t(0, 2) = 0.25;
    Kokkos::deep_copy(mol.translations, h_t);

    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.1; h_r(0, 1) = -0.05; h_r(0, 2) = 0.08;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0, 0) = -0.6 * ANGSTROM_TO_BOHR; h_b(0, 1) = 0.0; h_b(0, 2) = 0.0;
    h_b(1, 0) =  0.6 * ANGSTROM_TO_BOHR; h_b(1, 1) = 0.0; h_b(1, 2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 0.8; h_q(1) = -0.8;
    Kokkos::deep_copy(mol.body_charges, h_q);

    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    h_bt(0) = 0; h_bt(1) = 1;
    Kokkos::deep_copy(mol.body_types, h_bt);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    for (int i = 0; i < 9; ++i) h_sr(0, i) = 0;
    h_sr(0, 0) = 1; h_sr(0, 4) = 1; h_sr(0, 8) = 1;
    // Symop 1: inversion
    for (int i = 0; i < 9; ++i) h_sr(1, i) = 0;
    h_sr(1, 0) = -1; h_sr(1, 4) = -1; h_sr(1, 8) = -1;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);

    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    for (int s = 0; s < 2; ++s)
        for (int d = 0; d < 3; ++d)
            h_st(s, d) = 0;
    Kokkos::deep_copy(mol.sym_translations, h_st);

    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0) = 0; h_si(1) = 0;  // both map to independent molecule 0
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = 1;
    mol.n_symops = 2;
    mol.n_sites_per_mol = 2;

    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            h_A(i,j) = h_B(i,j) = h_C(i,j) = 0.0;
    h_A(0,1) = 1000.0 / AU_TO_KJ_PER_MOL; h_A(1,0) = 1000.0 / AU_TO_KJ_PER_MOL;
    h_B(0,1) = 3.0 * BOHR_TO_ANGSTROM;    h_B(1,0) = 3.0 * BOHR_TO_ANGSTROM;
    h_C(0,1) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_C(1,0) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    // Same-type interactions so self-shift pairs see Buckingham forces
    h_A(0,0) = 500.0 / AU_TO_KJ_PER_MOL;
    h_B(0,0) = 2.5 * BOHR_TO_ANGSTROM;
    h_C(0,0) = 5.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_A(1,1) = 500.0 / AU_TO_KJ_PER_MOL;
    h_B(1,1) = 2.5 * BOHR_TO_ANGSTROM;
    h_C(1,1) = 5.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B);
    Kokkos::deep_copy(ws.buck_C, h_C);
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR;
    ws.n_types = 2;

    return {mol, ws};
}

TEST_CASE("Asymunit gradient matches finite difference (P-1 dimer)", "[batch][gradient]") {
    auto _result = make_p1bar_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    expand_asymmetric_unit<0>(mol, ws);
    RealType cell[3][3];
    cell_params_to_matrix(mol.cell_params, cell);
    compute_ewald_params(cell, ws.n_atoms, 1e-8, ws);

    int n_batch = 2;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);

    for (int b = 0; b < n_batch; ++b)
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_bt(b, k, d) = h_mt(k, d) + b * 0.01;
                h_br(b, k, d) = h_mr(k, d) + b * 0.005;
            }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();
    BatchOptimizeSettings settings;
    settings.n_iterations = 1;
    auto lbfgs_mp = allocate_batch_lbfgs_state(n_batch, mol.n_indep * 6, 5);
    batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
    Kokkos::fence();

    for (int b = 0; b < n_batch; ++b) {
        INFO("batch entry " << b);
        std::printf("  P-1 dimer, batch entry %d:\n", b);
        REQUIRE(testing::worst_dof_gradient_mismatch<0>(bws, settings, b, 1e-5) < 1e-6);
    }
}

template<typename F>
static double bench_ms(F&& fn, int n_repeats) {
    Kokkos::fence();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeats; ++i) fn();
    Kokkos::fence();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / n_repeats;
}

// Z'=1 P1 crystal of n_symops molecules (identity symops, spread translations).
static std::pair<MoleculeData, Workspace> make_high_z_crystal(int n_symops) {
    int n_sites = 4;  // 4 atoms per molecule
    auto mol = allocate_molecule_data(1, n_sites, n_symops);
    auto ws  = allocate_workspace(n_symops * n_sites, 50000, 100000, 2);

    RealType a = 8.0 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.1; h_t(0, 1) = 0.15; h_t(0, 2) = 0.2;
    Kokkos::deep_copy(mol.translations, h_t);

    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.05; h_r(0, 1) = -0.03; h_r(0, 2) = 0.04;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0,0) = -0.5 * ANGSTROM_TO_BOHR; h_b(0,1) =  0.0; h_b(0,2) = 0.0;
    h_b(1,0) =  0.5 * ANGSTROM_TO_BOHR; h_b(1,1) =  0.0; h_b(1,2) = 0.0;
    h_b(2,0) =  0.0; h_b(2,1) = -0.5 * ANGSTROM_TO_BOHR; h_b(2,2) = 0.0;
    h_b(3,0) =  0.0; h_b(3,1) =  0.5 * ANGSTROM_TO_BOHR; h_b(3,2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 0.5; h_q(1) = -0.5; h_q(2) = 0.3; h_q(3) = -0.3;
    Kokkos::deep_copy(mol.body_charges, h_q);

    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    h_bt(0) = 0; h_bt(1) = 1; h_bt(2) = 0; h_bt(3) = 1;
    Kokkos::deep_copy(mol.body_types, h_bt);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    for (int s = 0; s < n_symops; ++s) {
        for (int i = 0; i < 9; ++i) h_sr(s, i) = 0;
        h_sr(s, 0) = 1; h_sr(s, 4) = 1; h_sr(s, 8) = 1;
        h_st(s, 0) = RealType(s) / RealType(n_symops);
        h_st(s, 1) = RealType((s * 3) % n_symops) / RealType(n_symops);
        h_st(s, 2) = RealType((s * 7) % n_symops) / RealType(n_symops);
        h_si(s) = 0;  // all map to independent molecule 0
    }
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    Kokkos::deep_copy(mol.sym_translations, h_st);
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = 1;
    mol.n_symops = n_symops;
    mol.n_sites_per_mol = n_sites;

    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B2 = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            h_A(i,j) = 500.0 / AU_TO_KJ_PER_MOL;
            h_B2(i,j) = 2.5 * BOHR_TO_ANGSTROM;
            h_C(i,j) = 5.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
        }
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B2);
    Kokkos::deep_copy(ws.buck_C, h_C);
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR;
    ws.n_types = 2;

    return {mol, ws};
}

TEST_CASE("Asymunit gradient matches finite difference at higher symmetry", "[batch][gradient]") {
    for (int n_symops : {2, 4, 8, 18}) {
        auto _result = make_high_z_crystal(n_symops); auto& mol = _result.first; auto& ws = _result.second;
        expand_asymmetric_unit<0>(mol, ws);
        RealType cell[3][3];
        cell_params_to_matrix(mol.cell_params, cell);
        compute_ewald_params(cell, ws.n_atoms, 1e-8, ws);

        int n_batch = 4;
        auto bws = allocate_batch_workspace(n_batch, ws, mol);
        setup_batch_cells(bws, mol, 1e-8);

        auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
        auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
        auto h_mt = Kokkos::create_mirror_view(mol.translations);
        auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
        Kokkos::deep_copy(h_mt, mol.translations);
        Kokkos::deep_copy(h_mr, mol.rot_vectors);
        for (int b = 0; b < n_batch; ++b)
            for (int k = 0; k < mol.n_indep; ++k)
                for (int d = 0; d < 3; ++d) {
                    h_bt(b,k,d) = h_mt(k,d) + b*0.01;
                    h_br(b,k,d) = h_mr(k,d) + b*0.005;
                }
        Kokkos::deep_copy(bws.dofs.translations, h_bt);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

        BatchOptimizeSettings settings;
        settings.n_iterations = 1;

        batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
        Kokkos::fence();
        auto lbfgs_mp = allocate_batch_lbfgs_state(n_batch, mol.n_indep*6, 5);
        batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
        Kokkos::fence();

        for (int b = 0; b < n_batch; ++b) {
            INFO("n_symops=" << n_symops << " batch=" << b);
            std::printf("  n_symops=%d, batch entry %d:\n", n_symops, b);
            REQUIRE(testing::worst_dof_gradient_mismatch<0>(bws, settings, b, 1e-5) < 1e-6);
        }
    }
}


TEST_CASE("Asymunit batch energy matches single-structure energy", "[batch]") {
    auto _result = make_p1bar_dimer(); auto& mol = _result.first; auto& ws = _result.second;
    setup_ewald(mol, ws);

    RealType E_single = batch_evaluate_energy(mol, ws, ws.max_multipole_rank);

    int n_batch = 1;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_bt(0, k, d) = h_mt(k, d);
            h_br(0, k, d) = h_mr(k, d);
        }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();
    BatchOptimizeSettings bsettings;
    bsettings.n_iterations = 1;
    auto lbfgs = allocate_batch_lbfgs_state(n_batch, mol.n_indep * 6, 5);
    batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, bsettings, 0);
    Kokkos::fence();

    auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
    Kokkos::deep_copy(h_E, bws.geom.energies);

    printf("  P-1 (n_symops=2) batch vs single: batch=%.6f single=%.6f diff=%.6f\n",
           h_E(0), E_single, h_E(0) - E_single);
    REQUIRE_THAT(h_E(0), WithinAbs(E_single, 0.5 / AU_TO_KJ_PER_MOL));
}

TEST_CASE("Asymunit batch energy matches single for higher Z", "[batch]") {
    for (int n_symops : {4, 8, 18}) {
        auto _result = make_high_z_crystal(n_symops); auto& mol = _result.first; auto& ws = _result.second;
        expand_asymmetric_unit<0>(mol, ws);
        RealType cell[3][3];
        cell_params_to_matrix(mol.cell_params, cell);
        compute_ewald_params(cell, ws.n_atoms, 1e-8, ws);
        RealType cutoff = (ws.rcut > ws.buck_cutoff) ? ws.rcut : ws.buck_cutoff;
        fill_shifts(cell, cutoff, ws);

        // Use the same Buckingham damping as BatchOptimizeSettings' default; the
        // undamped dispersion runs away at short contacts.
        RealType E_single = batch_evaluate_energy(mol, ws, ws.max_multipole_rank,
                                                  /*taper_on=*/0, /*taper_off=*/0,
                                                  TangToenniesDamping{});

        auto bws = allocate_batch_workspace(1, ws, mol);
        setup_batch_cells(bws, mol, 1e-8);
        auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
        auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
        auto h_mt = Kokkos::create_mirror_view(mol.translations);
        auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
        Kokkos::deep_copy(h_mt, mol.translations);
        Kokkos::deep_copy(h_mr, mol.rot_vectors);
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_bt(0,k,d) = h_mt(k,d);
                h_br(0,k,d) = h_mr(k,d);
            }
        Kokkos::deep_copy(bws.dofs.translations, h_bt);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

        batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
        Kokkos::fence();
        BatchOptimizeSettings bsettings;
        bsettings.n_iterations = 1;
        auto lbfgs = allocate_batch_lbfgs_state(1, mol.n_indep*6, 5);
        batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, bsettings, 0);
        Kokkos::fence();

        auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
        Kokkos::deep_copy(h_E, bws.geom.energies);

        printf("  n_symops=%2d: batch=%.4f single=%.4f diff=%.4f ratio=%.4f\n",
               n_symops, h_E(0), E_single, h_E(0) - E_single,
               E_single != 0 ? h_E(0) / E_single : 0.0);
        REQUIRE_THAT(h_E(0), WithinAbs(E_single, 1.0 / AU_TO_KJ_PER_MOL));
    }
}

TEST_CASE("Asymunit gradient matches finite difference with multipoles", "[batch][gradient]") {
    // P-1 crystal (n_symops=2, n_indep=1) with rank-4 multipoles
    auto mol = allocate_molecule_data(1, 2, 2, 4);
    auto ws  = allocate_workspace(64, 30000, 50000, 2, 4);

    RealType a = 5.6 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.15; h_t(0, 1) = 0.20; h_t(0, 2) = 0.25;
    Kokkos::deep_copy(mol.translations, h_t);
    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.1; h_r(0, 1) = -0.05; h_r(0, 2) = 0.08;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0, 0) = -0.6 * ANGSTROM_TO_BOHR; h_b(0, 1) = 0.0; h_b(0, 2) = 0.0;
    h_b(1, 0) =  0.6 * ANGSTROM_TO_BOHR; h_b(1, 1) = 0.0; h_b(1, 2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 0.8; h_q(1) = -0.8;
    Kokkos::deep_copy(mol.body_charges, h_q);
    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    h_bt(0) = 0; h_bt(1) = 1;
    Kokkos::deep_copy(mol.body_types, h_bt);

    auto h_bmp = Kokkos::create_mirror_view(mol.body_multipoles);
    for (int i = 0; i < 2; ++i)
        for (int c = 0; c < 35; ++c)
            h_bmp(i, c) = 0.0;
    h_bmp(0, 0) =  0.8;  h_bmp(1, 0) = -0.8;   // charges
    h_bmp(0, 1) =  0.02; h_bmp(1, 1) = -0.015;  // x-dipole
    h_bmp(0, 2) =  0.01; h_bmp(1, 3) = -0.008;  // y-dipole on 0, z-dipole on 1
    h_bmp(0, 4) =  0.005; h_bmp(1, 7) = 0.003;  // quadrupole components
    Kokkos::deep_copy(mol.body_multipoles, h_bmp);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    for (int i = 0; i < 9; ++i) h_sr(0, i) = 0;
    h_sr(0, 0) = 1; h_sr(0, 4) = 1; h_sr(0, 8) = 1;
    for (int i = 0; i < 9; ++i) h_sr(1, i) = 0;
    h_sr(1, 0) = -1; h_sr(1, 4) = -1; h_sr(1, 8) = -1;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);

    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    for (int s = 0; s < 2; ++s)
        for (int d = 0; d < 3; ++d) h_st(s, d) = 0;
    Kokkos::deep_copy(mol.sym_translations, h_st);

    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0) = 0; h_si(1) = 0;
    Kokkos::deep_copy(mol.sym_indep_index, h_si);
    mol.n_indep = 1; mol.n_symops = 2; mol.n_sites_per_mol = 2;

    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B2 = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        h_A(i,j) = h_B2(i,j) = h_C(i,j) = 0.0;
    h_A(0,1) = 1000.0 / AU_TO_KJ_PER_MOL; h_A(1,0) = 1000.0 / AU_TO_KJ_PER_MOL;
    h_B2(0,1) = 3.0 * BOHR_TO_ANGSTROM;   h_B2(1,0) = 3.0 * BOHR_TO_ANGSTROM;
    h_C(0,1) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_C(1,0) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_A(0,0) = 500.0 / AU_TO_KJ_PER_MOL;
    h_B2(0,0) = 2.5 * BOHR_TO_ANGSTROM;
    h_C(0,0) = 5.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    h_A(1,1) = 500.0 / AU_TO_KJ_PER_MOL;
    h_B2(1,1) = 2.5 * BOHR_TO_ANGSTROM;
    h_C(1,1) = 5.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B2);
    Kokkos::deep_copy(ws.buck_C, h_C);
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR; ws.n_types = 2;

    expand_asymmetric_unit<4>(mol, ws);
    RealType cell[3][3];
    cell_params_to_matrix(mol.cell_params, cell);
    compute_ewald_params(cell, ws.n_atoms, 1e-8, ws);

    int n_batch = 2;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-8);

    auto h_btt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_brr = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);
    for (int b = 0; b < n_batch; ++b)
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_btt(b, k, d) = h_mt(k, d) + b * 0.01;
                h_brr(b, k, d) = h_mr(k, d) + b * 0.005;
            }
    Kokkos::deep_copy(bws.dofs.translations, h_btt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_brr);

    batch_expand_asymmetric_unit<4>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();
    BatchOptimizeSettings settings;
    settings.n_iterations = 1;
    auto lbfgs_mp = allocate_batch_lbfgs_state(n_batch, mol.n_indep * 6, 5);
    batch_energy_gradient_asymunit<4>(bws.tables, bws.geom, bws.dofs, settings, 0);
    Kokkos::fence();

    for (int b = 0; b < n_batch; ++b) {
        INFO("batch entry " << b);
        std::printf("  rank-4 multipoles, batch entry %d:\n", b);
        REQUIRE(testing::worst_dof_gradient_mismatch<4>(bws, settings, b, 1e-5) < 1e-6);
    }

    // The same check with the rank >= 2 term cut short (--mp-cutoff).
    settings.model.mp_cutoff = RealType(10.0 * ANGSTROM_TO_BOHR);
    batch_energy_gradient_asymunit<4>(bws.tables, bws.geom, bws.dofs, settings, 0);
    Kokkos::fence();
    for (int b = 0; b < n_batch; ++b) {
        INFO("batch entry " << b << " with a 10 A multipole cutoff");
        std::printf("  rank-4 multipoles, 10 A multipole cutoff, batch entry %d:\n", b);
        REQUIRE(testing::worst_dof_gradient_mismatch<4>(bws, settings, b, 1e-5) < 1e-6);
    }
}

TEST_CASE("Asymunit kernel timing benchmark", "[.bench]") {
    for (int n_symops : {2, 4, 8, 18}) {
        auto _result = make_high_z_crystal(n_symops); auto& mol = _result.first; auto& ws = _result.second;
        expand_asymmetric_unit<0>(mol, ws);
        RealType cell[3][3];
        cell_params_to_matrix(mol.cell_params, cell);
        compute_ewald_params(cell, ws.n_atoms, 1e-8, ws);

        int n_batch = 4;
        auto bws = allocate_batch_workspace(n_batch, ws, mol);
        setup_batch_cells(bws, mol, 1e-8);

        auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
        auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
        auto h_mt = Kokkos::create_mirror_view(mol.translations);
        auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
        Kokkos::deep_copy(h_mt, mol.translations);
        Kokkos::deep_copy(h_mr, mol.rot_vectors);
        for (int b = 0; b < n_batch; ++b)
            for (int k = 0; k < mol.n_indep; ++k)
                for (int d = 0; d < 3; ++d) {
                    h_bt(b,k,d) = h_mt(k,d) + b*0.003;
                    h_br(b,k,d) = h_mr(k,d);
                }
        Kokkos::deep_copy(bws.dofs.translations, h_bt);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

        BatchOptimizeSettings settings;
        settings.n_iterations = 1;
        int n_atoms = n_symops * mol.n_sites_per_mol;
        int n_mol_upper = n_symops * (n_symops + 1) / 2;
        int n_repeats = 20;

        batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
        Kokkos::fence();
        auto lbfgs_mp = allocate_batch_lbfgs_state(n_batch, mol.n_indep*6, 5);
        batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
        Kokkos::fence();

        Kokkos::deep_copy(bws.dofs.translations, h_bt);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);
        double t_mp = bench_ms([&]{
            batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
            Kokkos::fence();
            batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
        }, n_repeats);

        Kokkos::deep_copy(bws.dofs.translations, h_bt);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);
        batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
        Kokkos::fence();
        auto lbfgs_au = allocate_batch_lbfgs_state(n_batch, mol.n_indep*6, 5);
        batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
        Kokkos::fence();

        Kokkos::deep_copy(bws.dofs.translations, h_bt);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);
        double t_au = bench_ms([&]{
            batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
            Kokkos::fence();
            batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, settings, 0);
        }, n_repeats);

        printf("  n_symops=%2d  n_atoms=%3d  mol_upper=%4d  "
               "eval1=%.2f ms  asymunit=%.2f ms  speedup=%.1fx\n",
               n_symops, n_atoms, n_mol_upper, t_mp, t_au, t_mp / t_au);
    }
}

TEST_CASE("Gradient FD check: Buckingham-only P-1 dimer", "[gradient]") {
    // P-1: n_symops=2, Z'=1, 2 sites/mol, zero charges (Buckingham only).
    auto _result = make_nacl_dimer();
    auto& mol = _result.first;
    auto& ws = _result.second;

    auto h_bq = Kokkos::create_mirror_view(mol.body_charges);
    for (int i = 0; i < 2; ++i) h_bq(i) = 0.0;
    Kokkos::deep_copy(mol.body_charges, h_bq);

    // Buckingham params for all type pairs, including same-type
    {
        auto h_A = Kokkos::create_mirror_view(ws.buck_A);
        auto h_B = Kokkos::create_mirror_view(ws.buck_B);
        auto h_C = Kokkos::create_mirror_view(ws.buck_C);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                h_A(i,j) = 1000.0 / AU_TO_KJ_PER_MOL;
                h_B(i,j) = 3.0 * BOHR_TO_ANGSTROM;
                h_C(i,j) = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
            }
        Kokkos::deep_copy(ws.buck_A, h_A);
        Kokkos::deep_copy(ws.buck_B, h_B);
        Kokkos::deep_copy(ws.buck_C, h_C);
    }

    // Inversion symop, set before allocate_batch_workspace
    mol.n_symops = 2;
    ws.n_atoms = mol.n_symops * mol.n_sites_per_mol;  // 4
    mol.sym_rotations = decltype(mol.sym_rotations)("sym_rot", 2, 9);
    mol.sym_translations = decltype(mol.sym_translations)("sym_trans", 2, 3);
    mol.sym_indep_index = decltype(mol.sym_indep_index)("sym_idx", 2);
    {
        auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
        h_sr(0,0)=1; h_sr(0,1)=0; h_sr(0,2)=0;
        h_sr(0,3)=0; h_sr(0,4)=1; h_sr(0,5)=0;
        h_sr(0,6)=0; h_sr(0,7)=0; h_sr(0,8)=1;
        h_sr(1,0)=-1; h_sr(1,1)=0;  h_sr(1,2)=0;
        h_sr(1,3)=0;  h_sr(1,4)=-1; h_sr(1,5)=0;
        h_sr(1,6)=0;  h_sr(1,7)=0;  h_sr(1,8)=-1;
        Kokkos::deep_copy(mol.sym_rotations, h_sr);
        auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
        h_st(0,0)=0; h_st(0,1)=0; h_st(0,2)=0;
        h_st(1,0)=0; h_st(1,1)=0; h_st(1,2)=0;
        Kokkos::deep_copy(mol.sym_translations, h_st);
        auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
        h_si(0) = 0; h_si(1) = 0;
        Kokkos::deep_copy(mol.sym_indep_index, h_si);
    }

    auto bws = allocate_batch_workspace(1, ws, mol);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_bt(0, k, d) = h_mt(k, d);
            h_br(0, k, d) = h_mr(k, d);
        }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);

    bws.dofs.n_cell_dof = 0;
    // Cell strain views are needed even if unused
    bws.dofs.cell_ref = decltype(bws.dofs.cell_ref)("cell_ref", 1, 3, 3);
    bws.dofs.strain = decltype(bws.dofs.strain)("strain", 1, 6);
    bws.dofs.d_g_strain = decltype(bws.dofs.d_g_strain)("d_g_strain", 1, 6);
    Kokkos::deep_copy(bws.dofs.cell_ref, bws.geom.cell);
    Kokkos::deep_copy(bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(bws.dofs.d_g_strain, RealType(0));

    const int n_dof = mol.n_indep * 6;
    BatchOptimizeSettings bsettings;
    bsettings.n_iterations = 1;
    bsettings.max_step = 0.05;
    bsettings.lbfgs_memory = 5;
    bsettings.model.taper_on = 0; bsettings.model.taper_off = 0;  // no taper
    bsettings.model.wolf = true;  // Wolf DSF — no reciprocal sum, avoids CUDA sum-order issues
    bsettings.energy_only = true;

    // Perturb away from the symmetric point to get non-zero gradients
    {
        auto h_bt_p = Kokkos::create_mirror_view(bws.dofs.translations);
        Kokkos::deep_copy(h_bt_p, bws.dofs.translations);
        h_bt_p(0, 0, 0) += 0.02;
        h_bt_p(0, 0, 1) += 0.03;
        h_bt_p(0, 0, 2) += 0.01;
        Kokkos::deep_copy(bws.dofs.translations, h_bt_p);
    }

    auto eval = [&]() -> RealType {
        batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
        Kokkos::fence();
        auto lbfgs2 = allocate_batch_lbfgs_state(1, n_dof, 5);
        batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, bsettings, 0);
        Kokkos::fence();
        auto hE = Kokkos::create_mirror_view(bws.geom.energies);
        Kokkos::deep_copy(hE, bws.geom.energies);
        return hE(0);
    };

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();
    auto lbfgs = allocate_batch_lbfgs_state(1, n_dof, 5);
    batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, bsettings, 0);
    Kokkos::fence();

    auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
    Kokkos::deep_copy(h_E, bws.geom.energies);
    RealType E0 = h_E(0);

    const auto h_dgt = testing::host_snapshot(bws.dofs.d_gt);
    const auto h_dgr = testing::host_snapshot(bws.dofs.d_gr);

    {
        auto h_pos = Kokkos::create_mirror_view(bws.geom.positions);
        Kokkos::deep_copy(h_pos, bws.geom.positions);
        printf("  n_atoms=%d n_symops=%d n_indep=%d n_sites=%d\n",
               bws.geom.n_atoms, bws.tables.n_symops, bws.tables.n_indep, bws.tables.n_sites_per_mol);
        for (int a = 0; a < bws.geom.n_atoms; ++a)
            printf("  pos[%d] = (%.4f, %.4f, %.4f)\n", a, h_pos(0,a,0), h_pos(0,a,1), h_pos(0,a,2));
    }
    printf("  E0 = %.10f Hartree\n", E0);
    printf("  d_gt = (%.6f, %.6f, %.6f)\n", h_dgt(0,0,0), h_dgt(0,0,1), h_dgt(0,0,2));
    printf("  d_gr = (%.6f, %.6f, %.6f)\n", h_dgr(0,0,0), h_dgr(0,0,1), h_dgr(0,0,2));

    // Real copies: a mirror of a host-space view aliases it.
    const auto h_bt_orig = testing::host_snapshot(bws.dofs.translations);
    const auto h_br_orig = testing::host_snapshot(bws.dofs.rot_vectors);

    const RealType eps = 1e-5;
    printf("  FD check (eps=%.0e):\n", eps);
    bool all_pass = true;
    for (int dof = 0; dof < 3; ++dof) {
        auto at = [&](double delta) -> RealType {
            auto h2 = testing::host_snapshot(h_bt_orig);
            h2(0, 0, dof) += static_cast<RealType>(delta);
            Kokkos::deep_copy(bws.dofs.translations, h2);
            Kokkos::deep_copy(bws.dofs.rot_vectors, h_br_orig);
            return eval();
        };
        RealType Ep = at(eps), Em = at(-eps);

        RealType fd = (Ep - Em) / (2.0 * eps);
        RealType anal = h_dgt(0, 0, dof);
        RealType scale = std::max({std::abs(anal), std::abs(fd), RealType(1e-10)});
        RealType rel = std::abs(anal - fd) / scale;
        // On an inversion centre the net force is zero, so only absolute
        // agreement is meaningful.
        bool ok = klasp::testing::fd_agrees(anal, fd);
        printf("    trans_%d: anal=%.8f fd=%.8f rel=%.2e %s\n",
               dof, anal, fd, rel, ok ? "OK" : "FAIL");
        if (!ok) all_pass = false;
    }
    Kokkos::deep_copy(bws.dofs.translations, h_bt_orig);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br_orig);
    REQUIRE(all_pass);
}

// Reference: occ/tests/descriptors_tests.cpp, "Face-centered cubic lattice".
// First-neighbour distance is sqrt(2)/2 in a 1 Å cell (12-atom first shell).
TEST_CASE("AMD matches occ PDD on FCC lattice", "[batch][amd]") {
    const int n_indep = 4;
    const int n_sites = 1;
    const int n_symops = 4;

    auto mol = allocate_molecule_data(n_indep, n_sites, n_symops);
    auto ws  = allocate_workspace(64, 30000, 50000, n_indep * n_sites);

    RealType a = 1.0 * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0,0)=0.0; h_t(0,1)=0.0; h_t(0,2)=0.0;
    h_t(1,0)=0.5; h_t(1,1)=0.5; h_t(1,2)=0.0;
    h_t(2,0)=0.5; h_t(2,1)=0.0; h_t(2,2)=0.5;
    h_t(3,0)=0.0; h_t(3,1)=0.5; h_t(3,2)=0.5;
    Kokkos::deep_copy(mol.translations, h_t);
    Kokkos::deep_copy(mol.rot_vectors, RealType(0));

    // One site per mol at its COM.
    Kokkos::deep_copy(mol.body_positions, RealType(0));
    Kokkos::deep_copy(mol.body_charges, RealType(0));
    Kokkos::deep_copy(mol.body_types, 0);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    for (int op = 0; op < n_symops; ++op) {
        for (int c = 0; c < 9; ++c) h_sr(op, c) = 0;
        h_sr(op, 0) = 1; h_sr(op, 4) = 1; h_sr(op, 8) = 1;
        for (int d = 0; d < 3; ++d) h_st(op, d) = 0;
        h_si(op) = op;  // each symop is the identity applied to indep mol `op`
    }
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    Kokkos::deep_copy(mol.sym_translations, h_st);
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = n_indep;
    mol.n_symops = n_symops;
    mol.n_sites_per_mol = n_sites;
    ws.n_types = 1;
    ws.buck_cutoff = 4.0 * ANGSTROM_TO_BOHR;  // widen max_shift enough to capture the first coordination shell

    int n_batch = 1;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-6);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    for (int k = 0; k < n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_bt(0, k, d) = h_t(k, d);
            h_br(0, k, d) = 0;
        }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();

    auto amd = compute_amd(bws.tables, bws.geom, AMD_K);

    REQUIRE(amd.vectors.size() == 1);
    REQUIRE(static_cast<int>(amd.vectors[0].size()) == AMD_K);

    // amd[0..11] cover the 12-neighbour first shell at sqrt(2)/2 Å.
    const double expected_first = std::sqrt(2.0) / 2.0;
    for (int i = 0; i < 12; ++i)
        REQUIRE_THAT(amd.vectors[0][i], WithinAbs(expected_first, 1e-6));

    for (int i = 1; i < AMD_K; ++i) {
        REQUIRE(std::isfinite(amd.vectors[0][i]));
        REQUIRE(amd.vectors[0][i] >= amd.vectors[0][i-1] - 1e-10);
    }
}

// First Bragg peak (100) at d = a, so 2θ = 2 asin(λ/(2a)). Two identical
// structures and one with a 10% larger cell.
TEST_CASE("Powder pattern first peak on primitive cubic", "[batch][powder]") {
    const int n_indep = 1;
    const int n_sites = 1;
    const int n_symops = 1;

    auto mol = allocate_molecule_data(n_indep, n_sites, n_symops);
    auto ws  = allocate_workspace(64, 30000, 50000, n_indep * n_sites);

    const double a_A = 5.0;  // Å
    RealType a = a_A * ANGSTROM_TO_BOHR;
    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0,0) = 0; h_t(0,1) = 0; h_t(0,2) = 0;
    Kokkos::deep_copy(mol.translations, h_t);
    Kokkos::deep_copy(mol.rot_vectors, RealType(0));
    Kokkos::deep_copy(mol.body_positions, RealType(0));
    Kokkos::deep_copy(mol.body_charges, RealType(0));
    Kokkos::deep_copy(mol.body_types, 0);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    for (int c = 0; c < 9; ++c) h_sr(0, c) = 0;
    h_sr(0, 0) = 1; h_sr(0, 4) = 1; h_sr(0, 8) = 1;
    for (int d = 0; d < 3; ++d) h_st(0, d) = 0;
    h_si(0) = 0;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    Kokkos::deep_copy(mol.sym_translations, h_st);
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = n_indep;
    mol.n_symops = n_symops;
    mol.n_sites_per_mol = n_sites;
    ws.n_types = 1;
    ws.buck_cutoff = 6.0 * ANGSTROM_TO_BOHR;

    const int n_batch = 3;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    setup_batch_cells(bws, mol, 1e-6);

    {
        auto h_cell = Kokkos::create_mirror_view(bws.geom.cell);
        auto h_cell_inv = Kokkos::create_mirror_view(bws.geom.cell_inv);
        Kokkos::deep_copy(h_cell, bws.geom.cell);
        Kokkos::deep_copy(h_cell_inv, bws.geom.cell_inv);
        const RealType scale = RealType(1.1);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                h_cell(2, i, j)     = h_cell(0, i, j) * scale;
                h_cell_inv(2, i, j) = h_cell_inv(0, i, j) / scale;
            }
        Kokkos::deep_copy(bws.geom.cell, h_cell);
        Kokkos::deep_copy(bws.geom.cell_inv, h_cell_inv);
    }

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    for (int b = 0; b < n_batch; ++b)
        for (int d = 0; d < 3; ++d) { h_bt(b, 0, d) = 0; h_br(b, 0, d) = 0; }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();

    std::vector<std::string> elements(bws.geom.n_atoms, "C");
    auto ff = build_powder_form_factors(elements);

    PowderSettings pset;
    pset.wavelength = 1.5406;
    pset.two_theta_min = 5.0;
    pset.two_theta_max = 50.0;
    pset.n_bins = 512;
    pset.peak_fwhm = 0.1;
    pset.max_hkl = 8;
    pset.lorentz_polarization = true;
    pset.normalization = PowderNormalization::L2;

    auto powder = compute_powder_pattern(bws.geom, ff, pset);

    REQUIRE(static_cast<int>(powder.patterns.size()) == n_batch);
    REQUIRE(powder.n_bins == pset.n_bins);
    REQUIRE(static_cast<int>(powder.two_theta.size()) == pset.n_bins);

    auto peak_bin = [](const std::vector<double>& p) {
        int idx = 0; double best = p[0];
        for (int i = 1; i < (int)p.size(); ++i)
            if (p[i] > best) { best = p[i]; idx = i; }
        return idx;
    };

    const double expected_2theta = 2.0 * std::asin(pset.wavelength / (2.0 * a_A))
                                   * 180.0 / M_PI;

    const int b0 = peak_bin(powder.patterns[0]);
    const int b1 = peak_bin(powder.patterns[1]);
    REQUIRE(b0 == b1);

    const double bin_step = (pset.two_theta_max - pset.two_theta_min) / pset.n_bins;
    REQUIRE_THAT(powder.two_theta[b0],
                 WithinAbs(expected_2theta, 1.5 * bin_step));

    for (int i = 0; i < pset.n_bins; ++i) {
        REQUIRE_THAT(powder.patterns[0][i],
                     WithinAbs(powder.patterns[1][i], 1e-8));
    }

    const int b2 = peak_bin(powder.patterns[2]);
    REQUIRE(b2 < b0);
    const double expected_2theta_scaled =
        2.0 * std::asin(pset.wavelength / (2.0 * 1.1 * a_A)) * 180.0 / M_PI;
    REQUIRE_THAT(powder.two_theta[b2],
                 WithinAbs(expected_2theta_scaled, 1.5 * bin_step));

    double norm_sq = 0;
    for (double v : powder.patterns[0]) norm_sq += v * v;
    REQUIRE_THAT(norm_sq, WithinAbs(1.0, 1e-6));

    for (double v : powder.patterns[0]) REQUIRE(v >= -1e-12);

    REQUIRE(powder_cosine_distance(powder.patterns[0], powder.patterns[1]) < 1e-10);
    REQUIRE(powder_cosine_distance(powder.patterns[0], powder.patterns[2]) > 0.1);
}

// Covering |g| <= g_max needs |h| <= |a| g_max, so the hkl range must grow with
// the cell. Tested as convergence: the default is converged, a clamp is not.
TEST_CASE("Powder hkl range follows the cell", "[batch][powder]") {
    const int n_indep = 1, n_sites = 1, n_symops = 1;
    auto mol = allocate_molecule_data(n_indep, n_sites, n_symops);
    auto ws  = allocate_workspace(64, 30000, 50000, n_indep * n_sites);

    // 25 A cube: d_min = 1.82 A at Cu Ka / 50 deg, so reflections run to h = 13.
    const double a_A = 25.0;
    RealType a = a_A * ANGSTROM_TO_BOHR;
    for (int i = 0; i < 3; ++i) mol.cell_params[i] = a;
    for (int i = 3; i < 6; ++i) mol.cell_params[i] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0,0) = 0; h_t(0,1) = 0; h_t(0,2) = 0;
    Kokkos::deep_copy(mol.translations, h_t);
    Kokkos::deep_copy(mol.rot_vectors, RealType(0));
    Kokkos::deep_copy(mol.body_positions, RealType(0));
    Kokkos::deep_copy(mol.body_charges, RealType(0));
    Kokkos::deep_copy(mol.body_types, 0);

    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    for (int c = 0; c < 9; ++c) h_sr(0, c) = 0;
    h_sr(0, 0) = 1; h_sr(0, 4) = 1; h_sr(0, 8) = 1;
    for (int d = 0; d < 3; ++d) h_st(0, d) = 0;
    h_si(0) = 0;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    Kokkos::deep_copy(mol.sym_translations, h_st);
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = n_indep;
    mol.n_symops = n_symops;
    mol.n_sites_per_mol = n_sites;
    ws.n_types = 1;
    ws.buck_cutoff = 6.0 * ANGSTROM_TO_BOHR;

    auto bws = allocate_batch_workspace(1, ws, mol);
    setup_batch_cells(bws, mol, 1e-6);
    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    for (int d = 0; d < 3; ++d) { h_bt(0, 0, d) = 0; h_br(0, 0, d) = 0; }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);
    batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
    Kokkos::fence();

    std::vector<std::string> elements(bws.geom.n_atoms, "C");
    auto ff = build_powder_form_factors(elements);

    PowderSettings pset;
    pset.n_bins = 512;
    pset.normalization = PowderNormalization::L2;

    PowderSettings truncated = pset;   // a range too small for this cell
    truncated.max_hkl = 12;
    PowderSettings wider = pset;
    wider.max_hkl = 40;                // beyond what this cell needs

    auto p_default   = compute_powder_pattern(bws.geom, ff, pset).patterns[0];
    auto p_truncated = compute_powder_pattern(bws.geom, ff, truncated).patterns[0];
    auto p_wider     = compute_powder_pattern(bws.geom, ff, wider).patterns[0];

    REQUIRE(powder_cosine_distance(p_default, p_wider) < 1e-12);

    REQUIRE(powder_cosine_distance(p_default, p_truncated) > 1e-6);

    // Truncation loses intensity in the raw (unnormalized) sum.
    PowderSettings raw = pset;
    raw.normalization = PowderNormalization::NONE;
    PowderSettings raw_truncated = raw;
    raw_truncated.max_hkl = 12;
    const auto r_default   = compute_powder_pattern(bws.geom, ff, raw).patterns[0];
    const auto r_truncated = compute_powder_pattern(bws.geom, ff, raw_truncated).patterns[0];
    const double sum_default =
        std::accumulate(r_default.begin(), r_default.end(), 0.0);
    const double sum_truncated =
        std::accumulate(r_truncated.begin(), r_truncated.end(), 0.0);
    REQUIRE(sum_default > 0.0);
    REQUIRE(sum_truncated < sum_default);
}

// --- Ewald rebracketing -----------------------------------------------------
//
// kappa is fixed while the cell moves and re-set between optimiser steps so the
// hkl box cannot grow with a runaway cell.

namespace {

// The state batch_update_cell_from_strain leaves after the cell grows by
// `vol_ratio` on a frozen kappa: hkl box grown to cover the same gmax.
struct GrownCell {
    RealType cell[3][3];
    int box_before[3];
};

GrownCell grow_cell_at_frozen_kappa(BatchWorkspace& bws, const RealType cp[6],
                                    RealType vol_ratio, const ForceField& ff) {
    GrownCell g{};
    const RealType scale = std::cbrt(vol_ratio);
    RealType base[3][3];
    cell_params_to_matrix(cp, base);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) g.cell[i][j] = base[i][j] * scale;

    auto h_cell = Kokkos::create_mirror_view(bws.geom.cell);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) h_cell(0, i, j) = g.cell[i][j];
    Kokkos::deep_copy(bws.geom.cell, h_cell);

    auto h_vol = Kokkos::create_mirror_view(bws.geom.volume);
    Kokkos::deep_copy(h_vol, bws.geom.volume);
    h_vol(0) *= vol_ratio;
    Kokkos::deep_copy(bws.geom.volume, h_vol);

    auto h_kappa = Kokkos::create_mirror_view(bws.geom.kappa);
    Kokkos::deep_copy(h_kappa, bws.geom.kappa);
    const RealType gmax = 2 * h_kappa(0) * ff.ewald_split;
    reciprocal_index_bounds(g.cell, gmax, g.box_before);
    auto h_mhkl = Kokkos::create_mirror_view(bws.geom.max_hkl);
    for (int i = 0; i < 3; ++i) h_mhkl(0, i) = g.box_before[i];
    Kokkos::deep_copy(bws.geom.max_hkl, h_mhkl);
    return g;
}

int box_count(const int hkl[3]) {
    return (2*hkl[0] + 1) * (2*hkl[1] + 1) * (2*hkl[2] + 1);
}

}  // namespace

TEST_CASE("a cell that has not moved is not rebracketed", "[batch][ewald]") {
    auto _r = make_nacl_dimer(); auto& mol = _r.first; auto& ws = _r.second;
    setup_ewald(mol, ws);
    auto bws = allocate_batch_workspace(1, ws, mol);
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);

    auto h_kappa0 = Kokkos::create_mirror_view(bws.geom.kappa);
    Kokkos::deep_copy(h_kappa0, bws.geom.kappa);
    const RealType kappa0 = h_kappa0(0);

    SView1D<int, ExecSpace> live;                     // empty -> every slot live
    SView1D<int, ExecSpace> flag("rebracketed", 1);
    batch_rebracket_ewald(bws.tables.ff, bws.geom, live, flag);

    auto h_flag = Kokkos::create_mirror_view(flag);
    Kokkos::deep_copy(h_flag, flag);
    auto h_kappa1 = Kokkos::create_mirror_view(bws.geom.kappa);
    Kokkos::deep_copy(h_kappa1, bws.geom.kappa);

    REQUIRE(h_flag(0) == 0);
    REQUIRE(h_kappa1(0) == kappa0);
}

TEST_CASE("a runaway cell is rebracketed and its hkl box comes back down",
          "[batch][ewald]") {
    auto _r = make_nacl_dimer(); auto& mol = _r.first; auto& ws = _r.second;
    setup_ewald(mol, ws);
    auto bws = allocate_batch_workspace(1, ws, mol);
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);

    auto h_box0 = Kokkos::create_mirror_view(bws.geom.max_hkl);
    Kokkos::deep_copy(h_box0, bws.geom.max_hkl);
    const int healthy[3] = {h_box0(0,0), h_box0(0,1), h_box0(0,2)};

    const RealType VOL_RATIO = 200.0;
    REQUIRE(VOL_RATIO > EWALD_REBRACKET_VOLUME_RATIO);
    auto grown = grow_cell_at_frozen_kappa(bws, mol.cell_params, VOL_RATIO,
                                           bws.tables.ff);

    // Frozen kappa blows up the box (less than the volume ratio at this size,
    // since reciprocal_index_bounds adds one to every axis).
    INFO("healthy " << box_count(healthy) << ", frozen-kappa "
                    << box_count(grown.box_before));
    REQUIRE(box_count(grown.box_before) > 10 * box_count(healthy));

    SView1D<int, ExecSpace> live;
    SView1D<int, ExecSpace> flag("rebracketed", 1);
    batch_rebracket_ewald(bws.tables.ff, bws.geom, live, flag);

    auto h_flag = Kokkos::create_mirror_view(flag);
    Kokkos::deep_copy(h_flag, flag);
    REQUIRE(h_flag(0) == 1);

    auto h_box1 = Kokkos::create_mirror_view(bws.geom.max_hkl);
    Kokkos::deep_copy(h_box1, bws.geom.max_hkl);
    const int after[3] = {h_box1(0,0), h_box1(0,1), h_box1(0,2)};

    // With kappa re-optimised, the k-vector count returns near the healthy one.
    INFO("after rebracket " << box_count(after) << " vs healthy "
                            << box_count(healthy));
    REQUIRE(box_count(after) < 4 * box_count(healthy));
}

TEST_CASE("a rebracketed box still covers its gmax sphere", "[batch][ewald]") {
    auto _r = make_nacl_dimer(); auto& mol = _r.first; auto& ws = _r.second;
    const RealType accuracy = 1e-6;
    setup_ewald(mol, ws);
    auto bws = allocate_batch_workspace(1, ws, mol);
    bws.tables.ewald_accuracy = accuracy;
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);

    auto grown = grow_cell_at_frozen_kappa(bws, mol.cell_params, 200.0,
                                           bws.tables.ff);
    SView1D<int, ExecSpace> live;
    SView1D<int, ExecSpace> flag("rebracketed", 1);
    batch_rebracket_ewald(bws.tables.ff, bws.geom, live, flag);

    auto h_kappa = Kokkos::create_mirror_view(bws.geom.kappa);
    Kokkos::deep_copy(h_kappa, bws.geom.kappa);
    const RealType gmax = 2 * h_kappa(0) * bws.tables.ff.ewald_split;

    RealType inv[3][3];
    REQUIRE(mat3_inv(grown.cell, inv));
    RealType b[3][3];
    for (int i = 0; i < 3; ++i)
        for (int d = 0; d < 3; ++d) b[i][d] = 2 * M_PI * inv[i][d];

    const int SWEEP = 48;
    int needed[3] = {0, 0, 0};
    for (int h = -SWEEP; h <= SWEEP; ++h)
        for (int k = -SWEEP; k <= SWEEP; ++k)
            for (int l = -SWEEP; l <= SWEEP; ++l) {
                RealType g2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const RealType gd = h*b[0][d] + k*b[1][d] + l*b[2][d];
                    g2 += gd * gd;
                }
                if (g2 >= gmax * gmax) continue;
                const int idx[3] = {std::abs(h), std::abs(k), std::abs(l)};
                for (int d = 0; d < 3; ++d)
                    if (idx[d] > needed[d]) needed[d] = idx[d];
            }

    auto h_box = Kokkos::create_mirror_view(bws.geom.max_hkl);
    Kokkos::deep_copy(h_box, bws.geom.max_hkl);
    for (int i = 0; i < 3; ++i) {
        INFO("axis " << i << " needs " << needed[i] << ", box is " << h_box(0, i));
        REQUIRE(needed[i] < SWEEP);
        REQUIRE(h_box(0, i) >= needed[i]);
    }
}
