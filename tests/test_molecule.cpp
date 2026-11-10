#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "klasp/molecule.hpp"
#include "klasp/cell.hpp"
#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinAbs;

// One independent molecule with an identity symop.
static std::pair<MoleculeData, Workspace> make_simple() {
    auto mol = allocate_molecule_data(1, 3, 1);
    auto ws  = allocate_workspace(10, 100, 100, 2);

    // Cubic cell 10x10x10
    mol.cell_params[0] = 10.0; mol.cell_params[1] = 10.0; mol.cell_params[2] = 10.0;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    // One independent molecule at fractional (0.5, 0.5, 0.5), no rotation
    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.5; h_t(0, 1) = 0.5; h_t(0, 2) = 0.5;
    Kokkos::deep_copy(mol.translations, h_t);

    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.0; h_r(0, 1) = 0.0; h_r(0, 2) = 0.0;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    // Body frame: 3-atom molecule along x
    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0, 0) = -1.0; h_b(0, 1) = 0.0; h_b(0, 2) = 0.0;
    h_b(1, 0) =  0.0; h_b(1, 1) = 0.0; h_b(1, 2) = 0.0;
    h_b(2, 0) =  1.0; h_b(2, 1) = 0.0; h_b(2, 2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 1.0; h_q(1) = -2.0; h_q(2) = 1.0;
    Kokkos::deep_copy(mol.body_charges, h_q);

    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    h_bt(0) = 0; h_bt(1) = 1; h_bt(2) = 0;
    Kokkos::deep_copy(mol.body_types, h_bt);

    // Identity symop
    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    h_sr(0, 0) = 1.0; h_sr(0, 1) = 0.0; h_sr(0, 2) = 0.0;
    h_sr(0, 3) = 0.0; h_sr(0, 4) = 1.0; h_sr(0, 5) = 0.0;
    h_sr(0, 6) = 0.0; h_sr(0, 7) = 0.0; h_sr(0, 8) = 1.0;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);

    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    h_st(0, 0) = 0.0; h_st(0, 1) = 0.0; h_st(0, 2) = 0.0;
    Kokkos::deep_copy(mol.sym_translations, h_st);

    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0) = 0;
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = 1;
    mol.n_symops = 1;
    mol.n_sites_per_mol = 3;

    return {mol, ws};
}

TEST_CASE("Identity symop + zero rotation: body positions offset by COM", "[molecule]") {
    auto _result = make_simple(); auto& mol = _result.first; auto& ws = _result.second;
    expand_asymmetric_unit(mol, ws);

    REQUIRE(ws.n_atoms == 3);

    auto h_pos = Kokkos::create_mirror_view(ws.positions);
    Kokkos::deep_copy(h_pos, ws.positions);

    // COM = (5.0, 5.0, 5.0), body offsets: (-1,0,0), (0,0,0), (1,0,0)
    REQUIRE_THAT(h_pos(0, 0), WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(h_pos(0, 1), WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(h_pos(0, 2), WithinAbs(5.0, 1e-10));

    REQUIRE_THAT(h_pos(1, 0), WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(h_pos(1, 1), WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(h_pos(1, 2), WithinAbs(5.0, 1e-10));

    REQUIRE_THAT(h_pos(2, 0), WithinAbs(6.0, 1e-10));
    REQUIRE_THAT(h_pos(2, 1), WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(h_pos(2, 2), WithinAbs(5.0, 1e-10));
}

TEST_CASE("180 rotation about z flips x,y", "[molecule]") {
    auto _result = make_simple(); auto& mol = _result.first; auto& ws = _result.second;

    // Set rotation to pi about z
    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.0; h_r(0, 1) = 0.0; h_r(0, 2) = M_PI;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    expand_asymmetric_unit(mol, ws);

    auto h_pos = Kokkos::create_mirror_view(ws.positions);
    Kokkos::deep_copy(h_pos, ws.positions);

    // Body (-1,0,0) rotated 180° about z → (1,0,0), then add COM (5,5,5) → (6,5,5)
    REQUIRE_THAT(h_pos(0, 0), WithinAbs(6.0, 1e-10));
    REQUIRE_THAT(h_pos(0, 1), WithinAbs(5.0, 1e-10));

    // Body (0,0,0) → (0,0,0) + COM → (5,5,5)
    REQUIRE_THAT(h_pos(1, 0), WithinAbs(5.0, 1e-10));

    // Body (1,0,0) rotated 180° about z → (-1,0,0) + COM → (4,5,5)
    REQUIRE_THAT(h_pos(2, 0), WithinAbs(4.0, 1e-10));
}

TEST_CASE("Expansion with 2 symops produces correct n_atoms", "[molecule]") {
    auto mol = allocate_molecule_data(1, 3, 4);
    auto ws  = allocate_workspace(20, 100, 100, 2);

    mol.cell_params[0] = 10.0; mol.cell_params[1] = 10.0; mol.cell_params[2] = 10.0;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    h_t(0, 0) = 0.1; h_t(0, 1) = 0.2; h_t(0, 2) = 0.3;
    Kokkos::deep_copy(mol.translations, h_t);

    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    h_r(0, 0) = 0.0; h_r(0, 1) = 0.0; h_r(0, 2) = 0.0;
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    auto h_b = Kokkos::create_mirror_view(mol.body_positions);
    h_b(0, 0) = -1.0; h_b(0, 1) = 0.0; h_b(0, 2) = 0.0;
    h_b(1, 0) =  0.0; h_b(1, 1) = 0.0; h_b(1, 2) = 0.0;
    h_b(2, 0) =  1.0; h_b(2, 1) = 0.0; h_b(2, 2) = 0.0;
    Kokkos::deep_copy(mol.body_positions, h_b);

    auto h_q = Kokkos::create_mirror_view(mol.body_charges);
    h_q(0) = 1.0; h_q(1) = -2.0; h_q(2) = 1.0;
    Kokkos::deep_copy(mol.body_charges, h_q);

    auto h_bt = Kokkos::create_mirror_view(mol.body_types);
    h_bt(0) = 0; h_bt(1) = 1; h_bt(2) = 0;
    Kokkos::deep_copy(mol.body_types, h_bt);

    // 2 symops: identity + inversion
    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    // Identity
    h_sr(0, 0) = 1; h_sr(0, 1) = 0; h_sr(0, 2) = 0;
    h_sr(0, 3) = 0; h_sr(0, 4) = 1; h_sr(0, 5) = 0;
    h_sr(0, 6) = 0; h_sr(0, 7) = 0; h_sr(0, 8) = 1;
    // Inversion
    h_sr(1, 0) = -1; h_sr(1, 1) = 0; h_sr(1, 2) = 0;
    h_sr(1, 3) = 0; h_sr(1, 4) = -1; h_sr(1, 5) = 0;
    h_sr(1, 6) = 0; h_sr(1, 7) = 0; h_sr(1, 8) = -1;
    Kokkos::deep_copy(mol.sym_rotations, h_sr);

    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    h_st(0, 0) = 0.0; h_st(0, 1) = 0.0; h_st(0, 2) = 0.0;
    h_st(1, 0) = 0.0; h_st(1, 1) = 0.0; h_st(1, 2) = 0.0;
    Kokkos::deep_copy(mol.sym_translations, h_st);

    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    h_si(0) = 0; h_si(1) = 0;
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = 1;
    mol.n_symops = 2;
    mol.n_sites_per_mol = 3;

    expand_asymmetric_unit(mol, ws);

    REQUIRE(ws.n_atoms == 6);

    // Verify charges copied correctly for both molecules
    auto h_charges = Kokkos::create_mirror_view(ws.charges);
    Kokkos::deep_copy(h_charges, ws.charges);
    REQUIRE_THAT(h_charges(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(h_charges(1), WithinAbs(-2.0, 1e-10));
    REQUIRE_THAT(h_charges(2), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(h_charges(3), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(h_charges(4), WithinAbs(-2.0, 1e-10));
    REQUIRE_THAT(h_charges(5), WithinAbs(1.0, 1e-10));
}

TEST_CASE("Cell matrix set correctly after expansion", "[molecule]") {
    auto _result = make_simple(); auto& mol = _result.first; auto& ws = _result.second;
    expand_asymmetric_unit(mol, ws);

    auto h_cell = Kokkos::create_mirror_view(ws.cell_matrix);
    Kokkos::deep_copy(h_cell, ws.cell_matrix);

    // Cubic 10x10x10: diagonal
    REQUIRE_THAT(h_cell(0, 0), WithinAbs(10.0, RealType(1e-4)));
    REQUIRE_THAT(h_cell(1, 1), WithinAbs(10.0, RealType(1e-4)));
    REQUIRE_THAT(h_cell(2, 2), WithinAbs(10.0, RealType(1e-4)));
    REQUIRE_THAT(h_cell(0, 1), WithinAbs(0.0, RealType(1e-4)));
}
