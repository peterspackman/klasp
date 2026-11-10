#include "klasp/log.hpp"
#include "klasp/molecule.hpp"
#include "klasp/rotation.hpp"
#include "klasp/cell.hpp"
#include "klasp/multipole.hpp"

namespace klasp {

MoleculeData allocate_molecule_data(int max_indep, int max_sites, int max_symops,
                                    int max_multipole_rank) {
    MoleculeData mol;
    mol.translations     = View2D<RealType>("translations",     max_indep, 3);
    mol.rot_vectors      = View2D<RealType>("rot_vectors",      max_indep, 3);

    mol.body_positions   = View2D<RealType>("body_positions",   max_indep * max_sites, 3);
    mol.body_charges     = View1D<RealType>("body_charges",     max_indep * max_sites);
    mol.body_types       = View1D<int>("body_types",            max_indep * max_sites);

    allocate_symmetry_ops(mol.symmetry(), max_symops);
    // Default to no padding: every molecule owns the full stride.
    Kokkos::deep_copy(mol.mol_site_count, max_sites);

    mol.max_multipole_rank = max_multipole_rank;
    if (max_multipole_rank > 0) {
        mol.body_multipoles = View2D<RealType>("body_multipoles", max_indep * max_sites, 35);
    }

    for (int i = 0; i < 6; ++i) mol.cell_params[i] = 0.0;
    mol.n_indep = 0;
    mol.n_symops = 0;
    mol.n_sites_per_mol = 0;

    return mol;
}

View2D<RealType> build_cartesian_symops(const MoleculeData& mol) {
    RealType cell[3][3];
    cell_params_to_matrix(mol.cell_params, cell);
    return build_cartesian_symops(mol, cell);
}

View2D<RealType> build_cartesian_symops(const MoleculeData& mol,
                                        const RealType cell[3][3]) {
    const int n_symops = mol.n_symops;
    // mat3_inv leaves its output untouched on a singular matrix, so start from
    // zeros: a degenerate cell then gives zero operators and zero forces.
    RealType cell_inv[3][3] = {};
    if (!mat3_inv(cell, cell_inv))
        logging::warn("warning: degenerate cell in build_cartesian_symops; "
                      "symmetry operators are zero\n");

    View2D<RealType> sym_cart("sym_cart", n_symops, 9);
    auto h_sf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mol.sym_rotations);
    auto h_sc = Kokkos::create_mirror_view(sym_cart);
    for (int s = 0; s < n_symops; ++s) {
        RealType Sf[3][3], Sc[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) Sf[i][j] = h_sf(s, i * 3 + j);
        frac_to_cart_symop(cell, cell_inv, Sf, Sc);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) h_sc(s, i * 3 + j) = Sc[i][j];
    }
    Kokkos::deep_copy(sym_cart, h_sc);
    return sym_cart;
}

template <int MaxRank>
void expand_asymmetric_unit(const MoleculeData& mol, Workspace& ws) {
    const int n_symops = mol.n_symops;
    const int n_sites  = mol.n_sites_per_mol;
    const int n_atoms = n_symops * n_sites;

    RealType cell[3][3];
    cell_params_to_matrix(mol.cell_params, cell);

    auto h_cell = Kokkos::create_mirror_view(ws.cell_matrix);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            h_cell(r, c) = cell[r][c];
    Kokkos::deep_copy(ws.cell_matrix, h_cell);

    auto translations   = mol.translations;
    auto rot_vectors    = mol.rot_vectors;
    auto body_positions = mol.body_positions;
    auto body_charges   = mol.body_charges;
    auto body_types     = mol.body_types;
    auto sym_rotations  = mol.sym_rotations;
    auto sym_translations = mol.sym_translations;

    // Cartesian operators for body coordinates and multipoles; fractional
    // operators for the fractional COM.
    View2D<RealType> sym_cart = build_cartesian_symops(mol);
    auto sym_indep_index = mol.sym_indep_index;
    auto positions      = ws.positions;
    auto charges        = ws.charges;
    auto types          = ws.types;
    auto cell_mat       = ws.cell_matrix;
    auto mol_id         = ws.mol_id;

    if constexpr (MaxRank > 0) {
        auto multipoles = ws.multipoles;
        auto body_mp    = mol.body_multipoles;

        Kokkos::parallel_for("expand_mp", n_symops * n_sites,
            KOKKOS_LAMBDA(const int flat) {
                const int s = flat / n_sites;
                const int j = flat % n_sites;
                const int atom_idx = flat;

                int indep_idx = sym_indep_index(s);

                RealType aa[3] = {rot_vectors(indep_idx, 0),
                                  rot_vectors(indep_idx, 1),
                                  rot_vectors(indep_idx, 2)};
                RealType R_mol[3][3];
                angle_axis_to_matrix(aa, R_mol);

                RealType R_sym[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int jj = 0; jj < 3; ++jj)
                        R_sym[i][jj] = sym_rotations(s, i * 3 + jj);

                RealType S_cart[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int jj = 0; jj < 3; ++jj)
                        S_cart[i][jj] = sym_cart(s, i * 3 + jj);

                RealType R_total[3][3];
                mat3_mul(S_cart, R_mol, R_total);

                // Apply symop to fractional COM: frac' = R_sym @ frac + t_sym
                RealType frac_indep[3] = {translations(indep_idx, 0),
                                          translations(indep_idx, 1),
                                          translations(indep_idx, 2)};
                RealType frac_com[3];
                mat3_vec(R_sym, frac_indep, frac_com);
                for (int d = 0; d < 3; ++d) {
                    frac_com[d] += sym_translations(s, d);
                }

                RealType cart_com[3];
                for (int i = 0; i < 3; ++i) {
                    cart_com[i] = 0.0;
                    for (int k = 0; k < 3; ++k)
                        cart_com[i] += frac_com[k] * cell_mat(i, k);
                }

                int body_idx = indep_idx * n_sites + j;
                RealType body[3] = {body_positions(body_idx, 0), body_positions(body_idx, 1), body_positions(body_idx, 2)};
                RealType rotated[3];
                mat3_vec(R_total, body, rotated);

                for (int d = 0; d < 3; ++d)
                    positions(atom_idx, d) = rotated[d] + cart_com[d];
                charges(atom_idx) = body_charges(body_idx);
                types(atom_idx) = body_types(body_idx);
                mol_id(atom_idx) = s;

                RealType body_cart[35];
                for (int c = 0; c < 35; ++c)
                    body_cart[c] = body_mp(body_idx, c);

                RealType lab_cart[35];
                rotate_cartesian_multipole(body_cart, R_total, MaxRank, lab_cart);

                for (int c = 0; c < 35; ++c)
                    multipoles(atom_idx, c) = lab_cart[c];

                // Cartesian dipoles from lab-frame multipoles: components 1, 2, 3
                // are x, y, z.
                auto dip = ws.dipoles;
                dip(atom_idx, 0) = lab_cart[hermite_index(1, 0, 0)];
                dip(atom_idx, 1) = lab_cart[hermite_index(0, 1, 0)];
                dip(atom_idx, 2) = lab_cart[hermite_index(0, 0, 1)];
            });
    } else {
        Kokkos::parallel_for("expand", n_symops * n_sites,
            KOKKOS_LAMBDA(const int flat) {
                const int s = flat / n_sites;
                const int j = flat % n_sites;
                const int atom_idx = flat;

                int indep_idx = sym_indep_index(s);

                RealType aa[3] = {rot_vectors(indep_idx, 0),
                                  rot_vectors(indep_idx, 1),
                                  rot_vectors(indep_idx, 2)};
                RealType R_mol[3][3];
                angle_axis_to_matrix(aa, R_mol);

                RealType R_sym[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int jj = 0; jj < 3; ++jj)
                        R_sym[i][jj] = sym_rotations(s, i * 3 + jj);

                RealType S_cart[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int jj = 0; jj < 3; ++jj)
                        S_cart[i][jj] = sym_cart(s, i * 3 + jj);

                RealType R_total[3][3];
                mat3_mul(S_cart, R_mol, R_total);

                // Apply symop to fractional COM: frac' = R_sym @ frac + t_sym
                RealType frac_indep[3] = {translations(indep_idx, 0),
                                          translations(indep_idx, 1),
                                          translations(indep_idx, 2)};
                RealType frac_com[3];
                mat3_vec(R_sym, frac_indep, frac_com);
                for (int d = 0; d < 3; ++d) {
                    frac_com[d] += sym_translations(s, d);
                }

                RealType cart_com[3];
                for (int i = 0; i < 3; ++i) {
                    cart_com[i] = 0.0;
                    for (int k = 0; k < 3; ++k)
                        cart_com[i] += frac_com[k] * cell_mat(i, k);
                }

                int body_idx = indep_idx * n_sites + j;
                RealType body[3] = {body_positions(body_idx, 0), body_positions(body_idx, 1), body_positions(body_idx, 2)};
                RealType rotated[3];
                mat3_vec(R_total, body, rotated);

                for (int d = 0; d < 3; ++d)
                    positions(atom_idx, d) = rotated[d] + cart_com[d];
                charges(atom_idx) = body_charges(body_idx);
                types(atom_idx) = body_types(body_idx);
                mol_id(atom_idx) = s;
            });
    }

    ws.n_atoms = n_atoms;
}

template void expand_asymmetric_unit<0>(const MoleculeData& mol, Workspace& ws);
template void expand_asymmetric_unit<1>(const MoleculeData& mol, Workspace& ws);
template void expand_asymmetric_unit<2>(const MoleculeData& mol, Workspace& ws);
template void expand_asymmetric_unit<3>(const MoleculeData& mol, Workspace& ws);
template void expand_asymmetric_unit<4>(const MoleculeData& mol, Workspace& ws);

}  // namespace klasp
