#include "klasp/chain_rule.hpp"
#include "klasp/rotation.hpp"
#include "klasp/cell.hpp"
#include "klasp/multipole.hpp"
#include <cstring>

namespace klasp {

namespace {
// d_cell is a device View; copy it to a host 3x3 for the operator build.
struct CellBuf { RealType m[3][3]; };
inline CellBuf read_cell(const View2D<RealType>& d_cell) {
    CellBuf c{};
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_cell);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) c.m[i][j] = h(i, j);
    return c;
}
}  // namespace

void compute_dof_gradients(const MoleculeData& mol,
                           const View2D<RealType>& forces,
                           const View2D<RealType>& d_cell,
                           View2D<RealType>& d_gt,
                           View2D<RealType>& d_gr,
                           RealType* grad_trans,
                           RealType* grad_rot) {
    const int n_indep = mol.n_indep;
    const int n_symops = mol.n_symops;
    const int n_sites = mol.n_sites_per_mol;

    std::memset(grad_trans, 0, n_indep * 3 * sizeof(RealType));
    std::memset(grad_rot, 0, n_indep * 3 * sizeof(RealType));

    Kokkos::deep_copy(d_gt, 0.0);
    Kokkos::deep_copy(d_gr, 0.0);

    auto rot_vectors    = mol.rot_vectors;
    auto body_positions = mol.body_positions;
    auto sym_indep_index = mol.sym_indep_index;
    // Cartesian operators, built from the cell passed in (mol.cell_params may be unset).
    auto sym_cart = build_cartesian_symops(mol, read_cell(d_cell).m);

    // One work item per symop, accumulating into its independent molecule's gradient.
    Kokkos::parallel_for("chain_rule", n_symops,
        KOKKOS_LAMBDA(const int s) {
            int k = sym_indep_index(s);

            RealType neg_f_sum[3] = {0.0, 0.0, 0.0};
            for (int j = 0; j < n_sites; ++j) {
                int a = s * n_sites + j;
                for (int d = 0; d < 3; ++d)
                    neg_f_sum[d] -= forces(a, d);
            }

            // dE/d(frac_d) = sum_i dE/d(cart_i) * cell[i][d]
            for (int d = 0; d < 3; ++d) {
                RealType g = 0.0;
                for (int i = 0; i < 3; ++i)
                    g += neg_f_sum[i] * d_cell(i, d);
                Kokkos::atomic_add(&d_gt(k, d), g);
            }

            RealType aa[3] = {rot_vectors(k, 0), rot_vectors(k, 1), rot_vectors(k, 2)};
            RealType dR0[3][3], dR1[3][3], dR2[3][3];
            angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);

            RealType R_sym[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R_sym[i][j] = sym_cart(s, i * 3 + j);

            RealType* dR_all[3] = {&dR0[0][0], &dR1[0][0], &dR2[0][0]};

            for (int m = 0; m < 3; ++m) {
                RealType (*dR_m)[3] = reinterpret_cast<RealType(*)[3]>(dR_all[m]);
                RealType SdR[3][3];
                mat3_mul(R_sym, dR_m, SdR);

                RealType g = 0.0;
                for (int j = 0; j < n_sites; ++j) {
                    int a = s * n_sites + j;
                    int body_idx = k * n_sites + j;
                    RealType body[3] = {body_positions(body_idx, 0), body_positions(body_idx, 1), body_positions(body_idx, 2)};
                    RealType v[3];
                    mat3_vec(SdR, body, v);
                    for (int d = 0; d < 3; ++d)
                        g += (-forces(a, d)) * v[d];
                }
                Kokkos::atomic_add(&d_gr(k, m), g);
            }
        });

    auto h_gt = Kokkos::create_mirror_view(d_gt);
    auto h_gr = Kokkos::create_mirror_view(d_gr);
    Kokkos::deep_copy(h_gt, d_gt);
    Kokkos::deep_copy(h_gr, d_gr);

    for (int k = 0; k < n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            grad_trans[k * 3 + d] = h_gt(k, d);
            grad_rot[k * 3 + d] = h_gr(k, d);
        }
}

void compute_dof_gradients(const MoleculeData& mol,
                           const View2D<RealType>& forces,
                           RealType* grad_trans,
                           RealType* grad_rot) {
    const int n_indep = mol.n_indep;

    View2D<RealType> d_gt("grad_trans", n_indep, 3);
    View2D<RealType> d_gr("grad_rot", n_indep, 3);

    RealType cell[3][3];
    cell_params_to_matrix(mol.cell_params, cell);
    View2D<RealType> d_cell("cell", 3, 3);
    auto h_cell = Kokkos::create_mirror_view(d_cell);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            h_cell(i, j) = cell[i][j];
    Kokkos::deep_copy(d_cell, h_cell);

    compute_dof_gradients(mol, forces, d_cell, d_gt, d_gr, grad_trans, grad_rot);
}

void compute_dof_gradients(const MoleculeData& mol,
                           const View2D<RealType>& forces,
                           const View2D<RealType>& dE_dmp,
                           const View2D<RealType>& d_cell,
                           View2D<RealType>& d_gt,
                           View2D<RealType>& d_gr,
                           RealType* grad_trans,
                           RealType* grad_rot) {
    const int n_indep = mol.n_indep;
    const int n_symops = mol.n_symops;
    const int n_sites = mol.n_sites_per_mol;
    const int mp_rank = mol.max_multipole_rank;

    std::memset(grad_trans, 0, n_indep * 3 * sizeof(RealType));
    std::memset(grad_rot, 0, n_indep * 3 * sizeof(RealType));

    Kokkos::deep_copy(d_gt, 0.0);
    Kokkos::deep_copy(d_gr, 0.0);

    auto rot_vectors     = mol.rot_vectors;
    auto body_positions  = mol.body_positions;
    auto body_mp         = mol.body_multipoles;
    // Cartesian operators, built from the cell passed in (mol.cell_params may be unset).
    auto sym_cart = build_cartesian_symops(mol, read_cell(d_cell).m);
    auto sym_indep_index = mol.sym_indep_index;

    Kokkos::parallel_for("chain_rule_mp", n_symops,
        KOKKOS_LAMBDA(const int s) {
            int k = sym_indep_index(s);

            RealType neg_f_sum[3] = {0.0, 0.0, 0.0};
            for (int j = 0; j < n_sites; ++j) {
                int a = s * n_sites + j;
                for (int d = 0; d < 3; ++d)
                    neg_f_sum[d] -= forces(a, d);
            }

            for (int d = 0; d < 3; ++d) {
                RealType g = 0.0;
                for (int i = 0; i < 3; ++i)
                    g += neg_f_sum[i] * d_cell(i, d);
                Kokkos::atomic_add(&d_gt(k, d), g);
            }

            RealType aa[3] = {rot_vectors(k, 0), rot_vectors(k, 1), rot_vectors(k, 2)};
            RealType R_mol[3][3];
            angle_axis_to_matrix(aa, R_mol);

            RealType dR0[3][3], dR1[3][3], dR2[3][3];
            angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);

            RealType R_sym[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R_sym[i][j] = sym_cart(s, i * 3 + j);

            // R_total = R_sym @ R_mol
            RealType R_total[3][3];
            mat3_mul(R_sym, R_mol, R_total);

            RealType* dR_all[3] = {&dR0[0][0], &dR1[0][0], &dR2[0][0]};

            for (int m = 0; m < 3; ++m) {
                RealType (*dR_m)[3] = reinterpret_cast<RealType(*)[3]>(dR_all[m]);
                RealType SdR[3][3];
                mat3_mul(R_sym, dR_m, SdR);

                RealType g = 0.0;

                for (int j = 0; j < n_sites; ++j) {
                    int a = s * n_sites + j;
                    int body_idx = k * n_sites + j;
                    RealType body[3] = {body_positions(body_idx, 0), body_positions(body_idx, 1), body_positions(body_idx, 2)};
                    RealType v[3];
                    mat3_vec(SdR, body, v);
                    for (int d = 0; d < 3; ++d)
                        g += (-forces(a, d)) * v[d];
                }

                RealType g_pos = g;  // position contribution so far

                // Multipole torque contribution.
                RealType g_torque = 0.0;
                if (mp_rank > 0) {
                    for (int j = 0; j < n_sites; ++j) {
                        int a = s * n_sites + j;
                        int mp_body_idx = k * n_sites + j;

                        RealType body_cart[35];
                        for (int c = 0; c < 35; ++c)
                            body_cart[c] = body_mp(mp_body_idx, c);

                        // d(lab_mp)/d(aa_m), with SdR = R_sym @ dR_m = dR_total/d(aa_m).
                        RealType dlab_m[35];
                        rotate_cartesian_multipole_deriv(body_cart, R_total, SdR,
                                                         mp_rank, dlab_m);

                        for (int c = 0; c < 35; ++c)
                            g_torque += dE_dmp(a, c) * dlab_m[c];
                    }
                }

                Kokkos::atomic_add(&d_gr(k, m), g_pos + g_torque);
            }
        });

    auto h_gt = Kokkos::create_mirror_view(d_gt);
    auto h_gr = Kokkos::create_mirror_view(d_gr);
    Kokkos::deep_copy(h_gt, d_gt);
    Kokkos::deep_copy(h_gr, d_gr);

    for (int k = 0; k < n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            grad_trans[k * 3 + d] = h_gt(k, d);
            grad_rot[k * 3 + d] = h_gr(k, d);
        }
}

}  // namespace klasp
