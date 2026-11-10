#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/rotation.hpp"
#include <Kokkos_Timer.hpp>
#include <fmt/base.h>
#include <cmath>

namespace klasp {

template <typename Space>
BatchLBFGSStateT<Space> allocate_batch_lbfgs_state(int n_batch, int n_dof, int memory) {
    BatchLBFGSStateT<Space> st;
    st.s_hist        = SView3D<RealType, Space>("blbfgs_s", n_batch, memory, n_dof);
    st.y_hist        = SView3D<RealType, Space>("blbfgs_y", n_batch, memory, n_dof);
    st.rho           = SView2D<RealType, Space>("blbfgs_rho", n_batch, memory);
    st.alpha_scratch = SView2D<RealType, Space>("blbfgs_alpha", n_batch, memory);
    st.direction     = SView2D<RealType, Space>("blbfgs_d", n_batch, n_dof);
    st.x             = SView2D<RealType, Space>("blbfgs_x", n_batch, n_dof);
    st.g             = SView2D<RealType, Space>("blbfgs_g", n_batch, n_dof);
    st.x_new         = SView2D<RealType, Space>("blbfgs_xn", n_batch, n_dof);
    st.g_new         = SView2D<RealType, Space>("blbfgs_gn", n_batch, n_dof);
    st.num_stored    = SView1D<int, Space>("blbfgs_ns", n_batch);
    st.cursor        = SView1D<int, Space>("blbfgs_cur", n_batch);
    st.n_dof  = n_dof;
    st.memory = memory;
    return st;
}

template <typename Space>
BatchWorkspaceT<Space> allocate_batch_workspace(int n_batch,
                                         const Workspace& ws,
                                         const MoleculeData& mol) {
    using SpaceMem = typename Space::memory_space;
    BatchWorkspaceT<Space> bws;
    bws.geom.n_batch = n_batch;

    bws.tables.buck_A        = Kokkos::create_mirror_view_and_copy(SpaceMem(), ws.buck_A);
    bws.tables.buck_B        = Kokkos::create_mirror_view_and_copy(SpaceMem(), ws.buck_B);
    bws.tables.buck_C        = Kokkos::create_mirror_view_and_copy(SpaceMem(), ws.buck_C);
    bws.tables.buck_Rmin     = Kokkos::create_mirror_view_and_copy(SpaceMem(), ws.buck_Rmin);
    bws.tables.buck_cutoff   = ws.buck_cutoff;
    // RadialSpline holds a bare pointer into the coefficient array; repoint it after the copy.
    if (ws.pair_tables) {
        bws.tables.pair_table_coeffs =
            Kokkos::create_mirror_view_and_copy(SpaceMem(), ws.pair_tables->coeffs);
        bws.tables.pair_tables =
            Kokkos::create_mirror_view_and_copy(SpaceMem(), ws.pair_tables->splines);
        rebase_pair_tables<Space>(bws.tables.pair_tables,
                                  ws.pair_tables->coeffs.data(),
                                  bws.tables.pair_table_coeffs.data());
    }
    bws.tables.ewald_accuracy = ws.ewald_accuracy;

    // Default model: short-range pair potential plus Ewald. Callers with another model overwrite it.
    bws.tables.ff = make_force_field(bws.tables, EnergyModelParams{});
    bws.tables.ghost_type    = ws.ghost_type;
    bws.tables.n_types       = ws.n_types;
    bws.tables.max_multipole_rank = mol.max_multipole_rank;

    bws.tables.body_positions   = Kokkos::create_mirror_view_and_copy(SpaceMem(), mol.body_positions);
    bws.tables.body_charges     = Kokkos::create_mirror_view_and_copy(SpaceMem(), mol.body_charges);
    bws.tables.body_types       = Kokkos::create_mirror_view_and_copy(SpaceMem(), mol.body_types);
    bws.tables.body_multipoles  = Kokkos::create_mirror_view_and_copy(SpaceMem(), mol.body_multipoles);
    copy_symmetry_ops(bws.tables.symmetry(), mol.symmetry());
    bws.tables.n_sites_per_mol  = mol.n_sites_per_mol;
    bws.geom.n_atoms          = mol.n_symops * mol.n_sites_per_mol;

    // Bounding radius per independent molecule (see ModelTablesT::mol_radius).
    {
        auto h_body = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mol.body_positions);
        auto h_cnt  = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mol.mol_site_count);
        const int n_ind = bws.tables.n_indep;
        const int n_st  = mol.n_sites_per_mol;
        Kokkos::View<RealType*, HostMemSpace> h_rad("mol_radius", n_ind);
        for (int k = 0; k < n_ind; ++k) {
            RealType r2max = 0;
            // Copy k is the identity copy of independent molecule k.
            for (int s = 0; s < h_cnt(k); ++s) {
                const RealType x = h_body(k*n_st + s, 0), y = h_body(k*n_st + s, 1),
                               z = h_body(k*n_st + s, 2);
                r2max = std::max(r2max, x*x + y*y + z*z);
            }
            h_rad(k) = std::sqrt(r2max);
        }
        bws.tables.mol_radius = Kokkos::create_mirror_view_and_copy(SpaceMem(), h_rad);
    }

    int na = bws.geom.n_atoms;
    int ni = bws.tables.n_indep;

    bws.geom.cells() = allocate_cell_state<Space>(n_batch, "b");

    bws.dofs.translations = SView3D<RealType, Space>("b_trans", n_batch, ni, 3);
    bws.dofs.rot_vectors  = SView3D<RealType, Space>("b_rots", n_batch, ni, 3);

    bws.geom.positions  = SView3D<RealType, Space>("b_pos", n_batch, na, 3);
    bws.geom.charges    = SView2D<RealType, Space>("b_q", n_batch, na);
    bws.geom.types      = SView2D<int, Space>("b_types", n_batch, na);
    bws.geom.multipoles = SView3D<RealType, Space>("b_mp", n_batch, na, 35);
    bws.geom.dipoles    = SView3D<RealType, Space>("b_dip", n_batch, na, 3);
    bws.geom.mol_id     = SView2D<int, Space>("b_molid", n_batch, na);

    // Asymmetric-unit sized: n_indep * n_sites entries.
    {
        const int n_asym = mol.n_indep * mol.n_sites_per_mol;
        bws.dofs.d_gt_c = SView3D<RealType, Space>("b_dgtc", n_batch, mol.n_indep, 3);
        bws.dofs.d_gr_c = SView3D<RealType, Space>("b_dgrc", n_batch, mol.n_indep, 3);
        bws.dofs.jac_t = SView3D<RealType, Space>("b_jt", n_batch, mol.n_indep, 9);
        bws.dofs.jac_r = SView3D<RealType, Space>("b_jr", n_batch, n_asym, 9);
        bws.dofs.jac_m = SView3D<RealType, Space>("b_jm", n_batch, n_asym, 3 * 35);
    }
    bws.dofs.d_gt         = SView3D<RealType, Space>("b_dgt", n_batch, ni, 3);
    bws.dofs.d_gr         = SView3D<RealType, Space>("b_dgr", n_batch, ni, 3);

    bws.geom.com_positions = SView3D<RealType, Space>("b_com", n_batch, bws.tables.n_symops, 3);
    bws.geom.com_positions_ref = SView3D<RealType, Space>("b_com_ref", n_batch, bws.tables.n_symops, 3);

    bws.geom.energies = SView1D<RealType, Space>("b_E", n_batch);

    return bws;
}

// Per-structure cell setup, on device.

template <typename Space>
void batch_setup_cells(const ModelTablesT<Space>& tables,
                       BatchGeometryT<Space>& geom) {
    const int n_batch = geom.n_batch;
    const int n_atoms = geom.n_atoms;
    const ForceField ff = tables.ff;

    auto cell_params = geom.cell_params;
    auto cell        = geom.cell;
    auto cell_inv    = geom.cell_inv;
    auto kappa       = geom.kappa;
    auto volume      = geom.volume;
    auto max_hkl     = geom.max_hkl;
    auto max_shift   = geom.max_shift;

    constexpr RealType PI = RealType(M_PI);
    const RealType deg2rad = PI / RealType(180);

    Kokkos::parallel_for("batch_setup_cells",
        Kokkos::RangePolicy<Space>(0, n_batch),
        KOKKOS_LAMBDA(const int b) {
            RealType a        = cell_params(b, 0);
            RealType bv       = cell_params(b, 1);
            RealType cv       = cell_params(b, 2);
            RealType alpha_d  = cell_params(b, 3);
            RealType beta_d   = cell_params(b, 4);
            RealType gamma_d  = cell_params(b, 5);

            // Cell matrix from params (upper triangular).
            RealType cos_a = Kokkos::cos(alpha_d * deg2rad);
            RealType cos_b = Kokkos::cos(beta_d  * deg2rad);
            RealType cos_g = Kokkos::cos(gamma_d * deg2rad);
            RealType sin_g = Kokkos::sin(gamma_d * deg2rad);

            RealType cm[3][3];
            cm[0][0] = a;
            cm[0][1] = bv * cos_g;
            cm[0][2] = cv * cos_b;
            cm[1][0] = 0;
            cm[1][1] = bv * sin_g;
            cm[1][2] = cv * (cos_a - cos_b * cos_g) / sin_g;
            cm[2][0] = 0;
            cm[2][1] = 0;
            cm[2][2] = Kokkos::sqrt(cv*cv - cm[0][2]*cm[0][2] - cm[1][2]*cm[1][2]);

            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    cell(b, r, c) = cm[r][c];

            RealType det = mat3_det(cm);
            RealType vol = det > RealType(0) ? det : -det;

            if (vol < RealType(1e-6) || Kokkos::isnan(vol)) {
                volume(b) = RealType(1);
                kappa(b)  = RealType(1);
                for (int d = 0; d < 3; ++d) { max_hkl(b,d) = 1; max_shift(b,d) = 2; }
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        cell_inv(b, r, c) = (r == c) ? RealType(1) : RealType(0);
                return;
            }
            volume(b) = vol;

            RealType ci[3][3];
            mat3_inv(cm, ci);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    cell_inv(b, r, c) = ci[r][c];

            RealType eta = Kokkos::cbrt(
                (static_cast<RealType>(n_atoms) * PI * PI * PI) / (vol * vol));
            RealType kap = Kokkos::sqrt(eta);
            kappa(b) = kap;

            // Reciprocal index box covering the gmax sphere, sized from lattice vector
            // lengths (see reciprocal_index_bounds).
            if (ff.has_reciprocal()) {
                RealType gmax = RealType(2) * kap * ff.ewald_split;
                int hkl[3];
                reciprocal_index_bounds(cm, gmax, hkl);
                for (int d = 0; d < 3; ++d) max_hkl(b, d) = hkl[d];
            } else {
                for (int d = 0; d < 3; ++d) max_hkl(b, d) = 0;
            }

            const RealType cutoff = ff.real_space_cutoff(kap);

            // Interplanar spacings for max_shift; lattice vectors are columns.
            RealType ax_ = cm[0][0], ay_ = cm[1][0], az_ = cm[2][0];
            RealType bx_ = cm[0][1], by_ = cm[1][1], bz_ = cm[2][1];
            RealType cx_ = cm[0][2], cy_ = cm[1][2], cz_ = cm[2][2];

            RealType bcx = by_*cz_ - bz_*cy_;
            RealType bcy = bz_*cx_ - bx_*cz_;
            RealType bcz = bx_*cy_ - by_*cx_;
            RealType cax = cy_*az_ - cz_*ay_;
            RealType cay = cz_*ax_ - cx_*az_;
            RealType caz = cx_*ay_ - cy_*ax_;
            RealType abx = ay_*bz_ - az_*by_;
            RealType aby = az_*bx_ - ax_*bz_;
            RealType abz = ax_*by_ - ay_*bx_;

            RealType nbc = Kokkos::sqrt(bcx*bcx + bcy*bcy + bcz*bcz);
            RealType nca = Kokkos::sqrt(cax*cax + cay*cay + caz*caz);
            RealType nab = Kokkos::sqrt(abx*abx + aby*aby + abz*abz);
            if (nbc < RealType(1e-10)) nbc = RealType(1e-10);
            if (nca < RealType(1e-10)) nca = RealType(1e-10);
            if (nab < RealType(1e-10)) nab = RealType(1e-10);

            RealType h_a = vol / nbc;
            RealType h_b = vol / nca;
            RealType h_c = vol / nab;
            if (h_a < RealType(0.1)) h_a = RealType(0.1);
            if (h_b < RealType(0.1)) h_b = RealType(0.1);
            if (h_c < RealType(0.1)) h_c = RealType(0.1);

            // No cap on the image count: oblique cells need it. batch_update_cell_from_strain
            // and cell.cpp must size it the same way.
            max_shift(b, 0) = static_cast<int>(Kokkos::ceil(cutoff / h_a)) + 1;
            max_shift(b, 1) = static_cast<int>(Kokkos::ceil(cutoff / h_b)) + 1;
            max_shift(b, 2) = static_cast<int>(Kokkos::ceil(cutoff / h_c)) + 1;
        });
}

template <int MaxRank>
static RealType batch_evaluate_energy_impl(const MoleculeData& mol, const Workspace& ws,
                                            RealType taper_on, RealType taper_off,
                                            TangToenniesDamping buck_damping, uint32_t components,
                                            RealType mp_cutoff, bool mp_single) {
    auto bws = allocate_batch_workspace<ExecSpace>(1, ws, mol);

    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = mol.cell_params[i];
    Kokkos::deep_copy(ExecSpace(), bws.geom.cell_params, h_cp);
    batch_setup_cells<ExecSpace>(bws.tables, bws.geom);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(ExecSpace(), h_mt, mol.translations);
    Kokkos::deep_copy(ExecSpace(), h_mr, mol.rot_vectors);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_bt(0,k,d) = h_mt(k,d);
            h_br(0,k,d) = h_mr(k,d);
        }
    Kokkos::deep_copy(ExecSpace(), bws.dofs.translations, h_bt);
    Kokkos::deep_copy(ExecSpace(), bws.dofs.rot_vectors, h_br);

    batch_expand_asymmetric_unit<MaxRank, ExecSpace>(bws.tables, bws.geom, bws.dofs);
    ExecSpace().fence();

    BatchOptimizeSettings bsettings;
    bsettings.model.taper_on = taper_on;
    bsettings.model.taper_off = taper_off;
    bsettings.energy_only = true;
    bsettings.model.buck_damping = buck_damping;
    bsettings.model.ewald_components = components;
    bsettings.model.mp_cutoff = mp_cutoff;
    bsettings.model.mp_single = mp_single;

    batch_energy_gradient_asymunit<MaxRank, ExecSpace>(bws.tables, bws.geom, bws.dofs, bsettings, 0);
    ExecSpace().fence();

    auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
    Kokkos::deep_copy(ExecSpace(), h_E, bws.geom.energies);
    return h_E(0);
}

RealType batch_evaluate_energy(const MoleculeData& mol, const Workspace& ws,
                                int max_rank,
                                RealType taper_on, RealType taper_off,
                                TangToenniesDamping buck_damping, uint32_t components,
                                RealType mp_cutoff, bool mp_single) {
    switch (max_rank) {
        case 1: return batch_evaluate_energy_impl<1>(mol, ws, taper_on, taper_off, buck_damping, components, mp_cutoff, mp_single);
        case 2: return batch_evaluate_energy_impl<2>(mol, ws, taper_on, taper_off, buck_damping, components, mp_cutoff, mp_single);
        case 3: return batch_evaluate_energy_impl<3>(mol, ws, taper_on, taper_off, buck_damping, components, mp_cutoff, mp_single);
        case 4: return batch_evaluate_energy_impl<4>(mol, ws, taper_on, taper_off, buck_damping, components, mp_cutoff, mp_single);
        default: return batch_evaluate_energy_impl<0>(mol, ws, taper_on, taper_off, buck_damping, components, mp_cutoff, mp_single);
    }
}


// Explicit template instantiations.
template BatchLBFGSStateT<ExecSpace> allocate_batch_lbfgs_state<ExecSpace>(int, int, int);
template BatchWorkspaceT<ExecSpace> allocate_batch_workspace<ExecSpace>(int, const Workspace&, const MoleculeData&);
template void batch_setup_cells<ExecSpace>(const ModelTables&, BatchGeometry&);

#ifdef KLASP_HAS_DEVICE_SPACE
template BatchLBFGSStateT<HostExecSpace> allocate_batch_lbfgs_state<HostExecSpace>(int, int, int);
template BatchWorkspaceT<HostExecSpace> allocate_batch_workspace<HostExecSpace>(int, const Workspace&, const MoleculeData&);
template void batch_setup_cells<HostExecSpace>(const HostModelTables&, HostBatchGeometry&);
#endif

}  // namespace klasp
