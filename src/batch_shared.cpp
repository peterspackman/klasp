#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/cell.hpp"
#include "klasp/ewald.hpp"
#include "klasp/multipole.hpp"
#include "klasp/rotation.hpp"
#include "klasp/taper.hpp"
#include "klasp/pair_skip.hpp"
#include "klasp/padding.hpp"
#include "klasp/constants.hpp"
#include "klasp/lbfgs.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/energy_terms.hpp"   // DofGrad6
#include <Kokkos_Timer.hpp>
#include <fmt/base.h>
#include <cmath>

namespace klasp {

// Update cell matrices from Voigt strain.
template <typename Space>
void batch_update_cell_from_strain(const ForceField& ff,
                                   BatchGeometryT<Space>& geom,
                                   RigidDofStateT<Space>& dofs) {
    if (dofs.n_cell_dof == 0) return;

    const int n_batch = geom.n_batch;
    auto b_cell    = geom.cell;
    auto b_ci      = geom.cell_inv;
    auto b_vol     = geom.volume;
    auto cell_ref  = dofs.cell_ref;
    auto strain_v  = dofs.strain;
    auto b_mshift  = geom.max_shift;
    auto b_mhkl    = geom.max_hkl;

    auto b_kappa   = geom.kappa;
    // Captured by value: a device lambda must not capture a reference into the caller.
    const ForceField model = ff;
    const bool has_reciprocal = model.has_reciprocal();

    Kokkos::parallel_for("update_cell_from_strain", n_batch,
        KOKKOS_LAMBDA(const int b) {
            // Last-resort clamp on the linearised strain so a runaway step cannot invert or
            // collapse the cell. The trust region clips its step to the same bounds first.
            for (int s = 0; s < 3; ++s)
                strain_v(b, s) = Kokkos::fmax(-STRAIN_MAX_DIAG,
                                              Kokkos::fmin(STRAIN_MAX_DIAG, strain_v(b, s)));
            for (int s = 3; s < 6; ++s)
                strain_v(b, s) = Kokkos::fmax(-STRAIN_MAX_SHEAR,
                                              Kokkos::fmin(STRAIN_MAX_SHEAR, strain_v(b, s)));

            RealType eps[3][3] = {};
            eps[0][0] = strain_v(b,0);
            eps[1][1] = strain_v(b,1);
            eps[2][2] = strain_v(b,2);
            eps[1][2] = eps[2][1] = RealType(0.5) * strain_v(b,3);
            eps[0][2] = eps[2][0] = RealType(0.5) * strain_v(b,4);
            eps[0][1] = eps[1][0] = RealType(0.5) * strain_v(b,5);

            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    RealType v = 0;
                    for (int k = 0; k < 3; ++k)
                        v += ((r == k ? RealType(1) : RealType(0)) + eps[r][k]) * cell_ref(b, k, c);
                    b_cell(b, r, c) = v;
                }

            RealType cm[3][3], ci[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    cm[r][c] = b_cell(b, r, c);

            RealType det = mat3_det(cm);
            if (det < RealType(0.1) || Kokkos::isnan(det)) {
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) {
                        cm[r][c] = cell_ref(b, r, c);
                        b_cell(b, r, c) = cm[r][c];
                    }
                for (int s = 0; s < 6; ++s) strain_v(b, s) = 0;
                det = mat3_det(cm);
            }

            mat3_inv(cm, ci);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    b_ci(b, r, c) = ci[r][c];

            b_vol(b) = Kokkos::abs(det);

            const RealType cutoff = model.real_space_cutoff(b_kappa(b));

            // Grow the hkl box to keep covering the fixed gmax sphere as the cell changes;
            // kappa is held fixed.
            if (has_reciprocal) {
                const RealType gmax = RealType(2) * b_kappa(b) * model.ewald_split;
                int hkl[3];
                reciprocal_index_bounds(cm, gmax, hkl);
                for (int i = 0; i < 3; ++i) b_mhkl(b, i) = hkl[i];
            }

            for (int ax = 0; ax < 3; ++ax) {
                int j = (ax + 1) % 3, k = (ax + 2) % 3;
                RealType cx = cm[1][j]*cm[2][k] - cm[2][j]*cm[1][k];
                RealType cy = cm[2][j]*cm[0][k] - cm[0][j]*cm[2][k];
                RealType cz = cm[0][j]*cm[1][k] - cm[1][j]*cm[0][k];
                RealType cn = Kokkos::sqrt(cx*cx + cy*cy + cz*cz);
                if (cn < RealType(1e-10)) cn = RealType(1e-10);
                RealType d = b_vol(b) / cn;
                b_mshift(b, ax) = int(Kokkos::ceil(cutoff / d)) + 1;
            }
        });
    Space().fence();
}

template void batch_update_cell_from_strain<ExecSpace>(
    const ForceField&, BatchGeometry&, RigidDofState&);

// Ewald rebracketing (see klasp/batch.hpp).
template <typename Space>
void batch_rebracket_ewald(const ForceField& ff, BatchGeometryT<Space>& geom,
                           const SView1D<int, Space>& live,
                           const SView1D<int, Space>& rebracketed) {
    const int n_batch = geom.n_batch;
    const int n_atoms = geom.n_atoms;
    auto b_cell   = geom.cell;
    auto b_vol    = geom.volume;
    auto b_kappa  = geom.kappa;
    auto b_mhkl   = geom.max_hkl;
    auto b_mshift = geom.max_shift;
    const ForceField model = ff;
    const bool has_reciprocal = model.has_reciprocal();
    const bool use_live = live.extent(0) > 0;

    constexpr RealType PI = RealType(M_PI);
    // kappa goes as V^(-1/3), so compare the cube root of the volume ratio.
    const RealType kappa_ratio =
        Kokkos::pow(RealType(EWALD_REBRACKET_VOLUME_RATIO), RealType(1) / RealType(3));

    Kokkos::parallel_for("rebracket_ewald", Kokkos::RangePolicy<Space>(0, n_batch),
        KOKKOS_LAMBDA(const int b) {
            rebracketed(b) = 0;
            if (use_live && live(b) != SLOT_LIVE) return;

            const RealType vol = b_vol(b);
            const RealType kap_now = b_kappa(b);
            if (!(vol > RealType(0)) || !(kap_now > RealType(0))) return;

            const RealType eta = Kokkos::cbrt(
                (static_cast<RealType>(n_atoms) * PI * PI * PI) / (vol * vol));
            const RealType kap_opt = Kokkos::sqrt(eta);

            const RealType r = kap_opt > kap_now ? kap_opt / kap_now : kap_now / kap_opt;
            if (r <= kappa_ratio) return;

            b_kappa(b) = kap_opt;
            rebracketed(b) = 1;

            RealType cm[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) cm[i][j] = b_cell(b, i, j);

            if (has_reciprocal) {
                const RealType gmax = RealType(2) * kap_opt * model.ewald_split;
                int hkl[3];
                reciprocal_index_bounds(cm, gmax, hkl);
                for (int i = 0; i < 3; ++i) b_mhkl(b, i) = hkl[i];
            }

            const RealType cutoff = model.real_space_cutoff(kap_opt);
            for (int ax = 0; ax < 3; ++ax) {
                const int j = (ax + 1) % 3, k = (ax + 2) % 3;
                RealType cx = cm[1][j]*cm[2][k] - cm[2][j]*cm[1][k];
                RealType cy = cm[2][j]*cm[0][k] - cm[0][j]*cm[2][k];
                RealType cz = cm[0][j]*cm[1][k] - cm[1][j]*cm[0][k];
                RealType cn = Kokkos::sqrt(cx*cx + cy*cy + cz*cz);
                if (cn < RealType(1e-10)) cn = RealType(1e-10);
                b_mshift(b, ax) = int(Kokkos::ceil(cutoff / (vol / cn))) + 1;
            }
        });
    Space().fence();
}

template void batch_rebracket_ewald<ExecSpace>(
    const ForceField&, BatchGeometry&, const SView1D<int, ExecSpace>&,
    const SView1D<int, ExecSpace>&);

template <int MaxRank, typename Space>
void batch_expand_asymmetric_unit(const ModelTablesT<Space>& tables,
                                  BatchGeometryT<Space>& geom,
                                  const RigidDofStateT<Space>& dofs) {
    const int n_batch = geom.n_batch;
    const int n_symops = tables.n_symops;
    const int n_sites = tables.n_sites_per_mol;
    const int n_work = n_batch * n_symops * n_sites;

    auto b_trans   = dofs.translations;
    auto b_rots    = dofs.rot_vectors;
    auto b_pos     = geom.positions;
    auto b_q       = geom.charges;
    auto b_types   = geom.types;
    auto b_mp      = geom.multipoles;
    auto b_dip     = geom.dipoles;
    auto b_molid   = geom.mol_id;
    auto body_pos  = tables.body_positions;
    auto body_q    = tables.body_charges;
    auto body_t    = tables.body_types;
    auto body_mp   = tables.body_multipoles;
    auto sym_rot   = tables.sym_rotations;    // fractional (n_symops, 9)
    auto sym_trans = tables.sym_translations;
    auto sym_idx   = tables.sym_indep_index;
    auto b_nsites  = tables.mol_site_count;
    auto b_cell    = geom.cell;             // (batch, 3, 3)
    auto b_ci      = geom.cell_inv;         // (batch, 3, 3)
    auto b_com     = geom.com_positions;    // (batch, n_symops, 3)
    auto b_com_ref = geom.com_positions_ref;
    auto b_cellref = dofs.cell_ref;
    // cell_ref is only set where a strain DOF exists.
    const bool want_ref_com = dofs.n_cell_dof > 0
                           && b_cellref.extent(0) >= static_cast<size_t>(n_batch)
                           && b_com_ref.extent(0) >= static_cast<size_t>(n_batch);
    auto b_live    = geom.slot_mask;        // see BatchGeometryT::slot_mask
    const bool use_live = b_live.extent(0) > 0;

    Kokkos::parallel_for("batch_expand", n_work,
        KOKKOS_LAMBDA(const int flat) {
            // Force capture of all views (nvcc constexpr-if first-capture error).
            (void)body_mp; (void)b_mp; (void)b_dip;

            const int site = flat % n_sites;
            const int rem  = flat / n_sites;
            const int op   = rem % n_symops;
            const int b    = rem / n_symops;
            if (use_live && b_live(b) != SLOT_LIVE) return;

            const int k = sym_idx(op);  // independent molecule index
            const int body_idx = k * n_sites + site;
            const int atom_idx = op * n_sites + site;

            RealType S_frac[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    S_frac[r][c] = sym_rot(op, r*3+c);

            // Fractional COM: S_frac @ frac_trans + t_sym. Do not wrap: it breaks exact
            // symmetry relations between molecules.
            RealType ft[3] = {b_trans(b, k, 0), b_trans(b, k, 1), b_trans(b, k, 2)};
            RealType frac_com[3];
            for (int r = 0; r < 3; ++r) {
                frac_com[r] = S_frac[r][0]*ft[0] + S_frac[r][1]*ft[1] + S_frac[r][2]*ft[2]
                            + sym_trans(op, r);
            }

            RealType comx = b_cell(b,0,0)*frac_com[0] + b_cell(b,0,1)*frac_com[1] + b_cell(b,0,2)*frac_com[2];
            RealType comy = b_cell(b,1,0)*frac_com[0] + b_cell(b,1,1)*frac_com[1] + b_cell(b,1,2)*frac_com[2];
            RealType comz = b_cell(b,2,0)*frac_com[0] + b_cell(b,2,1)*frac_com[1] + b_cell(b,2,2)*frac_com[2];

            if (site == 0) {
                b_com(b, op, 0) = comx;
                b_com(b, op, 1) = comy;
                b_com(b, op, 2) = comz;
                // Same fractional COM through the reference cell (see com_positions_ref).
                if (want_ref_com) {
                    for (int r = 0; r < 3; ++r)
                        b_com_ref(b, op, r) = b_cellref(b,r,0)*frac_com[0]
                                            + b_cellref(b,r,1)*frac_com[1]
                                            + b_cellref(b,r,2)*frac_com[2];
                }
            }

            // S_cart = cell @ S_frac @ cell_inv
            RealType local_cell[3][3], local_ci[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    local_cell[r][c] = b_cell(b,r,c);
                    local_ci[r][c] = b_ci(b,r,c);
                }
            RealType S_cart[3][3];
            frac_to_cart_symop(local_cell, local_ci, S_frac, S_cart);

            // R_total = S_cart @ R_mol
            RealType aa[3] = {b_rots(b, k, 0), b_rots(b, k, 1), b_rots(b, k, 2)};
            RealType R[3][3];
            angle_axis_to_matrix(aa, R);

            RealType R_total[3][3];
            mat3_mul(S_cart, R, R_total);

            RealType bx = body_pos(body_idx, 0);
            RealType by = body_pos(body_idx, 1);
            RealType bz = body_pos(body_idx, 2);
            b_pos(b, atom_idx, 0) = R_total[0][0]*bx + R_total[0][1]*by + R_total[0][2]*bz + comx;
            b_pos(b, atom_idx, 1) = R_total[1][0]*bx + R_total[1][1]*by + R_total[1][2]*bz + comy;
            b_pos(b, atom_idx, 2) = R_total[2][0]*bx + R_total[2][1]*by + R_total[2][2]*bz + comz;

            b_q(b, atom_idx) = body_q(body_idx);
            b_types(b, atom_idx) = body_t(body_idx);
            b_molid(b, atom_idx) = op;  // molecule index = symop index

            if constexpr (MaxRank > 0) {
                constexpr int NMP = nhermsum(MaxRank);
                RealType body_cart[NMP], lab_cart[NMP];
                for (int c = 0; c < NMP; ++c) body_cart[c] = body_mp(body_idx, c);
                rotate_cartesian_multipole_gen<MaxRank>(body_cart, R_total, lab_cart);
                for (int c = 0; c < NMP; ++c) b_mp(b, atom_idx, c) = lab_cart[c];

                b_dip(b, atom_idx, 0) = lab_cart[1];  // x
                b_dip(b, atom_idx, 1) = lab_cart[2];  // y
                b_dip(b, atom_idx, 2) = lab_cart[3];  // z

            }
        });
}

template void batch_expand_asymmetric_unit<0, ExecSpace>(
    const ModelTables&, BatchGeometry&, const RigidDofState&);
template void batch_expand_asymmetric_unit<1, ExecSpace>(
    const ModelTables&, BatchGeometry&, const RigidDofState&);
template void batch_expand_asymmetric_unit<2, ExecSpace>(
    const ModelTables&, BatchGeometry&, const RigidDofState&);
template void batch_expand_asymmetric_unit<3, ExecSpace>(
    const ModelTables&, BatchGeometry&, const RigidDofState&);
template void batch_expand_asymmetric_unit<4, ExecSpace>(
    const ModelTables&, BatchGeometry&, const RigidDofState&);

// Post-pair processing: reciprocal, self, intra and pressure terms.
template <int MaxRank, typename Space>
void batch_post_pairs(const ForceField& ff,
                      const ModelTablesT<Space>& tables,
                      BatchGeometryT<Space>& geom,
                      RigidDofStateT<Space>& dofs,
                      const BatchOptimizeSettings& settings,
                      KernelTimings* timings) {
    const int n_batch = geom.n_batch;
    const int n_atoms = geom.n_atoms;
    const int n_indep = tables.n_indep;
    const int n_symops = tables.n_symops;
    const int n_sites = tables.n_sites_per_mol;
    const int n_cell_dof = dofs.n_cell_dof;
    // Gate on MaxRank too: b_dip is only written when MaxRank > 0 and is never
    // re-zeroed, so a rank-0 pass would read stale dipoles.
    const bool has_dipoles = MaxRank >= 1 && tables.max_multipole_rank >= 1;
    const RealType s_val = ff.ewald_split;
    const bool do_recip = ff.has_reciprocal();
    const bool do_self  = ff.has_self();
    const bool do_intra = ff.has_intra();

    auto b_pos     = geom.positions;
    auto b_q       = geom.charges;
    auto b_mp      = geom.multipoles;
    auto b_dip     = geom.dipoles;
    auto b_molid   = geom.mol_id;
    auto b_cell    = geom.cell;
    auto b_ci      = geom.cell_inv;
    auto b_kappa   = geom.kappa;
    auto b_vol     = geom.volume;
    auto b_mhkl    = geom.max_hkl;
    auto b_E       = geom.energies;
    auto b_dgt     = dofs.d_gt;
    auto b_dgr     = dofs.d_gr;
    auto b_dgs     = dofs.d_g_strain;
    auto b_strain  = dofs.strain;
    auto b_trans   = dofs.translations;
    auto b_rots    = dofs.rot_vectors;
    auto body_pos  = tables.body_positions;
    auto body_mp   = tables.body_multipoles;
    auto sym_rot   = tables.sym_rotations;
    auto sym_trans = tables.sym_translations;
    auto sym_idx   = tables.sym_indep_index;
    auto b_nsites  = tables.mol_site_count;

    const bool is_wolf = ff.is_wolf();
    const RealType wolf_self_shift = ff.wolf_self_shift;
    auto b_live    = geom.slot_mask;        // see BatchGeometryT::slot_mask
    const bool use_live = b_live.extent(0) > 0;

    using team_t = typename Kokkos::TeamPolicy<Space>::member_type;

    // Ewald reciprocal sum (TeamPolicy).
    Kokkos::Timer recip_timer;
    if (do_recip) {
    const bool contract = settings.contract_pairs;
    auto c_jt   = dofs.jac_t;
    auto c_jr   = dofs.jac_r;
    auto c_jm   = dofs.jac_m;
    auto c_dgt  = dofs.d_gt_c;
    auto c_dgr  = dofs.d_gr_c;
    auto sym_idx_r = tables.sym_indep_index;
    const int n_symops_r = tables.n_symops;

    Kokkos::parallel_for("batch_reciprocal",
        Kokkos::TeamPolicy<Space>(n_batch, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_t& team) {
            const int b = team.league_rank();
            if (use_live && b_live(b) != SLOT_LIVE) return;
            const RealType kappa = b_kappa(b);

            int mh = b_mhkl(b,0), mk = b_mhkl(b,1), ml = b_mhkl(b,2);
            int total_hkl = (2*mh+1) * (2*mk+1) * (2*ml+1);

            RealType gmax = 2.0 * kappa * s_val;
            RealType gmax_sq = gmax * gmax;
            // Taper the last part of the gmax sphere so k-vectors crossing it do not step
            // the energy. strain_ad.cpp must apply the same window.
            const RealType k2_on = RealType(EWALD_K_WINDOW_FRAC) * gmax_sq;
            RealType four_kappa_sq = 4.0 * kappa * kappa;
            RealType two_pi_over_vol = 2.0 * M_PI / b_vol(b);

            // One pass per independent molecule when contracting.
            const int n_grad_pass = contract ? n_indep : 1;
            for (int kk = 0; kk < n_grad_pass; ++kk) {
            int k_mult = 0;
            for (int op = 0; op < n_symops_r; ++op) if (sym_idx_r(op) == kk) ++k_mult;

            RecipAcc racc;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, total_hkl),
                [&](int flat_k, RecipAcc& racc_l) {
                    RealType e_sum = 0;
                    int n_mk = 2*mk+1, n_ml = 2*ml+1;
                    int h = flat_k / (n_mk * n_ml) - mh;
                    int rem_k = flat_k % (n_mk * n_ml);
                    int k_idx = rem_k / n_ml - mk;
                    int l = rem_k % n_ml - ml;
                    if (h == 0 && k_idx == 0 && l == 0) return;

                    RealType kx = 2.0*M_PI*(b_ci(b,0,0)*h + b_ci(b,1,0)*k_idx + b_ci(b,2,0)*l);
                    RealType ky = 2.0*M_PI*(b_ci(b,0,1)*h + b_ci(b,1,1)*k_idx + b_ci(b,2,1)*l);
                    RealType kz = 2.0*M_PI*(b_ci(b,0,2)*h + b_ci(b,1,2)*k_idx + b_ci(b,2,2)*l);
                    RealType k2 = kx*kx + ky*ky + kz*kz;

                    if (k2 >= gmax_sq) return;

                    RealType pref = two_pi_over_vol * Kokkos::exp(-k2 / four_kappa_sq) / k2;
                    if (k2 > k2_on) pref *= ewald_k_window(k2, k2_on, gmax_sq);

                    // Structure factor over all atoms.
                    RealType Sq_re = 0.0, Sq_im = 0.0;
                    RealType Smu_re = 0.0, Smu_im = 0.0;
                    for (int a = 0; a < n_atoms; ++a) {
                        if (atom_is_padding(a, n_sites, b_nsites)) continue;
                        RealType kr = kx*b_pos(b,a,0) + ky*b_pos(b,a,1) + kz*b_pos(b,a,2);
                        RealType si_a, ci_a;
                        fast_sincos(kr, si_a, ci_a);
                        Sq_re += b_q(b,a)*ci_a;
                        Sq_im += b_q(b,a)*si_a;
                        if (has_dipoles) {
                            RealType kdmu = kx*b_dip(b,a,0) + ky*b_dip(b,a,1) + kz*b_dip(b,a,2);
                            Smu_re += kdmu*ci_a;
                            Smu_im += kdmu*si_a;
                        }
                    }

                    RealType s2 = Sq_re*Sq_re + Sq_im*Sq_im;
                    if (has_dipoles) {
                        s2 += -2.0*(Sq_re*Smu_im - Sq_im*Smu_re)
                            + Smu_re*Smu_re + Smu_im*Smu_im;
                    }

                    RealType P = has_dipoles ? Sq_re - Smu_im : Sq_re;
                    RealType Q = has_dipoles ? Sq_im + Smu_re : Sq_im;
                    RealType pf = 2.0 * pref;

                    // Contracting needs only this molecule's identity atoms; scattering needs mol_0
                    // for Z'=1 and every atom otherwise.
                    const int a_lo = contract ? kk * n_sites : 0;
                    const int a_hi = contract ? (kk + 1) * n_sites
                                              : ((n_indep == 1) ? n_sites : n_atoms);
                    for (int a = a_lo; a < a_hi; ++a) {
                        RealType kr = kx*b_pos(b,a,0) + ky*b_pos(b,a,1) + kz*b_pos(b,a,2);
                        RealType si, ci_val;
                        fast_sincos(kr, si, ci_val);
                        RealType qa = b_q(b,a);
                        RealType mi = 0.0;
                        if (has_dipoles)
                            mi = kx*b_dip(b,a,0) + ky*b_dip(b,a,1) + kz*b_dip(b,a,2);

                        RealType A_val = qa*(P*si - Q*ci_val) + mi*(P*ci_val + Q*si);
                        const RealType fx = pf*A_val*kx, fy = pf*A_val*ky, fz = pf*A_val*kz;
                        RealType dmu_coeff = 0;
                        if (has_dipoles) dmu_coeff = pf*(Q*ci_val - P*si);

                        {
                            for (int m = 0; m < 3; ++m) {
                                racc_l.g[m]   -= fx*c_jt(b,kk,m*3+0) + fy*c_jt(b,kk,m*3+1)
                                               + fz*c_jt(b,kk,m*3+2);
                                racc_l.g[3+m] -= fx*c_jr(b,a,m*3+0) + fy*c_jr(b,a,m*3+1)
                                               + fz*c_jr(b,a,m*3+2);
                            }
                            if (has_dipoles)
                                for (int m = 0; m < 3; ++m)
                                    racc_l.g[3+m] += dmu_coeff
                                        * (kx*c_jm(b,a,m*35+1) + ky*c_jm(b,a,m*35+2)
                                           + kz*c_jm(b,a,m*35+3));
                        }
                    }

                    e_sum += pref * s2;

                    // Reciprocal strain is handled by the AD kernel. Count the energy in one pass only.
                    if (kk == 0) racc_l.e += e_sum;
                }, racc);
            // Single writer per address (b is the league index); no atomic needed.
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                b_E(b) += racc.e;
                if (contract)
                    for (int d = 0; d < 3; ++d) {
                        c_dgt(b, kk, d) += racc.g[d]     * RealType(k_mult);
                        c_dgr(b, kk, d) += racc.g[3 + d] * RealType(k_mult);
                    }
            });
            }   // per-molecule pass
        });
    Space().fence();

    if (n_cell_dof > 0)
        batch_recip_strain_ad<Space>(tables, geom, dofs, has_dipoles);

    } // end if (do_recip)

    if (timings) timings->recip_ms += recip_timer.seconds() * 1000.0;
    Kokkos::Timer post_timer;

    // Self energy, intramolecular erf correction, combine.
    {
        Kokkos::TeamPolicy<Space> policy4(n_batch, Kokkos::AUTO);

        Kokkos::parallel_for("batch_post", policy4,
            KOKKOS_LAMBDA(const team_t& team) {
                (void)body_mp; (void)b_mp; (void)b_dip;

                const int b = team.league_rank();
                if (use_live && b_live(b) != SLOT_LIVE) return;
                const RealType kappa = b_kappa(b);

                // Self energy:
                //   Ewald:   E = -κ/√π · Σ q²
                //   Wolf:    E = -(κ/√π + erfc(κRc)/(2Rc)) · Σ q²
                const int n_self_atoms = (n_indep == 1) ? n_sites : n_atoms;
                RealType E_self = 0, E_intra = 0;
                if (do_self || do_intra) {
                if (do_self) {
                RealType sum_q2 = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_self_atoms),
                    [&](int a, RealType& s) {
                        s += b_q(b,a)*b_q(b,a);
                    }, sum_q2);
                if (n_indep == 1) sum_q2 *= n_symops;
                E_self = -kappa/Kokkos::sqrt(RealType(M_PI))*sum_q2;
                if (is_wolf) E_self -= wolf_self_shift * sum_q2;

                if (has_dipoles) {
                    RealType sum_mu2 = 0.0;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_self_atoms),
                        [&](int a, RealType& s) {
                            s += b_dip(b,a,0)*b_dip(b,a,0)
                               + b_dip(b,a,1)*b_dip(b,a,1)
                               + b_dip(b,a,2)*b_dip(b,a,2);
                        }, sum_mu2);
                    if (n_indep == 1) sum_mu2 *= n_symops;
                    E_self -= (2.0*kappa*kappa*kappa/(3.0*Kokkos::sqrt(M_PI)))*sum_mu2;

                    // No dE/dmu: the self energy depends only on |mu|^2, so its rigid-body gradient
                    // is identically zero.
                }
                } // do_self
                team.team_barrier();

                // Ewald intramolecular erf correction.
                if (do_intra) {
                const int n_intra_atoms = (n_indep == 1) ? n_sites : n_atoms;
                // Gather: each thread owns atom i and sums over all j, so no atomics. The energy
                // is counted on i < j only.
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_intra_atoms),
                    [&](int i, RealType& e_intra) {
                      for (int j = 0; j < n_intra_atoms; ++j) {
                        if (j == i) continue;
                        const bool count_energy = (i < j);

                        // For Z'=1 all pairs are intramolecular; for Z'>1 filter by mol_id.
                        if (n_indep > 1) {
                            if (b_molid(b,i) < 0 || b_molid(b,i) != b_molid(b,j)) continue;
                        }
                        RealType rx = b_pos(b,j,0)-b_pos(b,i,0);
                        RealType ry = b_pos(b,j,1)-b_pos(b,i,1);
                        RealType rz = b_pos(b,j,2)-b_pos(b,i,2);
                        RealType r2 = rx*rx+ry*ry+rz*rz;
                        if (r2 < 1e-30) continue;
                        RealType r = Kokkos::sqrt(r2);
                        RealType kr = kappa*r;
                        RealType erf_kr = Kokkos::erf(kr);
                        RealType exp_kr2 = Kokkos::exp(-kr*kr);
                        RealType inv_r = 1.0/r;

                        if (count_energy) e_intra += b_q(b,i)*b_q(b,j)*erf_kr*inv_r;

                        RealType two_kappa_sqrtpi = 2.0*kappa/Kokkos::sqrt(M_PI);
                        // No force written: within one rigid molecule the force and multipole halves
                        // cancel exactly in the rigid-body gradient. Drop or restore both together.

                        // No intra virial: intramolecular distances do not change under cell strain.

                        if (has_dipoles) {
                            // Real-space dipole T-tensor algebra on the erf family:
                            //   A1 = -phi'/r, A2 = -A1'/r, A3 = -A2'/r,  phi = erf(kappa r)/r,
                            //   A_{n+1} = (2n+1) A_n/r^2 - (2 kappa^2)^n (2 kappa/sqrt(pi)) exp / r^2.
                            RealType inv_r2 = inv_r*inv_r;
                            RealType inv_r3 = inv_r2*inv_r;
                            RealType inv_r4 = inv_r2*inv_r2;
                            RealType inv_r5 = inv_r4*inv_r;
                            const RealType kappa2 = kappa*kappa;
                            const RealType gauss = two_kappa_sqrtpi*exp_kr2*inv_r2;

                            RealType A1 = erf_kr*inv_r3 - gauss;
                            RealType A2 = 3.0*erf_kr*inv_r5
                                        - 6.0*kappa/Kokkos::sqrt(M_PI)*exp_kr2*inv_r4
                                        - 4.0*kappa2*kappa/Kokkos::sqrt(M_PI)*exp_kr2*inv_r2;
                            RealType A3 = 5.0*A2*inv_r2 - (4.0*kappa2*kappa2)*gauss;

                            const RealType Rv[3] = {rx, ry, rz};
                            const RealType mu_iv[3] = {b_dip(b,i,0), b_dip(b,i,1), b_dip(b,i,2)};
                            const RealType mu_jv[3] = {b_dip(b,j,0), b_dip(b,j,1), b_dip(b,j,2)};
                            RealType e_d, f_i[3], dmp_i[3], dmp_j[3], miR, mjR, mimj;
                            dipole_pair_terms(A1, A2, A2, A3, b_q(b,i), b_q(b,j),
                                              mu_iv, mu_jv, Rv,
                                              e_d, f_i, dmp_i, dmp_j, miR, mjR, mimj);
                            if (count_energy) e_intra += e_d;
                            // Force and multipole adjoint dropped (see above).
                            (void)f_i; (void)dmp_i; (void)dmp_j;
                        }
                      }
                    }, E_intra);
                if (n_indep == 1) E_intra *= n_symops;
                team.team_barrier();

                } // do_intra
                } // end if (do_self || do_intra)

                // Combine forces, scale dE_dmp, accumulate self/intra energy. One writer per address.
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    b_E(b) += E_self - E_intra;
                });
                const int n_combine = n_atoms;  // always combine ALL atoms
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_combine),
                    [&](int a) {
                        (void)a;   // nothing to combine: no per-atom forces exist
                    });
            });
    }
    Space().fence();

    // Optional external pressure: pV added to the energy and pV F^-T (Voigt) to the
    // cumulative-strain gradient, since V = det(F) V_ref.
    if (settings.pressure_gpa != RealType(0) && dofs.n_cell_dof > 0) {
        constexpr RealType GPA_TO_AU = RealType(3.3989e-5);  // GPa → Hartree/Bohr³
        const RealType pressure = settings.pressure_gpa * GPA_TO_AU;
        auto b_E   = geom.energies;
        auto b_dgs = dofs.d_g_strain;
        auto b_vol = geom.volume;
        auto b_cell_p = geom.cell;
        auto b_cellref_p = dofs.cell_ref;
        Kokkos::parallel_for("batch_post_pairs_pV",
            Kokkos::RangePolicy<Space>(0, n_batch),
            KOKKOS_LAMBDA(const int b) {
                const RealType V = b_vol(b);
                b_E(b) += pressure * V;
                // F^-1 = cell_ref cell^-1
                RealType C[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) C[i][j] = b_cell_p(b,i,j);
                const RealType det =
                      C[0][0]*(C[1][1]*C[2][2] - C[1][2]*C[2][1])
                    - C[0][1]*(C[1][0]*C[2][2] - C[1][2]*C[2][0])
                    + C[0][2]*(C[1][0]*C[2][1] - C[1][1]*C[2][0]);
                if (!(Kokkos::fabs(det) > RealType(1e-12))) return;
                const RealType id = RealType(1) / det;
                RealType Ci[3][3];
                Ci[0][0] =  (C[1][1]*C[2][2] - C[1][2]*C[2][1]) * id;
                Ci[0][1] = -(C[0][1]*C[2][2] - C[0][2]*C[2][1]) * id;
                Ci[0][2] =  (C[0][1]*C[1][2] - C[0][2]*C[1][1]) * id;
                Ci[1][0] = -(C[1][0]*C[2][2] - C[1][2]*C[2][0]) * id;
                Ci[1][1] =  (C[0][0]*C[2][2] - C[0][2]*C[2][0]) * id;
                Ci[1][2] = -(C[0][0]*C[1][2] - C[0][2]*C[1][0]) * id;
                Ci[2][0] =  (C[1][0]*C[2][1] - C[1][1]*C[2][0]) * id;
                Ci[2][1] = -(C[0][0]*C[2][1] - C[0][1]*C[2][0]) * id;
                Ci[2][2] =  (C[0][0]*C[1][1] - C[0][1]*C[1][0]) * id;
                RealType Finv[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        RealType acc = 0;
                        for (int k = 0; k < 3; ++k)
                            acc += b_cellref_p(b,i,k) * Ci[k][j];
                        Finv[i][j] = acc;
                    }
                const RealType pV = pressure * V;
                b_dgs(b, 0) += pV * Finv[0][0];
                b_dgs(b, 1) += pV * Finv[1][1];
                b_dgs(b, 2) += pV * Finv[2][2];
                b_dgs(b, 3) += pV * RealType(0.5) * (Finv[2][1] + Finv[1][2]);
                b_dgs(b, 4) += pV * RealType(0.5) * (Finv[2][0] + Finv[0][2]);
                b_dgs(b, 5) += pV * RealType(0.5) * (Finv[1][0] + Finv[0][1]);
            });
        Space().fence();
    }

    // The strain gradient is already with respect to the cumulative strain (COM
    // lever arms from com_positions_ref, reference-frame duals in strain_ad.cpp, pV
    // F^-T above). Do not apply an F^-T fold here: the kernels accumulate sym(W),
    // which has already lost the antisymmetric COM virial that fold needs.

    if (timings) timings->post_ms += post_timer.seconds() * 1000.0;
}

// Rigid-body Jacobians of the asymmetric unit, built for identity copies only:
// a symmetry copy contributes exactly what its identity does
// (tests/test_contraction_collapse.cpp). S_k is applied explicitly since the
// identity-indexed operation is not always I.
template <int MaxRank, typename Space>
void batch_build_jacobians(const ModelTablesT<Space>& tables,
                           const BatchGeometryT<Space>& geom,
                           RigidDofStateT<Space>& dofs) {
    const int n_batch = geom.n_batch;
    const int n_indep = tables.n_indep;
    const int n_sites = tables.n_sites_per_mol;
    const int n_asym  = n_indep * n_sites;

    auto b_cell  = geom.cell;
    auto b_ci    = geom.cell_inv;
    auto b_rots  = dofs.rot_vectors;
    auto body_pos = tables.body_positions;
    auto body_mp  = tables.body_multipoles;
    auto sym_rot  = tables.sym_rotations;
    auto sym_idx  = tables.sym_indep_index;
    auto n_symops = tables.n_symops;
    auto Jt = dofs.jac_t;
    auto Jr = dofs.jac_r;
    auto Jm = dofs.jac_m;
    auto b_live = geom.slot_mask;
    const bool use_live = b_live.extent(0) > 0;

    Kokkos::parallel_for("batch_build_jacobians",
        Kokkos::RangePolicy<Space>(0, n_batch * n_asym),
        KOKKOS_LAMBDA(const int w) {
            const int a = w % n_asym;
            const int b = w / n_asym;
            if (use_live && b_live(b) != SLOT_LIVE) return;
            const int k  = a / n_sites;

            int op_k = -1;
            for (int op = 0; op < n_symops; ++op)
                if (sym_idx(op) == k) { op_k = op; break; }
            if (op_k < 0) return;

            RealType S_frac[3][3], S_cart[3][3], lc[3][3], lci[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    S_frac[r][c] = sym_rot(op_k, r*3+c);
                    lc[r][c]  = b_cell(b,r,c);
                    lci[r][c] = b_ci(b,r,c);
                }
            frac_to_cart_symop(lc, lci, S_frac, S_cart);

            RealType aa[3] = {b_rots(b,k,0), b_rots(b,k,1), b_rots(b,k,2)};
            RealType R[3][3], dR0[3][3], dR1[3][3], dR2[3][3];
            angle_axis_to_matrix(aa, R);
            angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);

            // Translation: one per molecule, written by the first site.
            if (a % n_sites == 0)
                for (int d = 0; d < 3; ++d)
                    for (int r = 0; r < 3; ++r)
                        Jt(b, k, d*3 + r) = S_cart[r][0]*b_cell(b,0,d)
                                          + S_cart[r][1]*b_cell(b,1,d)
                                          + S_cart[r][2]*b_cell(b,2,d);

            // Rotation, force half: S_cart applied to dR^m acting on the body position.
            const RealType bx = body_pos(a,0), by = body_pos(a,1), bz = body_pos(a,2);
            RealType dRb[3][3];
            for (int d = 0; d < 3; ++d) {
                dRb[0][d] = dR0[d][0]*bx + dR0[d][1]*by + dR0[d][2]*bz;
                dRb[1][d] = dR1[d][0]*bx + dR1[d][1]*by + dR1[d][2]*bz;
                dRb[2][d] = dR2[d][0]*bx + dR2[d][1]*by + dR2[d][2]*bz;
            }
            for (int m = 0; m < 3; ++m)
                for (int d = 0; d < 3; ++d)
                    Jr(b, a, m*3 + d) = S_cart[d][0]*dRb[m][0]
                                      + S_cart[d][1]*dRb[m][1]
                                      + S_cart[d][2]*dRb[m][2];

            // Rotation, multipole half. Views are named outside the `if constexpr` because
            // nvcc cannot first-capture inside it.
            const auto mp_in  = body_mp;
            const auto mp_out = Jm;
            if constexpr (MaxRank > 0) {
                constexpr int NMP = nhermsum(MaxRank);
                RealType M[3][3];
                mat3_mul(S_cart, R, M);
                RealType body_cart[NMP];
                for (int c = 0; c < NMP; ++c) body_cart[c] = mp_in(a, c);
                // Rotate once, then differentiate the rotated multipoles with the generator
                // dM M^-1 (rotate_cartesian_multipole_generator).
                RealType lab_cart[NMP];
                rotate_cartesian_multipole_gen<MaxRank>(body_cart, M, lab_cart);
                RealType Minv[3][3];
                if (!mat3_inv(M, Minv)) {
                    // Singular M (corrupt cell or operator): write zeros.
                    for (int m = 0; m < 3; ++m)
                        for (int c = 0; c < NMP; ++c) mp_out(b, a, m*35 + c) = RealType(0);
                } else {
                    for (int m = 0; m < 3; ++m) {
                        RealType dM[3][3], Omega[3][3];
                        RealType (*dR)[3] = (m==0) ? dR0 : (m==1) ? dR1 : dR2;
                        mat3_mul(S_cart, dR, dM);
                        mat3_mul(dM, Minv, Omega);
                        RealType dlab[NMP];
                        rotate_cartesian_multipole_generator(lab_cart, Omega, MaxRank, dlab);
                        for (int c = 0; c < NMP; ++c) mp_out(b, a, m*35 + c) = dlab[c];
                    }
                }
            }
        });
    Space().fence();
}

template void batch_post_pairs<0, ExecSpace>(const ForceField&, const ModelTables&, BatchGeometry&,
                                          RigidDofState&,
                                   const BatchOptimizeSettings&, KernelTimings*);
template void batch_post_pairs<1, ExecSpace>(const ForceField&, const ModelTables&, BatchGeometry&,
                                          RigidDofState&,
                                   const BatchOptimizeSettings&, KernelTimings*);
template void batch_post_pairs<2, ExecSpace>(const ForceField&, const ModelTables&, BatchGeometry&,
                                          RigidDofState&,
                                   const BatchOptimizeSettings&, KernelTimings*);
template void batch_post_pairs<3, ExecSpace>(const ForceField&, const ModelTables&, BatchGeometry&,
                                          RigidDofState&,
                                   const BatchOptimizeSettings&, KernelTimings*);
template void batch_post_pairs<4, ExecSpace>(const ForceField&, const ModelTables&, BatchGeometry&,
                                          RigidDofState&,
                                   const BatchOptimizeSettings&, KernelTimings*);

template void batch_build_jacobians<0, ExecSpace>(
    const ModelTables&, const BatchGeometry&, RigidDofState&);
template void batch_build_jacobians<1, ExecSpace>(
    const ModelTables&, const BatchGeometry&, RigidDofState&);
template void batch_build_jacobians<2, ExecSpace>(
    const ModelTables&, const BatchGeometry&, RigidDofState&);
template void batch_build_jacobians<3, ExecSpace>(
    const ModelTables&, const BatchGeometry&, RigidDofState&);
template void batch_build_jacobians<4, ExecSpace>(
    const ModelTables&, const BatchGeometry&, RigidDofState&);

}  // namespace klasp
