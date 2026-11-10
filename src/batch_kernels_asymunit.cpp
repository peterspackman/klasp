// Generalized asymmetric-unit batch energy/gradient kernel, any Z'.
//
// For each independent molecule k, iterates pairs of mol_k against every target
// molecule j at every lattice shift, contracting intermolecular pair forces onto
// the identity copy of mol_k's rigid-body DOFs.

#include "klasp/batch_kernels.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/energy_terms.hpp"
#include "klasp/multipole.hpp"
#include "klasp/asymunit_fold.hpp"
#include "klasp/rotation.hpp"
#include "klasp/taper.hpp"
#include "klasp/constants.hpp"
#include "klasp/trust_region.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace klasp {

template <int MaxRank, typename Space>
void batch_energy_gradient_asymunit(const ModelTablesT<Space>& tables,
                                     BatchGeometryT<Space>& geom,
                                     RigidDofStateT<Space>& dofs,
                                     const BatchOptimizeSettings& settings, int iteration,
                                     KernelTimings* timings) {
    const int n_batch = geom.n_batch;
    const int n_indep = tables.n_indep;

    // Rebuilt every call: they depend on the cell and rotation DOFs.
    batch_build_jacobians<MaxRank, Space>(tables, geom, dofs);
    // Requires MaxRank as well: b_dip is written only when MaxRank > 0, so a
    // rank-0 pass must not read it.
    const bool has_dipoles = MaxRank >= 1 && tables.max_multipole_rank >= 1;
    // Decided here so a misconfigured run throws before writing anything.
    const bool tabulated = use_tabulated_pair_model(settings.model, tables);
    const ForceField ff = make_force_field(tables, settings.model);
    const RealType s_val = ff.ewald_split;
    const TangToenniesDamping buck_damping = settings.model.buck_damping;
    // Optional shorter cutoff (squared) for the rank >= 2 multipole term (--mp-cutoff).
    const RealType mp_cut2 = settings.model.mp_cutoff > 0 ? settings.model.mp_cutoff*settings.model.mp_cutoff : RealType(0);

    auto buck_A    = tables.buck_A;
    auto buck_B    = tables.buck_B;
    auto buck_C    = tables.buck_C;
    auto buck_Rmin = tables.buck_Rmin;
    auto b_E       = geom.energies;
    auto c_dgt     = dofs.d_gt_c;
    auto c_dgr     = dofs.d_gr_c;
    auto b_dgt     = dofs.d_gt;
    auto b_dgr     = dofs.d_gr;
    auto b_dgs     = dofs.d_g_strain;
    const int n_cell_dof = dofs.n_cell_dof;
    // Slot participation (see BatchGeometryT::slot_mask).
    auto b_live    = geom.slot_mask;
    const bool use_live = b_live.extent(0) > 0;

    using team_t = typename Kokkos::TeamPolicy<Space>::member_type;

    Kokkos::Timer pairs_timer;

    // -----------------------------------------------------------------------
    // Kernel 0: Zero scratch arrays
    // -----------------------------------------------------------------------
    Kokkos::parallel_for("asymunit_zero",
        Kokkos::TeamPolicy<Space>(n_batch, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_t& team) {
            const int b = team.league_rank();
            if (use_live && b_live(b) != SLOT_LIVE) return;
            // d_gt_c / d_gr_c are zeroed here; everything downstream accumulates.
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_indep),
                [&](int k) { for (int d = 0; d < 3; ++d) {
                    b_dgt(b,k,d) = 0; b_dgr(b,k,d) = 0;
                    c_dgt(b,k,d) = 0; c_dgr(b,k,d) = 0;
                } });
            if (n_cell_dof > 0)
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    for (int s = 0; s < 6; ++s) b_dgs(b, s) = 0;
                });
            Kokkos::single(Kokkos::PerTeam(team), [&]() { b_E(b) = 0; });
        });
    Space().fence();

    // -----------------------------------------------------------------------
    // Wolf DSF constants (precomputed, shared across batch)
    // -----------------------------------------------------------------------
    WolfDSF wolf_params;
    if (settings.model.wolf) {
        auto h_kappa = Kokkos::create_mirror_view(geom.kappa);
        Kokkos::deep_copy(h_kappa, geom.kappa);
        RealType ref_kappa = h_kappa(0);
        RealType rcut = (settings.model.wolf_cutoff > 0) ? settings.model.wolf_cutoff : s_val / ref_kappa;
        wolf_params = compute_wolf_constants(ref_kappa, rcut);
    }

    // -----------------------------------------------------------------------
    // Kernel 1: Generalized asymmetric-unit pairs (batch_asymunit_pair_iteration).
    // The tabulated model is instantiated in its own translation unit; see
    // asymunit_fold.hpp.
    // -----------------------------------------------------------------------
    if (tabulated)
        asymunit_pair_pass_tabulated<MaxRank, Space>(
            tables, geom, dofs, c_dgt, c_dgr, has_dipoles, wolf_params,
            ff, settings.model, mp_cut2);
    else
        asymunit_pair_pass<MaxRank, Space>(
            tables, geom, dofs, c_dgt, c_dgr, has_dipoles, wolf_params,
            ff, settings.model, mp_cut2,
            BuckinghamModelT<Space>{buck_A, buck_B, buck_C, buck_Rmin,
                                    buck_damping});

    // Forces are not scaled by the energy factor (n_symops/2/n_indep): each is
    // the one-sided intermolecular force the chain rule needs.

    if (timings) timings->pairs_ms += pairs_timer.seconds() * 1000.0;

    // Reciprocal + self + intra energy, force combination, chain rule to d_gt/d_gr.
    batch_post_pairs<MaxRank, Space>(ff, tables, geom, dofs, settings, timings);

    // Fold the contracted gradient (pair loop plus reciprocal sum) into the output.
    {
        auto dgt = dofs.d_gt, dgr = dofs.d_gr;
        auto cgt = dofs.d_gt_c, cgr = dofs.d_gr_c;
        auto mask = geom.slot_mask;
        const bool masked = mask.extent(0) > 0;
        const int ni = n_indep;
        Kokkos::parallel_for("add_contracted_grad",
            Kokkos::RangePolicy<Space>(0, n_batch * ni),
            KOKKOS_LAMBDA(const int w) {
                const int k = w % ni;
                const int b = w / ni;
                if (masked && mask(b) != SLOT_LIVE) return;
                for (int d = 0; d < 3; ++d) {
                    dgt(b, k, d) += cgt(b, k, d);
                    dgr(b, k, d) += cgr(b, k, d);
                }
            });
        Space().fence();
    }
}

template void batch_energy_gradient_asymunit<0, ExecSpace>(const ModelTables&,
                                          BatchGeometry&, RigidDofState&,
                                                const BatchOptimizeSettings&, int,
                                                KernelTimings*);
template void batch_energy_gradient_asymunit<1, ExecSpace>(const ModelTables&,
                                          BatchGeometry&, RigidDofState&,
                                                const BatchOptimizeSettings&, int,
                                                KernelTimings*);
template void batch_energy_gradient_asymunit<2, ExecSpace>(const ModelTables&,
                                          BatchGeometry&, RigidDofState&,
                                                const BatchOptimizeSettings&, int,
                                                KernelTimings*);
template void batch_energy_gradient_asymunit<3, ExecSpace>(const ModelTables&,
                                          BatchGeometry&, RigidDofState&,
                                                const BatchOptimizeSettings&, int,
                                                KernelTimings*);
template void batch_energy_gradient_asymunit<4, ExecSpace>(const ModelTables&,
                                          BatchGeometry&, RigidDofState&,
                                                const BatchOptimizeSettings&, int,
                                                KernelTimings*);

// The same host-side model choice the gradient kernel makes.
template <typename Space>
static void batch_force_constants(const ModelTablesT<Space>& tables,
                                  const BatchGeometryT<Space>& geom,
                                  TrustRegionScratchT<Space>& trs,
                                  const BatchOptimizeSettings& settings) {
    if (use_tabulated_pair_model(settings.model, tables))
        asymunit_force_constants_tabulated<Space>(tables, geom, trs, settings);
    else
        asymunit_force_constants(tables, geom, trs, settings,
                                 BuckinghamModelT<Space>{
                                     tables.buck_A, tables.buck_B, tables.buck_C,
                                     tables.buck_Rmin,
                                     settings.model.buck_damping});
}

// Seed the scaled Hessian for the slots selected by bws.tr.hess_mask (all when
// empty). Shared by the lockstep and streaming trust regions.
//
// The molecular block comes warm from the previous stage's stored block or from
// the Gauss-Newton contraction of the per-atom force constants. The strain block
// is always re-derived at the current geometry from batch_force_constants.
template <int MaxRank, typename Space>
void batch_seed_hessian_blocks(BatchWorkspaceT<Space>& bws,
                               const BatchOptimizeSettings& settings,
                               const tr::StepViews<Space>& sv, int nb,
                               bool warm_mol) {
    // The allocated capacity, not bws.geom.n_batch. Callers that pre-bind these
    // views into `sv` must size them the same way, or sv keeps a stale handle.
    const int cap = bws.geom.energies.extent(0);
    const int ni = bws.tables.n_indep, ns = bws.tables.n_sites_per_mol, nsym = bws.tables.n_symops;
    const int nmd = ni * 6, ncd = bws.dofs.n_cell_dof;
    // Own the scratch, so any caller is safe.
    if ((int)bws.tr.fc.extent(0) != cap || (int)bws.tr.fc.extent(1) != bws.geom.n_atoms
        || (int)bws.tr.fc.extent(2) != 12)
        bws.tr.fc = decltype(bws.tr.fc)("tr_fc", cap, bws.geom.n_atoms, 12);
    if ((int)bws.tr.strain_hess.extent(0) != cap)
        bws.tr.strain_hess = decltype(bws.tr.strain_hess)("tr_strain_hess", cap, 6, 6);
    auto B0 = sv.B0;
    auto Dsc = sv.Dsc;
    auto cellv = bws.geom.cell;
    auto body = bws.tables.body_positions;
    auto rots = bws.dofs.rot_vectors;
    auto b_fc = bws.tr.fc;
    auto sym_rot = bws.tables.sym_rotations;
    auto sym_idx = bws.tables.sym_indep_index;
    auto b_ci = bws.geom.cell_inv;
    auto b_mask = bws.tr.hess_mask;
    const bool use_mask = b_mask.extent(0) > 0;
    (void)cap;
    // Molecular Hessian block (translation + rotation): either a warm start from
    // the previous call's block, or analytic Gauss-Newton
    // B_jj = sum_a (J_a^j)^T K_a J_a^j with K_a the per-atom force constant from
    // bws.tr.fc and J_a the rigid-body Jacobians the chain rule uses.
    {
        batch_force_constants(bws.tables, bws.geom, bws.tr, settings);
        if (!warm_mol) {
            // Fill the per-atom force constants (bws.tr.fc), then for each symop
            // copy of each atom rotate K by S_cart and contract with the
            // translation (S_cart@cell_d) and rotation (S_cart@dRb) Jacobians,
            // reducing six diagonal entries per independent molecule.
            const int gn_work = nsym * ns;
            using gn_team_t = typename Kokkos::TeamPolicy<Space>::member_type;
            Kokkos::parallel_for("tr_gn_hessian",
                Kokkos::TeamPolicy<Space>(nb, Kokkos::AUTO),
                KOKKOS_LAMBDA(const gn_team_t& gn_team) {
                    const int b = gn_team.league_rank();
                    if (use_mask && b_mask(b) == 0) return;
                    for (int kk = 0; kk < ni; ++kk) {
                    DofGrad6 gn_acc;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(gn_team, gn_work),
                        [&](int w, DofGrad6& gacc) {
                    const int si = w % ns;
                    const int op = w / ns;
                    // Only the slots being seeded; a mid-optimisation slot keeps
                    // the block BFGS has been refining.
                    const int k  = sym_idx(op);
                    if (k != kk) return;              // another pass owns this copy
                    const int id_atom = k * ns + si;   // identity copy (where fc lives)

                    // 0..5 full, 6..11 self-image only; translation uses the
                    // difference (see batch_force_constants), rotation the full sum.
                    auto load = [&](int base, RealType M[3][3]) {
                        M[0][0]=b_fc(b,id_atom,base+0);
                        M[1][1]=b_fc(b,id_atom,base+1);
                        M[2][2]=b_fc(b,id_atom,base+2);
                        M[0][1]=M[1][0]=b_fc(b,id_atom,base+3);
                        M[0][2]=M[2][0]=b_fc(b,id_atom,base+4);
                        M[1][2]=M[2][1]=b_fc(b,id_atom,base+5);
                    };
                    RealType K[3][3], Kself[3][3], Ktr[3][3];
                    load(0, K); load(6, Kself);
                    for (int i=0;i<3;++i) for (int j=0;j<3;++j) Ktr[i][j] = K[i][j] - Kself[i][j];

                    RealType aa[3] = {rots(b,k,0), rots(b,k,1), rots(b,k,2)};
                    RealType S_frac[3][3], S_cart[3][3], lc[3][3], lci[3][3];
                    for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
                        S_frac[r][c]=sym_rot(op,r*3+c); lc[r][c]=cellv(b,r,c); lci[r][c]=b_ci(b,r,c);
                    }
                    frac_to_cart_symop(lc, lci, S_frac, S_cart);

                    // K_copy = S_cart K S_cart^T
                    RealType Kc[3][3], Kctr[3][3];
                    auto rotate = [&](const RealType M[3][3], RealType out[3][3]) {
                        RealType M1[3][3];
                        for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
                            RealType s=0; for (int p=0;p<3;++p) s+=S_cart[i][p]*M[p][j]; M1[i][j]=s;
                        }
                        for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
                            RealType s=0; for (int p=0;p<3;++p) s+=M1[i][p]*S_cart[j][p]; out[i][j]=s;
                        }
                    };
                    rotate(K, Kc); rotate(Ktr, Kctr);
                    auto contract = [&](const RealType M[3][3], const RealType J[3]) {
                        RealType KJ[3];
                        for (int i=0;i<3;++i){ RealType s=0; for(int p=0;p<3;++p) s+=M[i][p]*J[p]; KJ[i]=s; }
                        return J[0]*KJ[0]+J[1]*KJ[1]+J[2]*KJ[2];
                    };
                    // translation DOFs: J = S_cart @ cell_d
                    for (int d=0; d<3; ++d) {
                        RealType J[3];
                        for (int r=0;r<3;++r) J[r]=S_cart[r][0]*cellv(b,0,d)+S_cart[r][1]*cellv(b,1,d)+S_cart[r][2]*cellv(b,2,d);
                        gacc.g[d] += contract(Kctr, J);
                    }
                    // rotation DOFs: J = S_cart @ dRb[m]
                    RealType bx=body(k*ns+si,0), by=body(k*ns+si,1), bz=body(k*ns+si,2);
                    RealType dR0[3][3], dR1[3][3], dR2[3][3];
                    angle_axis_to_matrix_deriv(aa, dR0, dR1, dR2);
                    RealType dRb[3][3];
                    for (int d=0; d<3; ++d) {
                        dRb[0][d]=dR0[d][0]*bx+dR0[d][1]*by+dR0[d][2]*bz;
                        dRb[1][d]=dR1[d][0]*bx+dR1[d][1]*by+dR1[d][2]*bz;
                        dRb[2][d]=dR2[d][0]*bx+dR2[d][1]*by+dR2[d][2]*bz;
                    }
                    for (int m=0; m<3; ++m) {
                        RealType J[3];
                        for (int r=0;r<3;++r) J[r]=S_cart[r][0]*dRb[m][0]+S_cart[r][1]*dRb[m][1]+S_cart[r][2]*dRb[m][2];
                        gacc.g[3 + m] += contract(Kc, J);
                    }
                        }, gn_acc);
                    // One writer per address: this team owns b, this pass owns kk.
                    Kokkos::single(Kokkos::PerTeam(gn_team), [&]() {
                        for (int d = 0; d < 3; ++d) {
                            B0(b, kk*3+d)        += gn_acc.g[d];
                            B0(b, ni*3+kk*3+d) += gn_acc.g[3 + d];
                        }
                    });
                    }
                });
            // Scale to the optimizer's scaled coords and floor positive (SPD).
            Kokkos::parallel_for("tr_gn_scale", Kokkos::RangePolicy<Space>(0, nb),
                KOKKOS_LAMBDA(const int b) {
                    if (use_mask && b_mask(b) == 0) return;
                    for (int jc=0; jc<nmd; ++jc) {
                        RealType v = B0(b,jc) / (Dsc(b,jc)*Dsc(b,jc));
                        B0(b,jc) = (v < RealType(1e-3)) ? RealType(1e-3) : v;
                    }
                });
        }
        // Strain block, projected from bws.tr.strain_hess. Masked: strain_hess is
        // only refreshed for the slots being seeded.
        if (ncd > 0)
            Kokkos::parallel_for("tr_seed_strain", Kokkos::RangePolicy<Space>(0, nb),
                KOKKOS_LAMBDA(const int b) {
                    if (use_mask && b_mask(b) == 0) return;
                    tr::seed_strain_hessian(sv, b);
                });
    }
}

// Public wrapper: allocate and fill fc + strain_hess. See batch_kernels.hpp.
template <typename Space>
void batch_strain_hessian(const ModelTablesT<Space>& tables,
                          const BatchGeometryT<Space>& geom,
                          TrustRegionScratchT<Space>& trs,
                          const BatchOptimizeSettings& settings) {
    const int cap = geom.energies.extent(0);
    if ((int)trs.fc.extent(0) != cap || (int)trs.fc.extent(1) != geom.n_atoms
        || (int)trs.fc.extent(2) != 12)
        trs.fc = decltype(trs.fc)("tr_fc", cap, geom.n_atoms, 12);
    if ((int)trs.strain_hess.extent(0) != cap)
        trs.strain_hess = decltype(trs.strain_hess)("tr_strain_hess", cap, 6, 6);
    batch_force_constants(tables, geom, trs, settings);
    Space().fence();
}

template void batch_strain_hessian<ExecSpace>(const ModelTables&, const BatchGeometry&,
                                              TrustRegionScratch&,
                                              const BatchOptimizeSettings&);

template void batch_seed_hessian_blocks<0, ExecSpace>(BatchWorkspace&, const BatchOptimizeSettings&, const tr::StepViews<ExecSpace>&, int, bool);
template void batch_seed_hessian_blocks<1, ExecSpace>(BatchWorkspace&, const BatchOptimizeSettings&, const tr::StepViews<ExecSpace>&, int, bool);
template void batch_seed_hessian_blocks<2, ExecSpace>(BatchWorkspace&, const BatchOptimizeSettings&, const tr::StepViews<ExecSpace>&, int, bool);
template void batch_seed_hessian_blocks<3, ExecSpace>(BatchWorkspace&, const BatchOptimizeSettings&, const tr::StepViews<ExecSpace>&, int, bool);
template void batch_seed_hessian_blocks<4, ExecSpace>(BatchWorkspace&, const BatchOptimizeSettings&, const tr::StepViews<ExecSpace>&, int, bool);

// ---------------------------------------------------------------------------
// Scaled trust-region quasi-Newton step, the stage-2 optimizer. Works in scaled
// coords x̂ = D·x so the radius Δ measures atomic displacement (Bohr): rotation
// DOFs scaled by R_gyr, strain by the longest cell edge, translation by 1. Each
// dogleg step is accepted or rejected on ρ = actual/predicted energy reduction;
// non-finite energy or ρ ≫ 1 rejects and shrinks Δ.
template <int MaxRank, typename Space>
static BatchOptimizeResult batch_trust_region_optimize(
        BatchWorkspaceT<Space>& bws, const BatchOptimizeSettings& settings) {
    Kokkos::Timer timer;
    KernelTimings kt;
    const ForceField ff = make_force_field(bws.tables, settings.model);
    const int nb  = bws.geom.n_batch;
    const int cap = bws.geom.energies.extent(0);
    const int ni  = bws.tables.n_indep;
    const int ns  = bws.tables.n_sites_per_mol;
    const int ncd = bws.dofs.n_cell_dof;
    const int nmd = ni * 6;
    const int ndof = nmd + ncd;

    const tr::TRParams trp;              // shared with the streaming optimiser
    const RealType DELTA0 = trp.delta0;
    const RealType grad_tol  = settings.grad_tol;

    using V1 = Kokkos::View<RealType*, Space>;
    using V2 = Kokkos::View<RealType**, Kokkos::LayoutRight, Space>;
    using V3 = Kokkos::View<RealType***, Kokkos::LayoutRight, Space>;
    V2 Dsc("tr_Dsc", cap, ndof);
    V1 Eacc("tr_Eacc", cap), delta("tr_delta", cap), pred("tr_pred", cap);
    V3 acc_t("tr_acc_t", cap, ni, 3), acc_r("tr_acc_r", cap, ni, 3);
    V2 acc_s("tr_acc_s", cap, 6);
    V1 d_gmax("tr_gmax", cap), d_grms("tr_grms", cap);
    Kokkos::View<int*, Space> d_iters("tr_iters", cap);
    Kokkos::View<int*, Space> d_restarts("tr_restarts", cap);
    auto cellv = bws.geom.cell; auto body = bws.tables.body_positions;
    auto b_nsites = bws.tables.mol_site_count;
    auto trans = bws.dofs.translations; auto rots = bws.dofs.rot_vectors; auto strain = bws.dofs.strain;
    auto dgt = bws.dofs.d_gt, dgr = bws.dofs.d_gr; auto dgs = bws.dofs.d_g_strain;
    auto energies = bws.geom.energies;
    auto b_ci = bws.geom.cell_inv; auto sym_rot = bws.tables.sym_rotations; auto sym_idx = bws.tables.sym_indep_index;
    auto basis = bws.dofs.cell_dof_basis;
    // Rotations that had to be folded back to their principal branch this step;
    // their BFGS secant pair is not usable (see tr::apply_step).
    Kokkos::View<int*, Space> d_wrapped("tr_wrapped", cap);

    // Everything a step reads and writes, shared with the streaming optimiser
    // (klasp/trust_region.hpp).
    tr::StepViews<Space> sv;
    tr::allocate_model(sv, cap, ndof);
    auto B0 = sv.B0;
    auto gacc = sv.gacc; auto pstep = sv.pstep;
    sv.Dsc = Dsc;
    sv.acc_t = acc_t; sv.acc_r = acc_r; sv.acc_s = acc_s;
    sv.trans = trans; sv.rots = rots; sv.strain = strain;
    sv.cell = cellv; sv.cell_ref = bws.dofs.cell_ref; sv.basis = basis;
    sv.dgt = dgt; sv.dgr = dgr; sv.dgs = dgs;
    sv.Eacc = Eacc; sv.delta = delta; sv.pred = pred;
    sv.gmax = d_gmax; sv.grms = d_grms;
    sv.restarts = d_restarts; sv.wrapped = d_wrapped;
    sv.ni = ni; sv.ncd = ncd; sv.nmd = nmd; sv.ndof = ndof;

    // Gauss-Newton force constants, filled once at init: 0..5 count every
    // neighbour, 6..11 only the atom's own periodic images (see
    // batch_force_constants). Reallocated whenever the shape is wrong.
    if ((int)bws.tr.fc.extent(0) != cap || (int)bws.tr.fc.extent(1) != bws.geom.n_atoms
        || (int)bws.tr.fc.extent(2) != 12)
        bws.tr.fc = decltype(bws.tr.fc)("tr_fc", cap, bws.geom.n_atoms, 12);
    auto b_fc = bws.tr.fc;
    // Analytic strain block, filled by the same kernel (see batch.hpp).
    if ((int)bws.tr.strain_hess.extent(0) != cap)
        bws.tr.strain_hess = decltype(bws.tr.strain_hess)("tr_strain_hess", cap, 6, 6);
    // Bound after allocation: reassigning a View above would leave an earlier
    // copy pointing at the old one.
    sv.strain_hess = bws.tr.strain_hess;

    // Per-DOF scaling D from geometry (rigid R_gyr per mol; longest cell edge).
    Kokkos::parallel_for("tr_scaling", Kokkos::RangePolicy<Space>(0, nb),
        KOKKOS_LAMBDA(const int b) {
            tr::compute_dof_scaling(b, ni, ns, nmd, ncd, cellv, body, Dsc, b_nsites);
        });

    // Init: expand, eval, B̂ = I, Δ = Δ0, accepted state = current.
    batch_expand_asymmetric_unit<MaxRank, Space>(bws.tables, bws.geom, bws.dofs);
    { BatchOptimizeSettings es = settings; es.energy_only = false;
      batch_energy_gradient_asymunit<MaxRank>(bws.tables, bws.geom, bws.dofs, es, 0, &kt); }
    Kokkos::parallel_for("tr_init", Kokkos::RangePolicy<Space>(0, nb),
        KOKKOS_LAMBDA(const int b) {
            RealType* g = &sv.mg(b,0);
            tr::pack_scaled_gradient(b, ni, ncd, nmd, dgt, dgr, dgs, basis, Dsc, g);
            for (int i = 0; i < ndof; ++i) gacc(b,i) = g[i];
            for (int k = 0; k < ni; ++k)
                for (int d = 0; d < 3; ++d) {
                    acc_t(b,k,d) = trans(b,k,d); acc_r(b,k,d) = rots(b,k,d);
                }
            if (ncd > 0) {
                RealType e[6], q[6];
                for (int v = 0; v < 6; ++v) e[v] = strain(b,v);
                basis_project(&basis(b,0,0), e, ncd, q);
                for (int s = 0; s < ncd; ++s) acc_s(b,s) = q[s];
            }
            // A non-finite first evaluation gives no usable reference energy; Eacc
            // must be non-finite (not a large finite sentinel) so pack_results
            // drops the structure.
            const RealType E0 = energies(b);
            const bool finite = (E0 == E0) && E0 < RealType(1e300) && E0 > RealType(-1e300);
            Eacc(b) = finite ? E0 : Kokkos::Experimental::infinity_v<RealType>;
            delta(b) = DELTA0; d_iters(b) = -1;
            d_restarts(b) = 0; d_wrapped(b) = 0;
            tr::gradient_norms(g, ndof, d_gmax(b), d_grms(b));
            for (int i = 0; i < ndof; ++i) B0(b,i) = 0;
            tr::model_reset(sv, b);
        });

    const bool warm_mol = settings.reuse_hessian
        && (int)bws.tr.tr_hessian_mol.extent(0) == cap
        && (int)bws.tr.tr_hessian_mol.extent(1) == nmd;
    if (warm_mol) {
        // The warm molecular block source differs per path, so the caller
        // supplies it; the shared seeder owns only the strain block.
        auto Hmol = bws.tr.tr_hessian_mol;
        auto Dsc_w = sv.Dsc;
        Kokkos::parallel_for("tr_reuse_mol", Kokkos::RangePolicy<Space>(0, nb),
            KOKKOS_LAMBDA(const int b) {
                // Re-scale into this call's coordinates: tr_hessian_mol holds the
                // unscaled curvature and B0 = D B D. Only the diagonal is used.
                for (int i = 0; i < nmd; ++i)
                    B0(b,i) = Hmol(b,i,i) * Dsc_w(b,i) * Dsc_w(b,i);
            });
    }
    // Every slot is seeded here, so drop any per-slot mask a streaming call left.
    bws.tr.hess_mask = decltype(bws.tr.hess_mask)();
    batch_seed_hessian_blocks<MaxRank, Space>(bws, settings, sv, nb, warm_mol);

    // Trajectory, if requested: one frame per iteration into device buffers sized
    // for the whole budget, copied back once after the loop.
    const bool record = settings.record_trajectory;
    const int max_frames = settings.n_iterations + 1;
    using V4 = Kokkos::View<RealType****, Kokkos::LayoutRight, Space>;
    V2 rec_E, rec_trial, rec_gmax, rec_delta;
    V3 rec_s;
    V4 rec_t, rec_r;
    if (record) {
        rec_E = V2("tr_rec_E", max_frames, nb);
        rec_trial = V2("tr_rec_trial", max_frames, nb);
        rec_gmax = V2("tr_rec_gmax", max_frames, nb);
        rec_delta = V2("tr_rec_delta", max_frames, nb);
        rec_t = V4("tr_rec_t", max_frames, nb, ni, 3);
        rec_r = V4("tr_rec_r", max_frames, nb, ni, 3);
        if (ncd > 0) rec_s = V3("tr_rec_s", max_frames, nb, 6);
    }
    const auto record_frame = [&](const int frame) {
        if (!record) return;
        auto E = rec_E; auto Et = rec_trial; auto G = rec_gmax; auto D = rec_delta;
        auto T = rec_t; auto R = rec_r; auto S = rec_s;
        const bool with_strain = ncd > 0;
        Kokkos::parallel_for("tr_record", Kokkos::RangePolicy<Space>(0, nb),
            KOKKOS_LAMBDA(const int b) {
                E(frame, b) = Eacc(b);
                Et(frame, b) = energies(b);
                G(frame, b) = d_gmax(b);
                D(frame, b) = delta(b);
                for (int k = 0; k < ni; ++k)
                    for (int d = 0; d < 3; ++d) {
                        T(frame, b, k, d) = trans(b, k, d);
                        R(frame, b, k, d) = rots(b, k, d);
                    }
                if (with_strain)
                    for (int v = 0; v < 6; ++v) S(frame, b, v) = strain(b, v);
            });
    };
    record_frame(0);

    int iters_run = settings.n_iterations;
    for (int iter = 0; iter < settings.n_iterations; ++iter) {
        // PROPOSE: dogleg step from the accepted state.
        Kokkos::parallel_for("tr_propose", Kokkos::RangePolicy<Space>(0, nb),
            KOKKOS_LAMBDA(const int b) {
                RealType pr = tr::model_dogleg(sv, b);
                const RealType t = tr::clip_cell_step(sv, b);
                if (t < RealType(1)) pr = tr::model_scale_step(sv, b, t);
                pred(b) = pr;
                d_wrapped(b) = tr::apply_step(sv, b) ? 1 : 0;
            });
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);
        batch_expand_asymmetric_unit<MaxRank, Space>(bws.tables, bws.geom, bws.dofs);
        { BatchOptimizeSettings es = settings; es.energy_only = false;
          batch_energy_gradient_asymunit<MaxRank>(bws.tables, bws.geom, bws.dofs, es, iter, &kt); }

        // ACCEPT / REJECT.
        Kokkos::parallel_for("tr_accept", Kokkos::RangePolicy<Space>(0, nb),
            KOKKOS_LAMBDA(const int b) {
                tr::advance_step(sv, b, energies(b), trp, trp.precision);
                if (d_iters(b) < 0 && grad_tol > 0 && d_gmax(b) < grad_tol)
                    d_iters(b) = iter;
            });
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);   // sync cell to the (committed/rolled-back) strain
        record_frame(iter + 1);

        // Stop early once every structure meets grad_tol; checked every 8
        // iterations to limit host syncs.
        if (grad_tol > 0 && (iter & 7) == 7) {
            int pending = 0;
            Kokkos::parallel_reduce("tr_pending", Kokkos::RangePolicy<Space>(0, nb),
                KOKKOS_LAMBDA(const int b, int& acc) { if (d_iters(b) < 0) ++acc; },
                pending);
            if (pending == 0) { iters_run = iter + 1; break; }
        }
    }

    // d_gmax / d_grms are the accepted point's scaled gradient.

    // Return the accepted state, not the last trial: energies and positions
    // (which feed the descriptors) must match the rolled-back DOFs.
    {
        auto energies_out = bws.geom.energies;
        Kokkos::parallel_for("tr_commit_energy", Kokkos::RangePolicy<Space>(0, nb),
            KOKKOS_LAMBDA(const int b) { energies_out(b) = Eacc(b); });
        Space().fence();
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);
        batch_expand_asymmetric_unit<MaxRank, Space>(bws.tables, bws.geom, bws.dofs);
        Space().fence();
    }

    // Persist the molecular Hessian block so the next call can warm-start from it.
    if ((int)bws.tr.tr_hessian_mol.extent(0) != cap || (int)bws.tr.tr_hessian_mol.extent(1) != nmd)
        bws.tr.tr_hessian_mol = decltype(bws.tr.tr_hessian_mol)("tr_hessian_mol", cap, nmd, nmd);
    {
        auto Hmol = bws.tr.tr_hessian_mol;
        auto nw = sv.mnw;
        Kokkos::parallel_for("tr_store_mol", Kokkos::RangePolicy<Space>(0, nb),
            KOKKOS_LAMBDA(const int b) {
                // The model's diagonal, stored unscaled.
                tr::model_unscaled_diagonal(sv, b);
                for (int i = 0; i < nmd; ++i)
                    for (int j = 0; j < nmd; ++j) Hmol(b,i,j) = (i == j) ? nw(b,i) : RealType(0);
            });
    }
    Space().fence();

    // create_mirror, not create_mirror_view_and_copy: the latter aliases on host
    // builds, and callers write the returned arrays.
    TrustRegionTrajectory traj;
    if (record) {
        const auto host = [](const auto& v) {
            auto h = Kokkos::create_mirror(Kokkos::HostSpace(), v);
            Kokkos::deep_copy(h, v);
            return h;
        };
        traj.n_frames = iters_run + 1;
        traj.energy = host(rec_E);
        traj.trial_energy = host(rec_trial);
        traj.grad_max = host(rec_gmax);
        traj.trust_radius = host(rec_delta);
        traj.translations = host(rec_t);
        traj.rotations = host(rec_r);
        if (ncd > 0) traj.strain = host(rec_s);
    }

    auto h_E = Kokkos::create_mirror(Kokkos::HostSpace(), bws.geom.energies);
    auto h_t = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.translations);
    auto h_r = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.rot_vectors);
    Kokkos::deep_copy(h_E, bws.geom.energies);
    Kokkos::deep_copy(h_t, bws.dofs.translations);
    Kokkos::deep_copy(h_r, bws.dofs.rot_vectors);
    BatchDiagnostics diag;
    if (settings.store_diagnostics) {
        diag.grad_max = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_gmax);
        diag.grad_rms = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_grms);
        diag.iters_to_converge = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_iters);
        diag.final_delta = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta);
        diag.restarts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_restarts);
    }
    return BatchOptimizeResult{h_E, h_t, h_r, iters_run, bws.geom.n_batch,
                               timer.seconds(), kt, diag, traj};
}

template <int MaxRank, typename Space>
BatchOptimizeResult batch_optimize_asymunit(BatchWorkspaceT<Space>& bws,
                                             const BatchOptimizeSettings& settings_in) {
    const RealType s_val = ewald_split_width(bws.tables.ewald_accuracy);
    BatchOptimizeSettings settings = settings_in;

    // Must run before the optimizer dispatch. Uses slot 0's kappa for the whole
    // batch, which is only right when the slots hold similar cells.
    if (settings.model.wolf) {
        auto h_kappa = Kokkos::create_mirror_view(bws.geom.kappa);
        Kokkos::deep_copy(h_kappa, bws.geom.kappa);
        const RealType ref_kappa = h_kappa(0);
        const RealType rcut = (settings.model.wolf_cutoff > 0) ? settings.model.wolf_cutoff
                                                         : s_val / ref_kappa;
        settings.model.wolf_self_shift = compute_wolf_constants(ref_kappa, rcut).self_shift;
    }

    return batch_trust_region_optimize<MaxRank, Space>(bws, settings);
}

template BatchOptimizeResult batch_optimize_asymunit<0>(BatchWorkspace&,
                                                        const BatchOptimizeSettings&);
template BatchOptimizeResult batch_optimize_asymunit<1>(BatchWorkspace&,
                                                        const BatchOptimizeSettings&);
template BatchOptimizeResult batch_optimize_asymunit<2>(BatchWorkspace&,
                                                        const BatchOptimizeSettings&);
template BatchOptimizeResult batch_optimize_asymunit<3>(BatchWorkspace&,
                                                        const BatchOptimizeSettings&);
template BatchOptimizeResult batch_optimize_asymunit<4>(BatchWorkspace&,
                                                        const BatchOptimizeSettings&);

}  // namespace klasp
