#include "klasp/batch.hpp"
#include "klasp/padding.hpp"
#include <vector>
#include "klasp/dual6.hpp"
#include "klasp/taper.hpp"
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>

namespace klasp {

using ExecSpace = Kokkos::DefaultExecutionSpace;

// Seeds dG/deps for the cumulative strain from cell_ref, given F^-T (Finvt,
// row-major). h(eps) = (I+eps) h_ref and G = 2 pi h^-T n give
// dG/deps_v = -F^-T (E_v G).
KOKKOS_INLINE_FUNCTION
void set_g_derivs(Dual6& gx, Dual6& gy, Dual6& gz, const RealType Finvt[9]) {
    RealType G[3] = {gx.v, gy.v, gz.v};
    // u[v] = E_v G, the instantaneous seed before the frame change.
    const RealType u[6][3] = {
        {G[0], 0, 0},
        {0, G[1], 0},
        {0, 0, G[2]},
        {0, G[2]*RealType(0.5), G[1]*RealType(0.5)},
        {G[2]*RealType(0.5), 0, G[0]*RealType(0.5)},
        {G[1]*RealType(0.5), G[0]*RealType(0.5), 0}};
    for (int v = 0; v < 6; ++v) {
        gx.g[v] = -(Finvt[0]*u[v][0] + Finvt[1]*u[v][1] + Finvt[2]*u[v][2]);
        gy.g[v] = -(Finvt[3]*u[v][0] + Finvt[4]*u[v][1] + Finvt[5]*u[v][2]);
        gz.g[v] = -(Finvt[6]*u[v][0] + Finvt[7]*u[v][1] + Finvt[8]*u[v][2]);
    }
}

// d(1/V)/deps for the cumulative strain. V = det(F) V_ref, so
// dV/deps_v = V tr(F^-1 E_v) and d(1/V)/deps_v = -(1/V) tr(F^-1 E_v).
KOKKOS_INLINE_FUNCTION
Dual6 make_inv_vol(RealType vol, const RealType Finv[9]) {
    Dual6 iv;
    iv.v = RealType(1) / vol;
    // tr(A E_v): diagonal picks A_vv; a shear picks the mean of the two
    // off-diagonal partners, matching the half-convention in set_g_derivs.
    const RealType tr[6] = {
        Finv[0], Finv[4], Finv[8],
        RealType(0.5) * (Finv[7] + Finv[5]),   // yz
        RealType(0.5) * (Finv[6] + Finv[2]),   // xz
        RealType(0.5) * (Finv[3] + Finv[1])};  // xy
    for (int v = 0; v < 6; ++v) iv.g[v] = -iv.v * tr[v];
    return iv;
}

// Seeds dcom/deps. com = F com_ref, so the seed is E_v com_ref: callers pass
// com_ref in `T`, while values stay in the current frame.
KOKKOS_INLINE_FUNCTION
void set_shift_derivs(Dual6& tx, Dual6& ty, Dual6& tz, const RealType T[3]) {
    tx.g[0] = T[0]; ty.g[0] = 0;      tz.g[0] = 0;
    tx.g[1] = 0;    ty.g[1] = T[1];   tz.g[1] = 0;
    tx.g[2] = 0;    ty.g[2] = 0;      tz.g[2] = T[2];
    tx.g[3] = 0;          ty.g[3] = T[2]*0.5; tz.g[3] = T[1]*0.5;
    tx.g[4] = T[2]*0.5;  ty.g[4] = 0;         tz.g[4] = T[0]*0.5;
    tx.g[5] = T[1]*0.5;  ty.g[5] = T[0]*0.5;  tz.g[5] = 0;
}

// AD reciprocal strain gradient. `has_dipoles` must agree with the pass that
// filled geom.dipoles; see batch_shared.cpp.
template <typename Space>
void batch_recip_strain_ad(const ModelTablesT<Space>& tables,
                           const BatchGeometryT<Space>& geom,
                           RigidDofStateT<Space>& dofs, bool has_dipoles) {
    if (dofs.n_cell_dof == 0) return;
    const int n_batch = geom.n_batch;
    const int n_atoms = geom.n_atoms;
    const RealType s_val = ewald_split_width(tables.ewald_accuracy);
    auto b_pos = geom.positions; auto b_q = geom.charges; auto b_dip = geom.dipoles;
    auto b_molid = geom.mol_id; auto b_com = geom.com_positions;
    auto b_com_ref = geom.com_positions_ref; auto b_cellref = dofs.cell_ref;
    auto b_ci = geom.cell_inv; auto b_kappa = geom.kappa; auto b_vol = geom.volume;
    auto b_mhkl = geom.max_hkl; auto b_dgs = dofs.d_g_strain;
    auto b_live = geom.slot_mask;           // see BatchGeometryT::slot_mask
    const bool use_live = b_live.extent(0) > 0;
    using team_t = typename Kokkos::TeamPolicy<Space>::member_type;

    Kokkos::parallel_for("recip_strain_ad",
        Kokkos::TeamPolicy<Space>(n_batch, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_t& team) {
            const int b = team.league_rank();
            if (use_live && b_live(b) != SLOT_LIVE) return;
            const RealType kappa = b_kappa(b);
            int mh=b_mhkl(b,0), mk=b_mhkl(b,1), ml=b_mhkl(b,2);
            int total_hkl = (2*mh+1)*(2*mk+1)*(2*ml+1);
            RealType gmax_sq = 4.0*kappa*kappa*s_val*s_val;
            const RealType k2_on = RealType(EWALD_K_WINDOW_FRAC) * gmax_sq;
            RealType four_kappa_sq = 4.0*kappa*kappa;
            // F^-T = (cell_ref cell^-1)^T. b_ci holds cell^-1 with the reciprocal
            // basis in its rows.
            RealType Finvt[9], Finv[9];
            {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        RealType acc = 0;
                        for (int kk = 0; kk < 3; ++kk)
                            acc += b_cellref(b,i,kk) * b_ci(b,kk,j);
                        Finv[i*3+j] = acc;
                    }
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) Finvt[i*3+j] = Finv[j*3+i];
            }
            Dual6 inv_vol = make_inv_vol(b_vol(b), Finv);
            Dual6 two_pi_inv_vol = inv_vol * (2.0*M_PI);
            Dual6 E_recip;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, total_hkl),
                [&](int fk, Dual6& acc) {
                    int nmk=2*mk+1, nml=2*ml+1;
                    int h=fk/(nmk*nml)-mh, rk=fk%(nmk*nml);
                    int ki=rk/nml-mk, l=rk%nml-ml;
                    if (h==0&&ki==0&&l==0) return;
                    Dual6 gx,gy,gz;
                    gx.v=2*M_PI*(b_ci(b,0,0)*h+b_ci(b,1,0)*ki+b_ci(b,2,0)*l);
                    gy.v=2*M_PI*(b_ci(b,0,1)*h+b_ci(b,1,1)*ki+b_ci(b,2,1)*l);
                    gz.v=2*M_PI*(b_ci(b,0,2)*h+b_ci(b,1,2)*ki+b_ci(b,2,2)*l);
                    RealType g2v=gx.v*gx.v+gy.v*gy.v+gz.v*gz.v;
                    if (g2v>=gmax_sq) return;
                    set_g_derivs(gx,gy,gz,Finvt);
                    Dual6 g2=gx*gx+gy*gy+gz*gz;
                    Dual6 pref=two_pi_inv_vol*d6_exp(g2*(-RealType(1)/four_kappa_sq))*d6_inv(g2);
                    // Same k-space window as the energy (batch_shared.cpp), applied
                    // to the Dual6 g2 so its dw/deps term is included.
                    if (g2v > k2_on) pref = pref * ewald_k_window(g2, k2_on, gmax_sq);
                    Dual6 Sqr,Sqi,Smr,Smi;
                    for (int a=0;a<n_atoms;++a) {
                        // Phase chain term k.dr/deps, with dr/deps from molecule COM
                        // affine motion (exact for identity symops).
                        Dual6 px, py, pz;
                        px.v = b_pos(b,a,0); py.v = b_pos(b,a,1); pz.v = b_pos(b,a,2);
                        int op = b_molid(b,a);
                        if (op >= 0) {
                            Dual6 cx, cy, cz;
                            cx.v = b_com(b,op,0); cy.v = b_com(b,op,1); cz.v = b_com(b,op,2);
                            const RealType cref[3] = {b_com_ref(b,op,0), b_com_ref(b,op,1),
                                                      b_com_ref(b,op,2)};
                            set_shift_derivs(cx, cy, cz, cref);
                            for (int s = 0; s < 6; ++s) {
                                px.g[s] = cx.g[s];
                                py.g[s] = cy.g[s];
                                pz.g[s] = cz.g[s];
                            }
                        }
                        Dual6 ph=gx*px+gy*py+gz*pz;
                        Dual6 sp,cp; d6_sincos(ph,sp,cp);
                        Sqr+=cp*b_q(b,a); Sqi+=sp*b_q(b,a);
                        if (has_dipoles) {
                            Dual6 kd=gx*b_dip(b,a,0)+gy*b_dip(b,a,1)+gz*b_dip(b,a,2);
                            Smr+=kd*cp; Smi+=kd*sp;
                        }
                    }
                    Dual6 s2=Sqr*Sqr+Sqi*Sqi;
                    if (has_dipoles) {
                        s2+=(Sqr*Smi-Sqi*Smr)*RealType(-2);
                        s2+=Smr*Smr+Smi*Smi;
                    }
                    acc += pref*s2;
                }, E_recip);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                for (int s=0;s<6;++s) b_dgs(b,s)+=E_recip.g[s];
            });
        });
    Space().fence();
}
template void batch_recip_strain_ad<ExecSpace>(const ModelTables&, const BatchGeometry&,
                                               RigidDofState&, bool);

// ---------------------------------------------------------------------------
// Ewald Q-Q diagnostic using the same pair enumeration (half-shell, e_weight)
// as the asymunit kernel. Computes energy and f.cd virial on the host.
// ---------------------------------------------------------------------------
template <typename Space>
void batch_ewald_nl_diagnostic(const ModelTablesT<Space>& tables,
                               const BatchGeometryT<Space>& geom,
                               const RigidDofStateT<Space>& dofs) {
    if (geom.n_batch != 1 || dofs.n_cell_dof == 0) return;

    auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom.positions);
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom.charges);
    auto h_com = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom.com_positions);
    auto h_cell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom.cell);
    auto h_kappa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom.kappa);
    auto h_mshift = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom.max_shift);
    auto h_symidx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tables.sym_indep_index);
    auto h_ns     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tables.mol_site_count);

    const int n_indep = tables.n_indep;
    const int n_mols = tables.n_symops;
    const int n_sites = tables.n_sites_per_mol;
    const RealType kappa = h_kappa(0);
    const RealType s_val = ewald_split_width(tables.ewald_accuracy);
    const RealType ewald_rcut = s_val / kappa;
    const RealType ewald_rcut_sq = ewald_rcut * ewald_rcut;
    const RealType scale = RealType(n_mols) / RealType(2 * n_indep);

    const int sa = h_mshift(0,0), sb = h_mshift(0,1), sc = h_mshift(0,2);
    std::printf("  [NL-exact params] kappa=%.6f ewald_rcut=%.4f sa=%d sb=%d sc=%d n_indep=%d n_mols=%d n_sites=%d\n",
                kappa, ewald_rcut, sa, sb, sc, n_indep, n_mols, n_sites);

    RealType E_total = 0;
    RealType virial_raw[6] = {};
    int n_pairs = 0;
    // Indexed [k*n_mols + mol_j] with k < n_indep and mol_j < n_mols.
    std::vector<int> combo_pairs(static_cast<size_t>(n_indep) * n_mols, 0);

    for (int k = 0; k < n_indep; ++k) {
        for (int mol_j = 0; mol_j < n_mols; ++mol_j) {
            const int j_k = h_symidx(mol_j);
            const bool is_self_image = (j_k == k);

            for (int ia = -sa; ia <= sa; ++ia)
            for (int ib = -sb; ib <= sb; ++ib)
            for (int ic = -sc; ic <= sc; ++ic) {
                if (mol_j == k && ia == 0 && ib == 0 && ic == 0) continue;
                if (is_self_image) {
                    if (mol_j < k) continue;
                    if (mol_j == k) {
                        if (ia < 0) continue;
                        if (ia == 0 && ib < 0) continue;
                        if (ia == 0 && ib == 0 && ic < 0) continue;
                    }
                }
                const bool is_self_shift = (mol_j == k);
                const RealType e_weight = is_self_shift ? 2.0 : 1.0;

                RealType sx = ia*h_cell(0,0,0) + ib*h_cell(0,0,1) + ic*h_cell(0,0,2);
                RealType sy = ia*h_cell(0,1,0) + ib*h_cell(0,1,1) + ic*h_cell(0,1,2);
                RealType sz = ia*h_cell(0,2,0) + ib*h_cell(0,2,1) + ic*h_cell(0,2,2);

                RealType cdx = h_com(0,mol_j,0) - h_com(0,k,0) + sx;
                RealType cdy = h_com(0,mol_j,1) - h_com(0,k,1) + sy;
                RealType cdz = h_com(0,mol_j,2) - h_com(0,k,2) + sz;

                for (int si = 0; si < h_ns(k); ++si) {
                    const int i = k * n_sites + si;
                    const RealType qi = h_q(0, i);
                    for (int sj = 0; sj < h_ns(mol_j); ++sj) {
                        const int j = mol_j * n_sites + sj;
                        RealType rx = h_pos(0,j,0) - h_pos(0,i,0) + sx;
                        RealType ry = h_pos(0,j,1) - h_pos(0,i,1) + sy;
                        RealType rz = h_pos(0,j,2) - h_pos(0,i,2) + sz;
                        RealType r2 = rx*rx + ry*ry + rz*rz;
                        if (r2 >= ewald_rcut_sq || r2 < 1e-30) continue;
                        RealType r = std::sqrt(r2);

                        RealType erfc_kr = std::erfc(kappa * r);
                        RealType exp_kr2 = std::exp(-kappa*kappa*r2);
                        RealType phi = erfc_kr / r;
                        RealType dphi = -erfc_kr/r2 - 2*kappa/std::sqrt(M_PI)*exp_kr2/r;
                        RealType qj = h_q(0, j);
                        RealType e_qq = qi * qj * phi;
                        RealType fs_qq = qi * qj * dphi / r;
                        RealType fw = e_weight * fs_qq;

                        E_total += e_qq * e_weight;
                        virial_raw[0] += fw*rx*cdx;
                        virial_raw[1] += fw*ry*cdy;
                        virial_raw[2] += fw*rz*cdz;
                        virial_raw[3] += 0.5*fw*(ry*cdz + rz*cdy);
                        virial_raw[4] += 0.5*fw*(rx*cdz + rz*cdx);
                        virial_raw[5] += 0.5*fw*(rx*cdy + ry*cdx);
                        n_pairs++;
                        combo_pairs[k*n_mols+mol_j]++;
                        static int nlp = 0;
                        if (nlp < 3) {
                            std::printf("  [NL-pair] k=%d mj=%d i=%d j=%d r=%.4f e_qq=%.8e e_w=%.0f\n",
                                        k, mol_j, i, j, r, e_qq, e_weight);
                            nlp++;
                        }
                    }
                }
            }
        }
    }

    std::printf("  [NL-exact combos] ");
    for (int c = 0; c < n_indep*n_mols; ++c) std::printf("k%d_mj%d=%d ", c/n_mols, c%n_mols, combo_pairs[c]);
    std::printf("\n");
    std::printf("  [NL-exact] %d qq_pairs, E_raw=%.10f E*scale=%.10f\n",
                n_pairs, E_total, E_total * scale);
    std::printf("  [NL-exact virial raw] ");
    for (int s = 0; s < 6; ++s) std::printf("%.6f ", virial_raw[s]);
    std::printf("\n  [NL-exact virial*sc] ");
    for (int s = 0; s < 6; ++s) std::printf("%.6f ", virial_raw[s] * scale);
    std::printf("\n");
}
template void batch_ewald_nl_diagnostic<ExecSpace>(const ModelTables&, const BatchGeometry&,
                                                   const RigidDofState&);

} // namespace klasp
