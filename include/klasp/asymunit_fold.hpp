#pragma once

// The production energy model folded over the asymmetric-unit pair enumeration,
// and the force-constant seed that must agree with it.
//
// Each PairModel is instantiated in its own translation unit (Buckingham in
// batch_kernels_asymunit.cpp, tabulated in batch_kernels_tabulated.cpp): GCC's
// FMA contraction is per-TU, and sharing one changes Buckingham energies' last bits.

#include "klasp/batch.hpp"
#include "klasp/constants.hpp"
#include "klasp/energy_terms.hpp"
#include "klasp/pair_iteration.hpp"
#include "klasp/pair_model.hpp"
#include "klasp/taper.hpp"

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace klasp {

// The energy model folded over each pair: short-range potential, real-space
// Ewald, and rank >= 2 multipoles. Per-structure constants (cutoffs, tapers)
// live in Batch so they are not recomputed per pair.
template <int MaxRank, typename Space, typename PairModel, typename Real = RealType>
struct AsymunitFold {
    PairModel pair_model;                   // the short-range potential
    SView1D<RealType, Space> kappa_of;      // geom.kappa
    WolfDSF wolf;
    RealType s_val;
    RealType taper_on, taper_off;
    RealType mp_cut2;                       // squared --mp-cutoff, 0 = disabled
    // Which terms this model has; decided on the host before launch.
    bool do_pair_potential;
    bool do_real_space_electrostatics;
    bool do_multipoles;

    // Per-structure constants. The Ewald real term has its own cutoff and taper
    // window; the pair and multipole terms take the potential's.
    struct Batch {
        RealType rcut_sq;                   // what the scaffolding rejects on
        RealType t_on, t_off;
        RealType kappa;
        RealType two_kappa_sqrtpi;          // ewald_gauss_prefactor(kappa)
        RealType ewald_rcut, ewald_rcut_sq, ewald_taper_on;
        RealType mp_on, mp_off;             // the multipole term's own taper, if mp_cut2 > 0

        KOKKOS_INLINE_FUNCTION
        void taper(RealType r, RealType& ft, RealType& dft) const {
            ft = cutoff_spline(r, t_on, t_off);
            dft = cutoff_spline_deriv(r, t_on, t_off);
        }
    };

    KOKKOS_INLINE_FUNCTION
    Batch begin_batch(int b) const {
        const RealType kappa = kappa_of(b);
        const RealType ewald_rcut = s_val / kappa;
        // Ramp the Ewald real term to zero over the last 2 A of its own cutoff,
        // which lies inside the potential's taper window where g.ft is still 1.
        const RealType ewald_taper_on =
            Kokkos::fmax(RealType(0.5) * ewald_rcut,
                         ewald_rcut - RealType(2.0 * ANGSTROM_TO_BOHR));
        const RealType rcut = (taper_off > ewald_rcut) ? taper_off : ewald_rcut;
        // Without a configured taper, pair and multipole terms are cut sharply at rcut.
        RealType t_on, t_off;
        effective_taper(taper_on, taper_off, rcut,
                        RealType(2.0 * ANGSTROM_TO_BOHR), t_on, t_off);
        // A shorter multipole cutoff ramps down over its own last 2 A.
        RealType mp_on = 0, mp_off = 0;
        if (mp_cut2 > 0) {
            mp_off = Kokkos::sqrt(mp_cut2);
            mp_on = Kokkos::fmax(RealType(0.5) * mp_off,
                                 mp_off - RealType(2.0 * ANGSTROM_TO_BOHR));
        }
        return Batch{rcut * rcut, t_on, t_off, kappa, ewald_gauss_prefactor(kappa),
                     ewald_rcut, ewald_rcut * ewald_rcut, ewald_taper_on,
                     mp_on, mp_off};
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Batch& bc, const PairGeom& pg,
                    const AtomStateI& ai, const AtomStateJ& aj,
                    PairAcc& pa) const {
        // Short-range pair potential
        if (do_pair_potential) {
            typename PairPotentialTerm<PairModel>::Params bp{pair_model};
            PairPotentialTerm<PairModel>{}.evaluate(pg, ai, aj, bp, pa);
        }

        // Ewald real space: QQ, dipole-charge, dipole-dipole (Wolf DSF aware).
        if (do_real_space_electrostatics) {
            EwaldRealTerm::Params ep{bc.kappa, bc.two_kappa_sqrtpi, bc.ewald_rcut_sq,
                                     bc.ewald_taper_on, bc.ewald_rcut, wolf};
            EwaldRealTerm{}.evaluate(pg, ai, aj, ep, pa);
        }

        // Multipoles of rank >= 2, tapered to zero at mp_cut2 and skipped beyond.
        if constexpr (MaxRank > 0) {
            if (do_multipoles && (mp_cut2 <= 0 || pg.r2 < mp_cut2)) {
                typename MultipoleHigherTerm<MaxRank, Real>::Params mp{bc.mp_on, bc.mp_off,
                                                                       mp_cut2 > 0};
                MultipoleHigherTerm<MaxRank, Real>{}.evaluate(pg, ai, aj, mp, pa);
            }
        }
    }
};

// One pair pass with the run's short-range model. A template rather than a
// branch in the fold so only one model is live in the kernel body.
template <int MaxRank, typename Space, typename PairModel, typename Real>
static void asymunit_pair_pass_in(const ModelTablesT<Space>& tables,
                                  BatchGeometryT<Space>& geom,
                                  RigidDofStateT<Space>& dofs,
                                  const SView3D<RealType, Space>& c_dgt,
                                  const SView3D<RealType, Space>& c_dgr,
                                  bool has_dipoles, const WolfDSF& wolf,
                                  const ForceField& ff,
                                  const EnergyModelParams& m, RealType mp_cut2,
                                  PairModel model,
                                  SView1D<int, Space> prec_mask, int prec_want,
                                  bool use_pair_moments) {
    AsymunitFold<MaxRank, Space, PairModel, Real> fold{
        model, geom.kappa, wolf, ff.ewald_split, m.taper_on, m.taper_off,
        mp_cut2,
        ff.has_pair_potential(),
        ff.has_real_space_electrostatics(),
        ff.has_multipoles()};
    batch_asymunit_pair_iteration<MaxRank, Space>(
        tables, geom, dofs, c_dgt, c_dgr, has_dipoles,
        std::is_same_v<Real, float> ? "asymunit_pairs_f32" : "asymunit_pairs", fold,
        prec_mask, prec_want, use_pair_moments);
}

// Resolves the multipole scalar type from the model setting: two kernels per rank.
template <int MaxRank, typename Space, typename PairModel>
static void asymunit_pair_pass(const ModelTablesT<Space>& tables,
                               BatchGeometryT<Space>& geom,
                               RigidDofStateT<Space>& dofs,
                               const SView3D<RealType, Space>& c_dgt,
                               const SView3D<RealType, Space>& c_dgr,
                               bool has_dipoles, const WolfDSF& wolf,
                               const ForceField& ff,
                               const EnergyModelParams& m, RealType mp_cut2,
                               PairModel model) {
    const SView1D<int, Space> none;
    // The pair-moment path is double precision only.
    const bool moments = m.mp_pair_moments && MaxRank >= 2;
    if (m.mp_single && MaxRank >= 2 && !moments) {
        if (geom.mp_precision.extent(0) > 0) {
            // One pass per precision, each skipping the other's slots; every slot
            // is touched by exactly one pass.
            asymunit_pair_pass_in<MaxRank, Space, PairModel, float>(
                tables, geom, dofs, c_dgt, c_dgr, has_dipoles, wolf, ff, m, mp_cut2, model,
                geom.mp_precision, 1, false);
            asymunit_pair_pass_in<MaxRank, Space, PairModel, RealType>(
                tables, geom, dofs, c_dgt, c_dgr, has_dipoles, wolf, ff, m, mp_cut2, model,
                geom.mp_precision, 0, false);
        } else {
            asymunit_pair_pass_in<MaxRank, Space, PairModel, float>(
                tables, geom, dofs, c_dgt, c_dgr, has_dipoles, wolf, ff, m, mp_cut2, model,
                none, -1, false);
        }
    } else {
        asymunit_pair_pass_in<MaxRank, Space, PairModel, RealType>(
            tables, geom, dofs, c_dgt, c_dgr, has_dipoles, wolf, ff, m, mp_cut2, model,
            none, -1, moments);
    }
}


// ---------------------------------------------------------------------------
// Per-atom force constants K_a = sum over inter-molecular neighbours of the pair
// Hessian H = fs*I + (V''-fs)/r^2 * (dr⊗dr), written to bws.tr.fc[b, identity atom,
// 0..5] (symmetric 3x3 Voigt). Seeds the trust-region Gauss-Newton Hessian; not
// part of the per-iteration kernel. Requires bws.geom.positions already expanded.
//
// Slots 6..11 hold the contribution from the atom's own molecule's periodic
// images. Rigid translation moves those images too, so the translation block
// contracts with (slots 0..5 - slots 6..11); the rotation block uses the full sum.
//
// Walks a full neighbour shell (no e_weight, no mol_j < k filter), unlike
// batch_asymunit_pair_iteration's half shell, and applies the asymmetric-unit
// scale once at the end.
template <typename Space, typename PairModel>
static void asymunit_force_constants(const ModelTablesT<Space>& tables,
                                  const BatchGeometryT<Space>& geom,
                                  TrustRegionScratchT<Space>& trs,
                                  const BatchOptimizeSettings& settings,
                                  PairModel pair_model) {
    const int nb = geom.n_batch;
    const int n_sites = tables.n_sites_per_mol;
    const int n_atoms = geom.n_atoms;
    const int n_id = tables.n_indep * n_sites;
    const int nsym = tables.n_symops, ni_mol = tables.n_indep;
    const RealType taper_on = settings.model.taper_on, taper_off = settings.model.taper_off;
    const RealType s_val = ewald_split_width(tables.ewald_accuracy);
    auto b_pos = geom.positions; auto b_q = geom.charges; auto b_types = geom.types;
    auto b_kappa = geom.kappa; auto b_cell = geom.cell; auto b_mshift = geom.max_shift;
    auto b_fc = trs.fc;
    auto b_sh = trs.strain_hess;
    auto b_com = geom.com_positions;
    const WolfDSF wolf{};   // plain Ewald for the guess
    // Zero only the slots being reseeded; masked slots keep their block.
    auto b_mask = trs.hess_mask;
    const bool use_mask = b_mask.extent(0) > 0;
    Kokkos::parallel_for("fc_zero", Kokkos::RangePolicy<Space>(0, nb),
        KOKKOS_LAMBDA(const int b) {
            if (use_mask && b_mask(b) == 0) return;
            for (int v = 0; v < 6; ++v)
                for (int w = 0; w < 6; ++w) b_sh(b,v,w) = RealType(0);
            for (int a = 0; a < n_id; ++a)
                for (int t = 0; t < 12; ++t) b_fc(b,a,t) = RealType(0);
        });

    // One thread per (structure, asymmetric-unit atom, target atom); per-atom
    // accumulators therefore go through atomics. sh_partial holds the
    // per-(structure, atom) strain block, summed in index order in a second pass.
    Kokkos::View<RealType***, Space> sh_partial("fc_sh_partial", nb, n_id, 21);
    Kokkos::deep_copy(sh_partial, RealType(0));

    using fc_team_t = typename Kokkos::TeamPolicy<Space>::member_type;
    Kokkos::parallel_for("batch_force_constants",
        Kokkos::TeamPolicy<Space>(nb * n_id, Kokkos::AUTO),
        KOKKOS_LAMBDA(const fc_team_t& fc_team) {
            const int w1 = fc_team.league_rank();
            const int i = w1 % n_id;
            const int b = w1 / n_id;
            const int k = i / n_sites;
            if (use_mask && b_mask(b) == 0) return;
            FcAcc fc_total;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(fc_team, n_atoms),
                [&](int j, FcAcc& fcacc) {
            const RealType pix=b_pos(b,i,0), piy=b_pos(b,i,1), piz=b_pos(b,i,2);
            const RealType cix=b_com(b,k,0), ciy=b_com(b,k,1), ciz=b_com(b,k,2);
            RealType HS[6][6] = {};
            const RealType qi=b_q(b,i); const int ti=b_types(b,i);
            const RealType kappa=b_kappa(b);
            const RealType ewald_rcut = s_val/kappa;
            const RealType rcut = (taper_off > ewald_rcut) ? taper_off : ewald_rcut;
            const RealType rcut_sq = rcut*rcut;
            const int sa=b_mshift(b,0), sb=b_mshift(b,1), sc=b_mshift(b,2);
            RealType K[12] = {};
            {
                const int mj = j / n_sites;
                const int off = (mj == k) ? 6 : 0;   // own periodic images
                const RealType qj=b_q(b,j); const int tj=b_types(b,j);
                for (int ia=-sa; ia<=sa; ++ia)
                for (int ib=-sb; ib<=sb; ++ib)
                for (int ic=-sc; ic<=sc; ++ic) {
                    if (mj==k && ia==0 && ib==0 && ic==0) continue;  // intra (home cell)
                    RealType shx=ia*b_cell(b,0,0)+ib*b_cell(b,0,1)+ic*b_cell(b,0,2);
                    RealType shy=ia*b_cell(b,1,0)+ib*b_cell(b,1,1)+ic*b_cell(b,1,2);
                    RealType shz=ia*b_cell(b,2,0)+ib*b_cell(b,2,1)+ic*b_cell(b,2,2);
                    RealType rx=b_pos(b,j,0)+shx-pix, ry=b_pos(b,j,1)+shy-piy, rz=b_pos(b,j,2)+shz-piz;
                    RealType r2=rx*rx+ry*ry+rz*rz;
                    if (r2>=rcut_sq || r2<RealType(1e-30)) continue;
                    RealType r=Kokkos::sqrt(r2);
                    RealType t_on, t_off;
                    effective_taper(taper_on, taper_off, rcut,
                                    RealType(2.0 * ANGSTROM_TO_BOHR), t_on, t_off);
                    RealType ft=cutoff_spline(r,t_on,t_off), dft=cutoff_spline_deriv(r,t_on,t_off);
                    RealType d2ft=cutoff_spline_deriv2(r,t_on,t_off);
                    // Same model and taper as the gradient kernel.
                    RealType e_raw, de_dr, d2e_raw;
                    const RealType inv_r = RealType(1)/r;
                    pair_model.eval(r, r2, inv_r, ti, tj,
                                    e_raw, de_dr, d2e_raw);
                    RealType e, de_t, d2v_b;
                    apply_taper(e_raw, de_dr, d2e_raw, ft, dft, d2ft, e, de_t, d2v_b);
                    const RealType fs_b = de_t * inv_r;
                    RealType erfcv=Kokkos::erfc(kappa*r), expkr2=Kokkos::exp(-kappa*kappa*r2);
                    RealType e2, fs_q=0, d2v_q=0;
                    pair_ewald_qq(r,inv_r,qi,qj,kappa,ewald_gauss_prefactor(kappa),
                                  erfcv,expkr2,e2,fs_q,wolf,&d2v_q);
                    const RealType fs=fs_b+fs_q, d2v=d2v_b+d2v_q;
                    const RealType a=(d2v-fs)/r2;
                    // Full sum in 0..5; the self-image part also goes into 6..11.
                    const RealType c[6] = {fs+a*rx*rx, fs+a*ry*ry, fs+a*rz*rz,
                                           a*rx*ry, a*rx*rz, a*ry*rz};
                    for (int t=0; t<6; ++t) K[t] += c[t];
                    if (off) for (int t=0; t<6; ++t) K[6+t] += c[t];

                    // Strain second derivative. Molecules are rigid, so only the
                    // image-shifted COM separation u responds to strain:
                    // R'(s) = R + sum_v s_v E_v u, with cell = (I + eps) cell_ref.
                    const RealType ux = b_com(b,mj,0)+shx-cix;
                    const RealType uy = b_com(b,mj,1)+shy-ciy;
                    const RealType uz = b_com(b,mj,2)+shz-ciz;
                    // d_v = E_v u for the six Voigt basis tensors (half on shears,
                    // matching batch_update_cell_from_strain).
                    const RealType h = RealType(0.5);
                    const RealType dv[6][3] = {
                        {ux, 0, 0}, {0, uy, 0}, {0, 0, uz},
                        {0, h*uz, h*uy}, {h*uz, 0, h*ux}, {h*uy, h*ux, 0}};
                    RealType Rd[6];
                    for (int v=0; v<6; ++v)
                        Rd[v] = rx*dv[v][0] + ry*dv[v][1] + rz*dv[v][2];
                    for (int v=0; v<6; ++v)
                        for (int wv=0; wv<=v; ++wv) {
                            const RealType dd = dv[v][0]*dv[wv][0]
                                              + dv[v][1]*dv[wv][1]
                                              + dv[v][2]*dv[wv][2];
                            HS[v][wv] += a*Rd[v]*Rd[wv] + fs*dd;
                        }
                }
            }
            // Accumulates over the j-parallel range; fc_zero cleared the reseeded slots.
            for (int t=0; t<12; ++t) fcacc.k[t] += K[t];
            // Sum over asymmetric-unit atoms, scaled by n_symops/(2 n_indep) as the
            // energy kernel is, to go from asymmetric-unit pairs to a whole cell.
            const RealType scale = RealType(nsym) / (RealType(2) * RealType(ni_mol));
            int t21 = 0;
            for (int v=0; v<6; ++v)
                for (int wv=0; wv<=v; ++wv)
                    fcacc.h[t21++] += scale * HS[v][wv];
                }, fc_total);
            // One writer per address: this team owns (b, i).
            Kokkos::single(Kokkos::PerTeam(fc_team), [&]() {
                for (int t=0; t<12; ++t) b_fc(b,i,t) += fc_total.k[t];
                for (int t=0; t<21; ++t) sh_partial(b,i,t) = fc_total.h[t];
            });
        });
    Space().fence();

    // The strain block, summed over asymmetric-unit atoms in index order.
    Kokkos::parallel_for("batch_force_constants_strain",
        Kokkos::RangePolicy<Space>(0, nb),
        KOKKOS_LAMBDA(const int b) {
            if (use_mask && b_mask(b) == 0) return;
            int t21 = 0;
            for (int v=0; v<6; ++v)
                for (int wv=0; wv<=v; ++wv, ++t21) {
                    RealType hv = 0;
                    for (int i=0; i<n_id; ++i) hv += sh_partial(b,i,t21);
                    b_sh(b,v,wv) += hv;
                    if (wv != v) b_sh(b,wv,v) += hv;
                }
        });

    // Hydrostatic pressure curvature: E_pV = P V_ref det(I + eps), whose
    // second-order part is (s0 s1 + s0 s2 + s1 s2) - (s3^2 + s4^2 + s5^2)/4.
    if (settings.pressure_gpa != RealType(0)) {
        constexpr RealType GPA_TO_AU = RealType(3.3989e-5);
        const RealType pressure = settings.pressure_gpa * GPA_TO_AU;
        auto b_vol = geom.volume;
        auto sh = trs.strain_hess;
        Kokkos::parallel_for("strain_hess_pV", Kokkos::RangePolicy<Space>(0, nb),
            KOKKOS_LAMBDA(const int b) {
                if (use_mask && b_mask(b) == 0) return;
                const RealType pv = pressure * b_vol(b);
                sh(b,0,1) += pv; sh(b,1,0) += pv;
                sh(b,0,2) += pv; sh(b,2,0) += pv;
                sh(b,1,2) += pv; sh(b,2,1) += pv;
                for (int s = 3; s < 6; ++s) sh(b,s,s) -= RealType(0.5) * pv;
            });
    }
}


// The tabulated model's instantiations live in batch_kernels_tabulated.cpp.
template <int MaxRank, typename Space>
void asymunit_pair_pass_tabulated(const ModelTablesT<Space>& tables,
                                  BatchGeometryT<Space>& geom,
                                  RigidDofStateT<Space>& dofs,
                                  const SView3D<RealType, Space>& c_dgt,
                                  const SView3D<RealType, Space>& c_dgr,
                                  bool has_dipoles, const WolfDSF& wolf,
                                  const ForceField& ff,
                                  const EnergyModelParams& m, RealType mp_cut2);

template <typename Space>
void asymunit_force_constants_tabulated(const ModelTablesT<Space>& tables,
                                        const BatchGeometryT<Space>& geom,
                                        TrustRegionScratchT<Space>& trs,
                                        const BatchOptimizeSettings& settings);

}  // namespace klasp
