#pragma once

#include "klasp/energy_terms.hpp"   // EnergyStrainAcc, PairGeom, AtomState*, PairAcc

// The asymmetric-unit pair enumeration. For each independent molecule k, every
// target molecule mol_j in [0, n_symops) at every lattice shift, accumulating
// onto the identity copy of mol_k only (see tests/test_contraction_collapse.cpp).
//
// The scaffolding owns index decoding, the co-crystal padded-slot skip, the
// half-shell filters and their e_weight, the shift vector and COM lever arm,
// the cutoff rejection, building PairGeom / AtomStateI / AtomStateJ / PairAcc,
// and the per-team reduction to energy, strain virial and rigid-body gradients.
// The caller supplies the physics as a fold functor.
//
// Callers zero energies / DOF gradients / strain before the call; everything
// here accumulates.

#include "klasp/types.hpp"
#include "klasp/batch.hpp"

#include <Kokkos_Core.hpp>

namespace klasp {

// The fold functor concept. Everything on it must be KOKKOS_INLINE_FUNCTION; it
// is taken by value as a template parameter so the fold inlines into the kernel.
//
//   struct Fold {
//       // Per-structure constants (cell- or kappa-derived).
//       struct Batch {
//           RealType rcut_sq;    // REQUIRED: pairs at or beyond this are dropped
//                                // before any per-atom state is read
//           KOKKOS_INLINE_FUNCTION
//           void taper(RealType r, RealType& ft, RealType& dft) const;
//       };
//       KOKKOS_INLINE_FUNCTION Batch begin_batch(int b) const;
//       KOKKOS_INLINE_FUNCTION
//       void operator()(const Batch&, const PairGeom&, const AtomStateI&,
//                       const AtomStateJ&, PairAcc&) const;
//   };
//
// `rcut_sq` is applied by the scaffolding before any per-atom loads.
//
// MaxRank and has_dipoles gate which per-atom fields are read; an unallocated
// multipole view must never be indexed, so MaxRank is a template parameter.
template <int MaxRank, typename Space, typename FoldFn>
void batch_asymunit_pair_iteration(const ModelTablesT<Space>& tables,
                                   const BatchGeometryT<Space>& geom,
                                   const RigidDofStateT<Space>& dofs,
                                   const SView3D<RealType, Space>& dof_grad_t,
                                   const SView3D<RealType, Space>& dof_grad_r,
                                   bool has_dipoles,
                                   const char* kernel_label,
                                   FoldFn fold,
                                   SView1D<int, Space> prec_mask = SView1D<int, Space>(),
                                   int prec_want = -1,
                                   bool use_pair_moments = false) {
    const int n_batch  = geom.n_batch;
    const int n_indep  = tables.n_indep;
    const int n_symops = tables.n_symops;
    const int n_sites  = tables.n_sites_per_mol;
    const int n_mols   = n_symops;
    const int n_cell_dof = dofs.n_cell_dof;

    auto b_pos     = geom.positions;
    auto b_q       = geom.charges;
    auto b_types   = geom.types;
    auto b_mp      = geom.multipoles;
    auto b_dip     = geom.dipoles;
    auto b_cell    = geom.cell;
    auto b_E       = geom.energies;
    auto b_mshift  = geom.max_shift;
    auto b_dgs     = dofs.d_g_strain;
    auto b_com_ref = geom.com_positions_ref;
    auto b_cellref = dofs.cell_ref;
    auto sym_idx   = tables.sym_indep_index;
    auto c_jt      = dofs.jac_t;
    auto c_jr      = dofs.jac_r;
    auto c_jm      = dofs.jac_m;
    auto b_nsites  = tables.mol_site_count;
    auto b_dgt     = dof_grad_t;
    auto b_dgr     = dof_grad_r;
    auto b_com     = geom.com_positions;
    auto mol_rad   = tables.mol_radius;
    // Pair multipole moments, built per chunk of site pairs in team scratch and
    // reused across that molecule copy's live shifts.
    constexpr bool moments_possible = MaxRank >= 2;
    const bool have_mom = moments_possible && use_pair_moments;
    using PM = PairMoments<(MaxRank >= 2) ? MaxRank : 2>;
    constexpr int MNT = PM::NT;               // moment entries per side
    constexpr int MROW = 7 * MNT;             // s, three v_m, three w_m
    // Site pairs per chunk, sized to the level-0 scratch budget: 46 KB on CUDA
    // (two 255-register blocks per 100 KB SM), 30 KB elsewhere (host backends cap
    // a team at 32 KB).
#ifdef KOKKOS_ENABLE_CUDA
    constexpr int MOM_BUDGET = std::is_same<Space, Kokkos::Cuda>::value ? 46000 : 30000;
#else
    constexpr int MOM_BUDGET = 30000;
#endif
    constexpr int MCHUNK = (MOM_BUDGET / (MROW * int(sizeof(RealType)))) > 0
                               ? (MOM_BUDGET / (MROW * int(sizeof(RealType)))) : 1;
    // Slot participation (BatchGeometryT::slot_mask): dead slots are skipped.
    auto b_live    = geom.slot_mask;
    const bool use_live = b_live.extent(0) > 0;
    // Precision filter: each slot belongs to exactly one per-precision pass
    // (see asymunit_pair_pass).
    auto b_prec    = prec_mask;
    const bool use_prec = b_prec.extent(0) > 0 && prec_want >= 0;

    using team_t = typename Kokkos::TeamPolicy<Space>::member_type;

    // Scratch for the live (mol_j, shift) list, sized for the largest live slot.
    // Dead slots are excluded: a placeholder cell can have a far larger shift box.
    int n_combos_max = 0;
    Kokkos::parallel_reduce("asymunit_combos_max", Kokkos::RangePolicy<Space>(0, n_batch),
        KOKKOS_LAMBDA(const int b, int& m) {
            if (use_live && b_live(b) != SLOT_LIVE) return;
            if (use_prec && b_prec(b) != prec_want) return;
            const int n = n_mols * (2*b_mshift(b,0)+1) * (2*b_mshift(b,1)+1) * (2*b_mshift(b,2)+1);
            if (n > m) m = n;
        }, Kokkos::Max<int>(n_combos_max));
    // No live slot: the Max reduction is still INT_MIN.
    if (n_combos_max <= 0) return;
    using scratch_t = Kokkos::View<int*, typename Space::scratch_memory_space,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using mscratch_t = Kokkos::View<RealType*, typename Space::scratch_memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const size_t scratch_bytes = scratch_t::shmem_size(n_combos_max)
                               + scratch_t::shmem_size(n_mols + 1);
    const size_t mom_bytes = have_mom ? mscratch_t::shmem_size(MCHUNK * MROW) : 0;

    Kokkos::parallel_for(kernel_label,
        Kokkos::TeamPolicy<Space>(n_batch, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(mom_bytes))
            .set_scratch_size(1, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const team_t& team) {
            const int b = team.league_rank();
            if (use_live && b_live(b) != SLOT_LIVE) return;
            if (use_prec && b_prec(b) != prec_want) return;
            // Declared outside `if constexpr`: nvcc extended lambdas cannot
            // first-capture a variable inside one.
            const int mchunk = MCHUNK, mrow = MROW, mnt = MNT;

            const auto bc = fold.begin_batch(b);
            const RealType rcut_sq = bc.rcut_sq;
            const RealType rcut = Kokkos::sqrt(rcut_sq);

            const int sa = b_mshift(b,0), sb = b_mshift(b,1), sc = b_mshift(b,2);
            const int n_sb = 2*sb+1, n_sc = 2*sc+1;
            const int n_shifts = (2*sa+1) * n_sb * n_sc;
            const int n_combos = n_mols * n_shifts;
            const int n_atom_pairs = n_sites * n_sites;
            scratch_t live(team.team_scratch(1), n_combos_max);
            scratch_t mol_start(team.team_scratch(1), n_mols + 1);
            // Team scratch is handed out once from a forward-only cursor, so allocate
            // it outside the per-molecule loop and reuse it.
            mscratch_t mom_sh = have_mom ? mscratch_t(team.team_scratch(0), mchunk * mrow)
                                         : mscratch_t();

            // One pass per independent molecule, each scaled by its own symmetry
            // multiplicity.
            for (int kk = 0; kk < n_indep; ++kk) {
            const int k = kk;

            // Copies of this molecule (n_symops / n_indep, as checked by
            // validate_symmetry_multiplicity).
            int k_mult = 0;
            for (int op = 0; op < n_symops; ++op) if (sym_idx(op) == kk) ++k_mult;

            // Pass 1: the (mol_j, shift) combinations that can hold a pair inside the
            // cutoff. |com_j + shift - com_k| >= rcut + R_k + R_j puts every atom
            // pair beyond rcut (see mol_radius), so the set of pairs is unchanged.
            // The self-molecule and self-image half-shell skips are applied here too.
            const RealType ckx = b_com(b,k,0), cky = b_com(b,k,1), ckz = b_com(b,k,2);
            const RealType rad_k = mol_rad(kk);
            int n_live = 0;
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, n_combos),
                [&](const int c, int& pos, const bool final_pass) {
                    const int mol_j = c / n_shifts;
                    const int s_flat = c % n_shifts;
                    const int ia = s_flat / (n_sb * n_sc) - sa;
                    const int rem = s_flat % (n_sb * n_sc);
                    const int ib = rem / n_sc - sb;
                    const int ic = rem % n_sc - sc;
                    const int j_k = sym_idx(mol_j);
                    bool ok = true;
                    // Skip: self-molecule at zero shift
                    if (mol_j == k && ia == 0 && ib == 0 && ic == 0) ok = false;
                    // Self-image half-shell
                    if (j_k == k) {
                        if (mol_j < k) ok = false;
                        if (mol_j == k) {
                            if (ia < 0) ok = false;
                            if (ia == 0 && ib < 0) ok = false;
                            if (ia == 0 && ib == 0 && ic < 0) ok = false;
                        }
                    }
                    if (ok) {
                        const RealType dx = b_com(b,mol_j,0) - ckx
                            + RealType(ia)*b_cell(b,0,0) + RealType(ib)*b_cell(b,0,1) + RealType(ic)*b_cell(b,0,2);
                        const RealType dy = b_com(b,mol_j,1) - cky
                            + RealType(ia)*b_cell(b,1,0) + RealType(ib)*b_cell(b,1,1) + RealType(ic)*b_cell(b,1,2);
                        const RealType dz = b_com(b,mol_j,2) - ckz
                            + RealType(ia)*b_cell(b,2,0) + RealType(ib)*b_cell(b,2,1) + RealType(ic)*b_cell(b,2,2);
                        // Slack so rounding cannot drop a pair exactly on the cutoff.
                        const RealType reach = rcut + (rad_k + mol_rad(j_k)) * RealType(1.0 + 1e-9);
                        ok = dx*dx + dy*dy + dz*dz < reach * reach;
                    }
                    if (final_pass && ok) live(pos) = c;
                    if (ok) ++pos;
                }, n_live);
            team.team_barrier();

            // Pass 2: the fold over live combinations x atom pairs. `visit` is the
            // per-pair body; the two loop shapes differ only in the moment source.
            EnergyStrainAcc acc_total;
            auto visit = [&](const int c, const int si, const int sj, EnergyStrainAcc& acc,
                             const RealType* mom, const RealType* wmom) {
                    // Decode: (mol_j, shift)
                    const int mol_j = c / n_shifts;
                    const int s_flat = c % n_shifts;

                    // n_sites is the padded stride; co-crystal padding slots are not
                    // atoms and must be skipped here, before any fold sees them (a ghost
                    // would still be repulsive in WCA12PairFn).
                    if (si >= b_nsites(k) || sj >= b_nsites(mol_j)) return;

                    // Decode shift
                    const int ia = s_flat / (n_sb * n_sc) - sa;
                    const int rem = s_flat % (n_sb * n_sc);
                    const int ib = rem / n_sc - sb;
                    const int ic = rem % n_sc - sc;

                    // Which indep mol does mol_j belong to?
                    const int j_k = sym_idx(mol_j);

                    const bool is_self_shift = (mol_j == k);
                    const RealType e_weight = is_self_shift ? RealType(2.0) : RealType(1.0);

                    // Shift vector
                    RealType sx = RealType(ia)*b_cell(b,0,0) + RealType(ib)*b_cell(b,0,1) + RealType(ic)*b_cell(b,0,2);
                    RealType sy = RealType(ia)*b_cell(b,1,0) + RealType(ib)*b_cell(b,1,1) + RealType(ic)*b_cell(b,1,2);
                    RealType sz = RealType(ia)*b_cell(b,2,0) + RealType(ib)*b_cell(b,2,1) + RealType(ic)*b_cell(b,2,2);

                    const int i = k * n_sites + si;
                    const int j = mol_j * n_sites + sj;

                    // Cutoff test before any per-atom state is loaded.
                    const RealType pix = b_pos(b,i,0), piy = b_pos(b,i,1), piz = b_pos(b,i,2);
                    RealType rx = b_pos(b,j,0) - pix + sx;
                    RealType ry = b_pos(b,j,1) - piy + sy;
                    RealType rz = b_pos(b,j,2) - piz + sz;
                    RealType r2 = rx*rx + ry*ry + rz*rz;
                    if (r2 >= rcut_sq || r2 < RealType(1e-30)) return;

                    // COM displacement for the strain virial in the reference cell (reference
                    // COMs and reference-cell shift). The cell DOF is the cumulative strain
                    // from cell_ref with com = F com_ref, so dE/dF = sum f (x) com_ref; using
                    // current COMs would lose the antisymmetric part of W.
                    RealType cdx = 0, cdy = 0, cdz = 0;
                    if (n_cell_dof > 0) {
                        const RealType rsx = RealType(ia)*b_cellref(b,0,0)
                                           + RealType(ib)*b_cellref(b,0,1)
                                           + RealType(ic)*b_cellref(b,0,2);
                        const RealType rsy = RealType(ia)*b_cellref(b,1,0)
                                           + RealType(ib)*b_cellref(b,1,1)
                                           + RealType(ic)*b_cellref(b,1,2);
                        const RealType rsz = RealType(ia)*b_cellref(b,2,0)
                                           + RealType(ib)*b_cellref(b,2,1)
                                           + RealType(ic)*b_cellref(b,2,2);
                        cdx = b_com_ref(b,mol_j,0) - b_com_ref(b,k,0) + rsx;
                        cdy = b_com_ref(b,mol_j,1) - b_com_ref(b,k,1) + rsy;
                        cdz = b_com_ref(b,mol_j,2) - b_com_ref(b,k,2) + rsz;
                    }

                    const RealType qi = b_q(b,i);
                    const int ta = b_types(b,i);
                    RealType dip_i[3];
                    if (has_dipoles) { dip_i[0]=b_dip(b,i,0); dip_i[1]=b_dip(b,i,1); dip_i[2]=b_dip(b,i,2); }
                    const RealType r = Kokkos::sqrt(r2);
                    // The one reciprocal every term shares (PairGeom::inv_r).
                    const RealType inv_r = RealType(1) / r;

                    RealType ft, dft;
                    bc.taper(r, ft, dft);

                    // Per-pair context; both atoms point straight into the multipole view.
                    PairGeom pg{rx, ry, rz, r, r2, inv_r, ft, dft,
                                cdx, cdy, cdz, e_weight,
                                is_self_shift, n_cell_dof > 0,
                                mom, is_self_shift ? wmom : nullptr};
                    AtomStateI ai{{pix, piy, piz}, qi, ta,
                                  has_dipoles ? dip_i : nullptr,
                                  (MaxRank > 0) ? (&b_mp(b,i,0)) : nullptr,
                                  i};
                    AtomStateJ aj{{b_pos(b,j,0), b_pos(b,j,1), b_pos(b,j,2)},
                                  b_q(b,j), b_types(b,j),
                                  has_dipoles ? (&b_dip(b,j,0)) : nullptr,
                                  (MaxRank > 0) ? (&b_mp(b,j,0)) : nullptr,
                                  j_k * n_sites + sj};
                    // Jacobians for the contracted path. Atom j gets one only when the
                    // partner is this molecule's own identity copy.
                    //
                    // Self-shift pairs take no translation Jacobian on either side: +f on i
                    // and -f on j cancel against the same Jacobian. Both must be skipped.
                    const RealType* jt_i = is_self_shift ? nullptr : &c_jt(b, k, 0);
                    const RealType* jr_i = &c_jr(b, i, 0);
                    const RealType* jr_j = is_self_shift ? &c_jr(b, j, 0) : nullptr;
                    const RealType* jm_i = (MaxRank > 0) ? &c_jm(b, i, 0) : nullptr;
                    const RealType* jm_j = (MaxRank > 0 && is_self_shift)
                                               ? &c_jm(b, j, 0) : nullptr;
                    // Every member is braced explicitly: dmp_i/dmp_j are structs, and brace
                    // elision would put both pointers into dmp_i and drop atom j's adjoint.
                    PairAcc pa{acc,
                               ForceAcc{nullptr, jt_i, jr_i, acc.g},
                               ForceAcc{nullptr, nullptr, jr_j, acc.g},
                               ForceAcc{nullptr, jt_i, jr_i, acc.g},
                               ForceAcc{nullptr, nullptr, jr_j, acc.g},
                               DmpAcc{nullptr, jm_i, acc.g},
                               DmpAcc{nullptr, jm_j, acc.g}};

                    fold(bc, pg, ai, aj, pa);
            };

            if (!have_mom) {
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_live * n_atom_pairs),
                    [&](int flat_local, EnergyStrainAcc& acc) {
                        // Decode: (live combination, atom_i, atom_j)
                        const int atom_flat = flat_local % n_atom_pairs;
                        const int c = live(flat_local / n_atom_pairs);
                        visit(c, atom_flat / n_sites, atom_flat % n_sites, acc, nullptr, nullptr);
                    }, acc_total);
            } else if (have_mom) {
                // With pair moments: build them for a chunk of site pairs in shared
                // memory, run the fold over that chunk x this molecule copy's live
                // shifts, and move on. The live list is ordered by combination, so a
                // copy's shifts are contiguous; mol_start marks the ranges.
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    int m = 0;
                    for (int idx = 0; idx < n_live; ++idx) {
                        const int mj = live(idx) / n_shifts;
                        while (m <= mj) mol_start(m++) = idx;
                    }
                    while (m <= n_mols) mol_start(m++) = n_live;
                });
                team.team_barrier();
                // One team reduction for the whole pass with the chunk barriers inside;
                // the range is exactly the team, so every thread reaches every barrier.
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, team.team_size()),
                    [&](const int tid, EnergyStrainAcc& acc) {
                    const int ts = team.team_size();
                    for (int mol_j = 0; mol_j < n_mols; ++mol_j) {
                        const int L0 = mol_start(mol_j), nL = mol_start(mol_j + 1) - L0;
                        if (nL == 0) continue;
                        // Only the partner's own copy carries a torque, so only its rows hold
                        // the w moments.
                        const bool selfmol = (mol_j == k);
                        const int row_len = selfmol ? mrow : 4 * mnt;
                        const int chunk = (mchunk * mrow) / row_len;
                        for (int p0 = 0; p0 < n_atom_pairs; p0 += chunk) {
                            const int P = (n_atom_pairs - p0 < chunk) ? (n_atom_pairs - p0) : chunk;
                            // Build: one thread per (site pair, moment entry), entry-major so
                            // a warp's lanes mostly share the switch case.
                            for (int item = tid; item < P * mnt; item += ts) {
                                const int t = item % P, e = item / P;
                                const int p = p0 + t;
                                const int si = p / n_sites, sj = p % n_sites;
                                // A padded slot's row is never read: visit rejects it.
                                if (si >= b_nsites(k) || sj >= b_nsites(mol_j)) continue;
                                const int i = k * n_sites + si;
                                const int j = mol_j * n_sites + sj;
                                PM::build_entry(e, &b_mp(b, i, 0), &b_mp(b, j, 0), &c_jm(b, i, 0),
                                                selfmol ? &c_jm(b, j, 0) : nullptr,
                                                &mom_sh(t * row_len));
                            }
                            team.team_barrier();
                            // Use: shift-major, so a warp's lanes pass or fail the cutoff
                            // together.
                            for (int item = tid; item < P * nL; item += ts) {
                                const int t = item % P;
                                const int c = live(L0 + item / P);
                                const int p = p0 + t;
                                const RealType* row = &mom_sh(t * row_len);
                                visit(c, p / n_sites, p % n_sites, acc, row,
                                      selfmol ? row + 4 * mnt : nullptr);
                            }
                            team.team_barrier();
                        }
                    }
                }, acc_total);
            }

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                // The half-shell partial times half this molecule's copy count. Energy
                // and strain share one 7-component reduction; destinations are zeroed by
                // the caller.
                const RealType scale = RealType(k_mult) / RealType(2);
                b_E(b) += acc_total.e * scale;
                if (n_cell_dof > 0)
                    for (int s = 0; s < 6; ++s) b_dgs(b, s) += acc_total.s[s] * scale;
                // Gradients scale by the full copy count: the forces are already whole,
                // unlike the half-shell energy.
                for (int d = 0; d < 3; ++d) {
                    b_dgt(b, kk, d) += acc_total.g[d]     * RealType(k_mult);
                    b_dgr(b, kk, d) += acc_total.g[3 + d] * RealType(k_mult);
                }
            });
            }   // per-molecule pass
        });
    Space().fence();
}

}  // namespace klasp
