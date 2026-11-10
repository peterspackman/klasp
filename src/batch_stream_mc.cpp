// The Metropolis stepper for the streaming engine (klasp/stream_mc.hpp).
// Kept in its own translation unit so its instantiations do not change FMA
// contraction, and hence energies, in batch_stream.cpp.

#include "klasp/stream_mc.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/constants.hpp"
#include "klasp/log.hpp"
#include "klasp/rotation.hpp"
#include "klasp/strain_basis.hpp"
#include "klasp/stream_engine.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace klasp {

namespace {

// Voigt strain -> deformed cell, matching batch_update_cell_from_strain (clamp
// included), so the prior test sees the cell the evaluator will build.
KOKKOS_INLINE_FUNCTION
void mc_clamp_strain(RealType s[6]) {
    for (int i = 0; i < 3; ++i)
        s[i] = Kokkos::fmax(-STRAIN_MAX_DIAG, Kokkos::fmin(STRAIN_MAX_DIAG, s[i]));
    for (int i = 3; i < 6; ++i)
        s[i] = Kokkos::fmax(-STRAIN_MAX_SHEAR, Kokkos::fmin(STRAIN_MAX_SHEAR, s[i]));
}

KOKKOS_INLINE_FUNCTION
void mc_cell_from_strain(const RealType s[6], const RealType ref[3][3],
                         RealType out[3][3]) {
    RealType eps[3][3] = {};
    eps[0][0] = s[0]; eps[1][1] = s[1]; eps[2][2] = s[2];
    eps[1][2] = eps[2][1] = RealType(0.5) * s[3];
    eps[0][2] = eps[2][0] = RealType(0.5) * s[4];
    eps[0][1] = eps[1][0] = RealType(0.5) * s[5];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            RealType v = 0;
            for (int k = 0; k < 3; ++k)
                v += ((r == k ? RealType(1) : RealType(0)) + eps[r][k]) * ref[k][c];
            out[r][c] = v;
        }
}

// Lattice vectors are the columns of `cell`, in Bohr; lengths are returned in
// Angstrom for the prior's volume budget. Angles in radians.
KOKKOS_INLINE_FUNCTION
void mc_cell_params(const RealType cell[3][3], RealType len_ang[3],
                    RealType& alpha, RealType& beta, RealType& gamma) {
    RealType v[3][3];   // v[j] = lattice vector j
    for (int j = 0; j < 3; ++j)
        for (int r = 0; r < 3; ++r) v[j][r] = cell[r][j];
    RealType n[3];
    for (int j = 0; j < 3; ++j)
        n[j] = Kokkos::sqrt(v[j][0]*v[j][0] + v[j][1]*v[j][1] + v[j][2]*v[j][2]);
    auto ang = [&](int i, int j) {
        if (n[i] <= RealType(0) || n[j] <= RealType(0)) return RealType(0);
        RealType d = (v[i][0]*v[j][0] + v[i][1]*v[j][1] + v[i][2]*v[j][2]) / (n[i]*n[j]);
        d = Kokkos::fmax(RealType(-1), Kokkos::fmin(RealType(1), d));
        return Kokkos::acos(d);
    };
    alpha = ang(1, 2); beta = ang(0, 2); gamma = ang(0, 1);
    for (int j = 0; j < 3; ++j) len_ang[j] = n[j] * RealType(BOHR_TO_ANGSTROM);
}

// A uniform point on the unit sphere from two uniform draws.
KOKKOS_INLINE_FUNCTION
void mc_random_axis(RealType u1, RealType u2, RealType axis[3]) {
    const RealType z = RealType(2) * u1 - RealType(1);
    const RealType phi = RealType(2 * M_PI) * u2;
    const RealType s = Kokkos::sqrt(Kokkos::fmax(RealType(0), RealType(1) - z*z));
    axis[0] = s * Kokkos::cos(phi);
    axis[1] = s * Kokkos::sin(phi);
    axis[2] = z;
}

}  // namespace

// Device functors, one struct per phase so each launch captures only its own
// views. Namespace scope (not anonymous) so CUDA closure types stay nameable.

// Gather pool entry `src` into slot `b` and start its chain.
template <typename Space>
struct McFill {
    // Slot side.
    SView3D<RealType, Space> cellv, cell_ref, b_cinv;
    SView1D<RealType, Space> b_kappa, b_vol;
    SView2D<int, Space> b_hkl, b_shift;
    SView2D<RealType, Space> b_cp, strain;
    SView3D<RealType, Space> trans, rots;
    SView3D<RealType, Space> cur_t, cur_r, best_t, best_r, best_cell;
    SView2D<RealType, Space> cur_s;
    SView3D<RealType, Space> basis;
    SView1D<RealType, Space> E_cur, E_best;
    SView1D<int, Space> mv, nacc, slot_entry;
    // Pool side.
    SView3D<RealType, Space> p_cell, p_cinv, p_tr, p_rv;
    SView1D<RealType, Space> p_kappa, p_vol;
    SView2D<int, Space> p_hkl, p_shift;
    SView2D<RealType, Space> p_cp;
    SView1D<int, Space> p_seeds;
    Kokkos::Random_XorShift64_Pool<Space> rng;
    int ni, ncd, lattice;

    KOKKOS_INLINE_FUNCTION void operator()(const int b, const int src) const {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                cellv(b,r,c)     = p_cell(src,r,c);
                cell_ref(b,r,c)  = p_cell(src,r,c);
                best_cell(b,r,c) = p_cell(src,r,c);
                b_cinv(b,r,c)    = p_cinv(src,r,c);
            }
        b_kappa(b) = p_kappa(src);
        b_vol(b)   = p_vol(src);
        for (int d = 0; d < 3; ++d) { b_hkl(b,d) = p_hkl(src,d); b_shift(b,d) = p_shift(src,d); }
        for (int d = 0; d < 6; ++d) { b_cp(b,d) = p_cp(src,d); strain(b,d) = 0; cur_s(b,d) = 0; }
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                const RealType t = p_tr(src,k,d), r = p_rv(src,k,d);
                trans(b,k,d) = t; cur_t(b,k,d) = t; best_t(b,k,d) = t;
                rots(b,k,d)  = r; cur_r(b,k,d) = r; best_r(b,k,d) = r;
            }
        // The strain basis belongs to the structure, not the slot.
        if (ncd > 0) {
            RealType lc[9], lb[36];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) lc[r*3 + c] = cellv(b,r,c);
            build_strain_basis(lattice, lc, lb, ncd);
            for (int s = 0; s < 6; ++s)
                for (int v = 0; v < 6; ++v) basis(b,s,v) = lb[s*6 + v];
        }
        mv(b) = 0; nacc(b) = 0;
        slot_entry(b) = src;
        E_cur(b)  = Kokkos::Experimental::infinity_v<RealType>;
        E_best(b) = Kokkos::Experimental::infinity_v<RealType>;
        // Seed slot b's RNG state from the structure's seed.
        rng.free_state(Kokkos::Random_XorShift64<Space>(mc_slot_seed(p_seeds(src)), b));
    }
};

// One trial move inside the prior's support. The draw count per move is fixed,
// so the stream position depends only on the move number. A strain step leaving
// the support is refused, not redrawn, which keeps both reproducibility and a
// symmetric proposal (detailed balance).
template <typename Space>
struct McPropose {
    SView3D<RealType, Space> cur_t, cur_r, trans, rots, cell_ref, basis;
    SView2D<RealType, Space> cur_s, strain;
    SView1D<int, Space> mv;
    Kokkos::Random_XorShift64_Pool<Space> rng;
    SamplingPrior prior;
    RealType step_trans, step_rot, step_strain;
    int ni, ncd;

    KOKKOS_INLINE_FUNCTION void operator()(const int b) const {
        if (mv(b) == 0) {
            // First round after a fill: evaluate the starting geometry.
            for (int k = 0; k < ni; ++k)
                for (int d = 0; d < 3; ++d) {
                    trans(b,k,d) = cur_t(b,k,d);
                    rots(b,k,d)  = cur_r(b,k,d);
                }
            for (int s = 0; s < 6; ++s) strain(b,s) = cur_s(b,s);
            return;
        }

        auto gen = rng.get_state(b);

        // Translations: wrapped walk on the unit cube (preserves the uniform prior).
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                RealType t = cur_t(b,k,d)
                           + step_trans * RealType(2 * gen.drand() - 1);
                t = t - Kokkos::floor(t);
                trans(b,k,d) = t;
            }

        // Orientations: small random rotation left-composed with the current one,
        // which leaves the Haar measure invariant.
        for (int k = 0; k < ni; ++k) {
            RealType axis[3];
            mc_random_axis(RealType(gen.drand()), RealType(gen.drand()), axis);
            const RealType ang = step_rot * RealType(2 * gen.drand() - 1);
            RealType dv[3] = {axis[0]*ang, axis[1]*ang, axis[2]*ang};
            RealType Rd[3][3], Rc[3][3], Rn[3][3], aa_cur[3];
            for (int d = 0; d < 3; ++d) aa_cur[d] = cur_r(b,k,d);
            angle_axis_to_matrix(dv, Rd);
            angle_axis_to_matrix(aa_cur, Rc);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    RealType v = 0;
                    for (int m = 0; m < 3; ++m) v += Rd[r][m] * Rc[m][c];
                    Rn[r][c] = v;
                }
            RealType aa_new[3];
            rotation_matrix_to_angle_axis(Rn, aa_new);
            for (int d = 0; d < 3; ++d) rots(b,k,d) = aa_new[d];
        }

        // Strain: only along the lattice-allowed basis rows; the angle window
        // and volume budget are tested on the resulting cell.
        RealType s_new[6];
        for (int s = 0; s < 6; ++s) s_new[s] = cur_s(b,s);
        for (int i = 0; i < ncd; ++i) {
            const RealType c = step_strain * RealType(2 * gen.drand() - 1);
            for (int v = 0; v < 6; ++v) s_new[v] += c * basis(b,i,v);
        }
        rng.free_state(gen);

        if (ncd > 0) {
            mc_clamp_strain(s_new);
            RealType ref[3][3], trial[3][3], len[3], al, be, ga;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) ref[r][c] = cell_ref(b,r,c);
            mc_cell_from_strain(s_new, ref, trial);
            mc_cell_params(trial, len, al, be, ga);
            if (!prior_accepts_cell(prior, len[0], len[1], len[2], al, be, ga)) {
                // Outside the prior: refuse the cell part, keep the pose part.
                for (int s = 0; s < 6; ++s) s_new[s] = cur_s(b,s);
            }
        }
        for (int s = 0; s < 6; ++s) strain(b,s) = s_new[s];
    }
};

// Boltzmann test and move budget. The uniform is drawn on every move so the
// stream position never depends on the energies.
template <typename Space>
struct McAccept {
    SView3D<RealType, Space> cur_t, cur_r, trans, rots, cellv, best_t, best_r, best_cell;
    SView2D<RealType, Space> cur_s, strain;
    SView1D<RealType, Space> E_cur, E_best, gap;
    SView1D<int, Space> mv, nacc;
    Kokkos::Random_XorShift64_Pool<Space> rng;
    RealType kT, min_gap;
    int ni, n_moves;

    KOKKOS_INLINE_FUNCTION bool operator()(const int b, const RealType E) const {
        const bool finite = (E == E) && E < RealType(1e300) && E > RealType(-1e300);
        if (mv(b) == 0) {
            // Adopt the starting geometry. A non-finite energy becomes infinity
            // (not 1e300) so pack_results drops the structure.
            const RealType E0 = finite ? E : Kokkos::Experimental::infinity_v<RealType>;
            E_cur(b) = E0; E_best(b) = E0;
            for (int k = 0; k < ni; ++k)
                for (int d = 0; d < 3; ++d) {
                    cur_t(b,k,d) = trans(b,k,d); best_t(b,k,d) = trans(b,k,d);
                    cur_r(b,k,d) = rots(b,k,d);  best_r(b,k,d) = rots(b,k,d);
                }
            for (int s = 0; s < 6; ++s) cur_s(b,s) = strain(b,s);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) best_cell(b,r,c) = cellv(b,r,c);
            mv(b) = 1;
            return 1 >= n_moves;
        }

        auto gen = rng.get_state(b);
        const RealType u = RealType(gen.drand());
        rng.free_state(gen);

        // A proposal inside the hard core is rejected like any out-of-support move.
        const bool clash = (min_gap > RealType(0)) && (gap(b) < min_gap);
        const RealType dE = E - E_cur(b);
        if (finite && !clash && mc_accepts(dE, kT, u)) {
            for (int k = 0; k < ni; ++k)
                for (int d = 0; d < 3; ++d) {
                    cur_t(b,k,d) = trans(b,k,d);
                    cur_r(b,k,d) = rots(b,k,d);
                }
            for (int s = 0; s < 6; ++s) cur_s(b,s) = strain(b,s);
            E_cur(b) = E;
            nacc(b) += 1;
            // Harvest the best state seen, not the last.
            if (E < E_best(b)) {
                E_best(b) = E;
                for (int k = 0; k < ni; ++k)
                    for (int d = 0; d < 3; ++d) {
                        best_t(b,k,d) = trans(b,k,d);
                        best_r(b,k,d) = rots(b,k,d);
                    }
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) best_cell(b,r,c) = cellv(b,r,c);
            }
        }
        // No rollback on reject: propose() rebuilds the trial from cur_* each round.
        mv(b) += 1;
        return mv(b) >= n_moves;
    }
};

// The best state the chain saw, into the host staging buffers.
template <typename Space>
struct McHarvest {
    SView1D<RealType, Space> d_hE;
    SView3D<RealType, Space> d_ht, d_hr, d_hcell;
    SView1D<int, Space> d_hseed, d_hacc;
    SView1D<RealType, Space> E_best;
    SView3D<RealType, Space> best_t, best_r, best_cell;
    SView1D<int, Space> slot_entry, p_seeds, nacc;
    int ni;

    KOKKOS_INLINE_FUNCTION void operator()(const int b, const int o) const {
        d_hE(o) = E_best(b);
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                d_ht(o,k,d) = best_t(b,k,d);
                d_hr(o,k,d) = best_r(b,k,d);
            }
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) d_hcell(o,r,c) = best_cell(b,r,c);
        d_hseed(o) = p_seeds(slot_entry(b));
        d_hacc(o) = nacc(b);
    }
};

// Shortest intermolecular contact per slot (minimum image), for the hard core.
// The force field is unbounded below at short range, so a chain can otherwise
// walk into the singularity. Intramolecular pairs and ghost_type padding sites
// are excluded.
template <typename Space>
struct McMinContact {
    SView3D<RealType, Space> pos, cellv, cinv;
    SView2D<int, Space> mol_id, types;
    SView1D<RealType, Space> gap;
    SView1D<int, Space> state;
    int n_atoms, ghost_type;

    KOKKOS_INLINE_FUNCTION void operator()(const int b) const {
        if (state(b) != SLOT_LIVE) return;
        RealType best = Kokkos::Experimental::infinity_v<RealType>;
        for (int i = 0; i < n_atoms; ++i) {
            if (types(b,i) == ghost_type) continue;
            for (int j = i + 1; j < n_atoms; ++j) {
                if (types(b,j) == ghost_type) continue;
                if (mol_id(b,i) == mol_id(b,j)) continue;
                RealType d[3];
                for (int k = 0; k < 3; ++k) d[k] = pos(b,i,k) - pos(b,j,k);
                // Minimum image, through the fractional difference.
                RealType f[3];
                for (int k = 0; k < 3; ++k)
                    f[k] = cinv(b,k,0)*d[0] + cinv(b,k,1)*d[1] + cinv(b,k,2)*d[2];
                for (int k = 0; k < 3; ++k) f[k] -= Kokkos::round(f[k]);
                for (int k = 0; k < 3; ++k)
                    d[k] = cellv(b,k,0)*f[0] + cellv(b,k,1)*f[1] + cellv(b,k,2)*f[2];
                const RealType r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
                if (r2 < best) best = r2;
            }
        }
        gap(b) = Kokkos::sqrt(best);
    }
};

// Give every slot a valid placeholder cell before anything runs over the whole
// capacity. Too small an edge makes the lattice-shift box explode.
template <typename Space>
struct McSafeCell {
    SView3D<RealType, Space> cellv, cell_ref;
    SView2D<RealType, Space> strain;
    SView1D<int, Space> mv;
    RealType SAFE_CELL;
    KOKKOS_INLINE_FUNCTION void operator()(const int b) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                cellv(b,i,j)    = (i==j) ? SAFE_CELL : RealType(0);
                cell_ref(b,i,j) = (i==j) ? SAFE_CELL : RealType(0);
            }
        for (int s = 0; s < 6; ++s) strain(b,s) = 0;
        mv(b) = 0;
    }
};

// Host side.

template <int MaxRank, typename Space>
struct McStepper {
    using V1 = SView1D<RealType, Space>;
    using V2 = SView2D<RealType, Space>;
    using V3 = SView3D<RealType, Space>;
    using Vi = SView1D<int, Space>;

    BatchWorkspaceT<Space>& bws;
    StreamPoolT<Space>& pool;
    BatchOptimizeSettings es;
    McSettings mc;
    ForceField ff;
    KernelTimings kt;

    int cap, ni, ncd, lattice;

    // Per-slot chain state: accepted point, best point seen, counters.
    V3 cur_t, cur_r, best_t, best_r, best_cell;
    V2 cur_s;
    V1 E_cur, E_best, gap;
    Vi mv, nacc, slot_entry;
    Kokkos::Random_XorShift64_Pool<Space> rng;

    // Harvest staging (device), copied to host once at the end.
    int Pn = 0;
    V1 d_hE;
    V3 d_ht, d_hr, d_hcell;
    Vi d_hseed, d_hacc;

    Vi state;   // the engine's slot lifecycle, also the energy mask

    McStepper(BatchWorkspaceT<Space>& b, StreamPoolT<Space>& p,
              const BatchOptimizeSettings& settings, const McSettings& m)
        : bws(b), pool(p), es(settings), mc(m),
          ff(make_force_field(b.tables, settings.model)),
          // Seed is irrelevant: every slot's state is overwritten at fill time.
          rng(1, static_cast<uint64_t>(Kokkos::max(1, b.geom.n_batch))) {
        es.energy_only = false;
        cap = bws.geom.n_batch;
        ni  = bws.tables.n_indep;
        ncd = bws.dofs.n_cell_dof;
        lattice = bws.dofs.cell_dof_lattice;

        cur_t = V3("mc_cur_t", cap, ni, 3); cur_r = V3("mc_cur_r", cap, ni, 3);
        best_t = V3("mc_best_t", cap, ni, 3); best_r = V3("mc_best_r", cap, ni, 3);
        best_cell = V3("mc_best_cell", cap, 3, 3);
        cur_s = V2("mc_cur_s", cap, 6);
        E_cur = V1("mc_E", cap); E_best = V1("mc_Eb", cap);
        gap = V1("mc_gap", cap);
        mv = Vi("mc_mv", cap); nacc = Vi("mc_acc", cap);
        slot_entry = Vi("mc_entry", cap);

        Pn = pool.filled;
        d_hE = V1("mc_hE", Pn > 0 ? Pn : 1);
        d_ht = V3("mc_ht", Pn > 0 ? Pn : 1, ni, 3);
        d_hr = V3("mc_hr", Pn > 0 ? Pn : 1, ni, 3);
        d_hcell = V3("mc_hcell", Pn > 0 ? Pn : 1, 3, 3);
        d_hseed = Vi("mc_hseed", Pn > 0 ? Pn : 1);
        d_hacc = Vi("mc_hacc", Pn > 0 ? Pn : 1);
    }

    ~McStepper() { bws.geom.slot_mask = decltype(bws.geom.slot_mask)(); }

    McStepper(const McStepper&) = delete;
    McStepper& operator=(const McStepper&) = delete;


    void bind(Vi slot_state) {
        state = slot_state;
        Kokkos::parallel_for("mc_cell_safe", Kokkos::RangePolicy<Space>(0, cap),
            McSafeCell<Space>{bws.geom.cell, bws.dofs.cell_ref, bws.dofs.strain,
                              mv, RealType(20)});
        bws.geom.slot_mask = state;
    }

    V1 energies() const { return bws.geom.energies; }

    void evaluate(int round) {
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);
        batch_expand_asymmetric_unit<MaxRank, Space>(bws.tables, bws.geom, bws.dofs);
        batch_energy_gradient_asymunit<MaxRank>(bws.tables, bws.geom, bws.dofs, es, round, &kt);
        // Contact screen over the expanded atoms the pair pass just used.
        if (mc.min_contact > 0)
            Kokkos::parallel_for("mc_contact", Kokkos::RangePolicy<Space>(0, cap),
                McMinContact<Space>{bws.geom.positions, bws.geom.cell, bws.geom.cell_inv,
                                    bws.geom.mol_id, bws.geom.types, gap, state,
                                    bws.geom.n_atoms, bws.tables.ghost_type});
    }

    // No curvature is carried between moves, so neither hook is needed.
    void on_filled(int) {}
    void after_round() {}

    void harvest(Vi idx, int n, int base) {
        stream::for_slot_pairs<Space>("mc_harvest", idx, n, base,
            McHarvest<Space>{d_hE, d_ht, d_hr, d_hcell, d_hseed, d_hacc,
                             E_best, best_t, best_r, best_cell,
                             slot_entry, pool.seeds, nacc, ni});
    }


    McFill<Space> filler() const {
        return McFill<Space>{
            bws.geom.cell, bws.dofs.cell_ref, bws.geom.cell_inv,
            bws.geom.kappa, bws.geom.volume, bws.geom.max_hkl, bws.geom.max_shift,
            bws.geom.cell_params, bws.dofs.strain,
            bws.dofs.translations, bws.dofs.rot_vectors,
            cur_t, cur_r, best_t, best_r, best_cell,
            cur_s, bws.dofs.cell_dof_basis,
            E_cur, E_best, mv, nacc, slot_entry,
            pool.cells.cell, pool.cells.cell_inv, pool.translations, pool.rot_vectors,
            pool.cells.kappa, pool.cells.volume,
            pool.cells.max_hkl, pool.cells.max_shift, pool.cells.cell_params,
            pool.seeds, rng, ni, ncd, lattice};
    }

    McPropose<Space> proposer() const {
        return McPropose<Space>{
            cur_t, cur_r, bws.dofs.translations, bws.dofs.rot_vectors,
            bws.dofs.cell_ref, bws.dofs.cell_dof_basis,
            cur_s, bws.dofs.strain, mv, rng, mc.prior,
            mc.step_trans, mc.step_rot, mc.step_strain, ni, ncd};
    }

    McAccept<Space> accepter() const {
        return McAccept<Space>{
            cur_t, cur_r, bws.dofs.translations, bws.dofs.rot_vectors,
            bws.geom.cell, best_t, best_r, best_cell,
            cur_s, bws.dofs.strain, E_cur, E_best, gap, mv, nacc, rng,
            mc.kT, mc.min_contact * RealType(ANGSTROM_TO_BOHR), ni, mc.n_moves};
    }
};

template <int MaxRank, typename Space>
StreamResult batch_optimize_stream_mc(BatchWorkspaceT<Space>& bws,
                                      StreamPoolT<Space>& pool,
                                      const BatchOptimizeSettings& settings,
                                      const McSettings& mc) {
    Kokkos::Timer timer;
    if (mc.n_moves < 1)
        Kokkos::abort("batch_optimize_stream_mc: the move budget is what the "
                      "stepper finishes on; it has to be at least one");
    if (bws.tables.n_indep > MAX_INDEP)
        Kokkos::abort("batch_optimize_stream_mc: n_indep exceeds MAX_INDEP");

    McStepper<MaxRank, Space> st(bws, pool, settings, mc);

    stream::Config cfg;
    cfg.capacity = st.cap;
    cfg.refill_interval = settings.refill_interval > 0 ? settings.refill_interval : 10;
    cfg.source_base = pool.consumed;
    cfg.source_count = pool.filled - pool.consumed;
    cfg.harvest_capacity = st.Pn;
    cfg.abort = settings.hooks.abort;

    const stream::Result r = stream::schedule<Space>(cfg, st);
    pool.consumed += r.n_consumed;
    const int n = r.n_harvested;
    const int ni = st.ni;

    StreamResult out;
    out.n = n;
    out.energies     = Kokkos::View<RealType*, Kokkos::HostSpace>("o_E", n);
    out.translations = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>("o_t", n, ni, 3);
    out.rotations    = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>("o_r", n, ni, 3);
    out.cell         = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>("o_cell", n, 3, 3);
    using P2 = std::pair<int,int>;
    P2 hr(0, n);
    Kokkos::deep_copy(out.energies,     Kokkos::subview(st.d_hE, hr));
    Kokkos::deep_copy(out.translations, Kokkos::subview(st.d_ht, hr, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(out.rotations,    Kokkos::subview(st.d_hr, hr, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(out.cell,         Kokkos::subview(st.d_hcell, hr, Kokkos::ALL, Kokkos::ALL));
    {
        auto h_seed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          Kokkos::subview(st.d_hseed, hr));
        out.seeds.resize(n);
        for (int j = 0; j < n; ++j) out.seeds[j] = h_seed(j);
    }
    // Acceptance ratio, counted per structure over its own budget.
    if (n > 0 && mc.n_moves > 1) {
        auto h_acc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         Kokkos::subview(st.d_hacc, hr));
        long long acc = 0;
        for (int j = 0; j < n; ++j) acc += h_acc(j);
        logging::detail("    mc: {} chains x {} moves, acceptance {:.2f}\n",
                        n, mc.n_moves, double(acc) / (double(n) * (mc.n_moves - 1)));
    }

    // No diagnostics: the gradient-norm and convergence columns do not apply.
    out.wall_time_s = timer.seconds();
    out.timings = st.kt;
    out.total_iters = r.total_iters;
    return out;
}

template StreamResult batch_optimize_stream_mc<0, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
template StreamResult batch_optimize_stream_mc<1, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
template StreamResult batch_optimize_stream_mc<2, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
template StreamResult batch_optimize_stream_mc<3, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
template StreamResult batch_optimize_stream_mc<4, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);

}  // namespace klasp
