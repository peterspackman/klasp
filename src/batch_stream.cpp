// The trust-region stepper for the streaming engine. See batch_stream.hpp for
// the public entry point and klasp/stream_engine.hpp for the scheduler.
// A refilled slot's first iteration is an init eval (pstep = 0) that populates
// its accepted gradient and energy.

#include "klasp/batch_stream.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/stream_engine.hpp"
#include "klasp/trust_region.hpp"
#include "klasp/strain_basis.hpp"

#include <Kokkos_Core.hpp>
#include <vector>

namespace klasp {

template <typename Space>
StreamPoolT<Space> allocate_stream_pool(int P, int n_indep) {
    StreamPoolT<Space> pool;
    pool.P = P;
    pool.n_indep = n_indep;
    pool.nmd = n_indep * 6;
    const int ni = n_indep, nmd = pool.nmd;
    pool.cells        = allocate_cell_state<Space>(P, "pool");
    pool.translations = SView3D<RealType, Space>("pool_tr", P, ni, 3);
    pool.rot_vectors  = SView3D<RealType, Space>("pool_rv", P, ni, 3);
    pool.hess_mol     = SView3D<RealType, Space>("pool_H", P, nmd, nmd);
    pool.s1_grad      = SView1D<RealType, Space>("pool_s1g", P);
    pool.s1_iters     = SView1D<int, Space>("pool_s1i", P);
    pool.seeds        = SView1D<int, Space>("pool_seeds", P);
    return pool;
}

template <typename Space>
void stream_pool_append(StreamPoolT<Space>& pool, const BatchWorkspaceT<Space>& bws,
                        int count, const std::vector<int>& seeds_host,
                        const BatchDiagnostics& s1_diag, bool with_hessian) {
    if (count <= 0) return;
    const int off = pool.filled;
    if (off + count > pool.P) count = pool.P - off;   // never overrun
    if (count <= 0) return;
    const int nmd = pool.nmd;
    using P2 = std::pair<int,int>;
    P2 d(off, off + count), s(0, count);
    copy_cell_state<Space>(pool.cells, off, bws.geom, 0, count);
    Kokkos::deep_copy(Kokkos::subview(pool.translations, d, Kokkos::ALL, Kokkos::ALL),
                      Kokkos::subview(bws.dofs.translations,   s, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(pool.rot_vectors, d, Kokkos::ALL, Kokkos::ALL),
                      Kokkos::subview(bws.dofs.rot_vectors,   s, Kokkos::ALL, Kokkos::ALL));
    // Molecular Hessian block from the stage-1 trust-region run.
    if (with_hessian
        && (int)bws.tr.tr_hessian_mol.extent(0) >= count
        && (int)bws.tr.tr_hessian_mol.extent(1) == nmd) {
        Kokkos::deep_copy(Kokkos::subview(pool.hess_mol, d, Kokkos::ALL, Kokkos::ALL),
                          Kokkos::subview(bws.tr.tr_hessian_mol, s, Kokkos::ALL, Kokkos::ALL));
    } else {
        // No curvature known: a zero block, which the loader turns into an
        // identity in its own scaled coordinates.
        Kokkos::deep_copy(Kokkos::subview(pool.hess_mol, d, Kokkos::ALL, Kokkos::ALL), RealType(0));
    }
    // Seeds + stage-1 convergence (host -> device tail).
    auto h_seeds = Kokkos::create_mirror_view(pool.seeds);
    auto h_s1g = Kokkos::create_mirror_view(pool.s1_grad);
    auto h_s1i = Kokkos::create_mirror_view(pool.s1_iters);
    Kokkos::deep_copy(h_seeds, pool.seeds);   // keep earlier entries
    Kokkos::deep_copy(h_s1g, pool.s1_grad);
    Kokkos::deep_copy(h_s1i, pool.s1_iters);
    for (int i = 0; i < count; ++i) {
        h_seeds(off + i) = seeds_host[i];
        h_s1g(off + i) = s1_diag.valid() ? s1_diag.grad_max(i) : RealType(-1);
        h_s1i(off + i) = s1_diag.valid() ? s1_diag.iters_to_converge(i) : -1;
    }
    Kokkos::deep_copy(pool.seeds, h_seeds);
    Kokkos::deep_copy(pool.s1_grad, h_s1g);
    Kokkos::deep_copy(pool.s1_iters, h_s1i);
    pool.filled = off + count;
}

// ===========================================================================
// The trust-region stepper: device functors and a host object owning their state.
// Separate functors keep each kernel's capture set small. They are at namespace
// scope because CUDA extended lambda closure types need external linkage.
// ===========================================================================

// --- fill: gather pool entry `src` into slot `b` --------------------------
template <typename Space>
struct TrFill {
    SView3D<RealType, Space> cellv, cell_ref, b_cinv;
    SView1D<RealType, Space> b_kappa, b_vol;
    SView2D<int, Space> b_hkl, b_shift;
    SView2D<RealType, Space> b_cp, strain;
    SView3D<RealType, Space> trans, rots, acc_t, acc_r;
    SView2D<RealType, Space> acc_s;
    SView3D<RealType, Space> basis;
    SView2D<RealType, Space> Dsc;
    SView2D<RealType, Space> body;
    SView1D<int, Space> b_nsites;
    tr::StepViews<Space> sv;
    SView1D<RealType, Space> delta, pred;
    SView1D<int, Space> s_restarts, slot_iter, s_conv_iter, s_wrapped, slot_entry;
    SView3D<RealType, Space> p_cell, p_cinv, p_tr, p_rv, p_H;
    SView1D<RealType, Space> p_kappa, p_vol;
    SView2D<int, Space> p_hkl, p_shift;
    SView2D<RealType, Space> p_cp;
    int ni, ns, nmd, ncd, ndof, lattice;
    bool warm_mol;
    RealType DELTA0;
    // Starting precision of the multipole term (1 = single when the hybrid is
    // on, else 0), and the re-evaluation latch; see TrAccept.
    SView1D<int, Space> s_prec, s_reeval;
    int start_prec;

    KOKKOS_INLINE_FUNCTION void operator()(const int b, const int src) const {
        // cell + Ewald state, copied verbatim
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                cellv(b,r,c)    = p_cell(src,r,c);
                cell_ref(b,r,c) = p_cell(src,r,c);
                b_cinv(b,r,c)   = p_cinv(src,r,c);
            }
        b_kappa(b) = p_kappa(src);
        b_vol(b)   = p_vol(src);
        for (int d = 0; d < 3; ++d) { b_hkl(b,d) = p_hkl(src,d); b_shift(b,d) = p_shift(src,d); }
        for (int d = 0; d < 6; ++d) { b_cp(b,d) = p_cp(src,d); strain(b,d) = 0; }
        // DOFs + accepted caches (start = stage-1-relaxed geometry)
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                trans(b,k,d) = p_tr(src,k,d); acc_t(b,k,d) = p_tr(src,k,d);
                rots(b,k,d)  = p_rv(src,k,d); acc_r(b,k,d) = p_rv(src,k,d);
            }
        for (int s = 0; s < 6; ++s) acc_s(b,s) = 0;
        // The strain basis belongs to the structure, not the slot (a rhombohedral
        // basis follows its own 3-fold axis), so rebuild it on every fill.
        if (ncd > 0) {
            RealType lc[9], lb[36];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) lc[r*3 + c] = cellv(b,r,c);
            build_strain_basis(lattice, lc, lb, ncd);
            for (int s = 0; s < 6; ++s)
                for (int v = 0; v < 6; ++v) basis(b,s,v) = lb[s*6 + v];
        }
        // Per-DOF scaling D (rotation ~ R_gyr; strain ~ longest cell edge).
        tr::compute_dof_scaling(b, ni, ns, nmd, ncd, cellv, body, Dsc, b_nsites);
        // Molecular block warm from the pool when enabled; everything else is
        // seeded by batch_seed_hessian_blocks after expansion, so zeroed here.
        auto B0 = sv.B0;
        for (int i = 0; i < ndof; ++i) B0(b,i) = 0;
        if (warm_mol)
            for (int i = 0; i < nmd; ++i) B0(b,i) = p_H(src,i,i);
        // The pool holds unscaled curvature: B0 = D B D in this slot's scaled
        // coordinates. A zero block means no curvature known; seed identity.
        {
            RealType tr_diag = 0;
            for (int i = 0; i < nmd; ++i) tr_diag += Kokkos::fabs(B0(b,i));
            for (int i = 0; i < nmd; ++i)
                B0(b,i) = tr_diag > RealType(0) ? B0(b,i) * Dsc(b,i) * Dsc(b,i) : RealType(1);
        }
        tr::model_reset(sv, b);
        delta(b) = DELTA0; pred(b) = 0; s_restarts(b) = 0;
        slot_iter(b) = 0; slot_entry(b) = src;
        s_conv_iter(b) = -1; s_wrapped(b) = 0;
        s_prec(b) = start_prec; s_reeval(b) = 0;
    }
};

// --- propose: move slot b to the point the oracle is asked about ----------
template <typename Space>
struct TrPropose {
    tr::StepViews<Space> sv;
    SView2D<RealType, Space> gacc, pstep;
    SView1D<RealType, Space> delta, pred;
    SView1D<int, Space> slot_iter, s_wrapped;
    int ndof;
    SView1D<int, Space> s_reeval;

    KOKKOS_INLINE_FUNCTION void operator()(const int b) const {
        if (slot_iter(b) == 0 || s_reeval(b)) {
            // Init eval at the unchanged starting geometry (pstep = 0); also the
            // re-evaluation after a switch to double precision.
            for (int i = 0; i < ndof; ++i) pstep(b,i) = 0;
            tr::rollback_step(sv, b);
            pred(b) = 0; s_wrapped(b) = 0;
            return;
        }
        RealType pr = tr::model_dogleg(sv, b);
        const RealType t = tr::clip_cell_step(sv, b);
        if (t < RealType(1)) pr = tr::model_scale_step(sv, b, t);
        pred(b) = pr;
        s_wrapped(b) = tr::apply_step(sv, b) ? 1 : 0;
    }
};

// --- accept: judge the evaluation, and say whether the slot is finished ---
//
// Returns true when the slot is finished: converged, stalled or out of iterations.
template <typename Space>
struct TrAccept {
    tr::StepViews<Space> sv;
    SView3D<RealType, Space> dgt, dgr, basis;
    SView2D<RealType, Space> dgs, Dsc, gacc;
    SView1D<RealType, Space> Eacc, s_gmax, s_grms, delta;
    SView1D<int, Space> slot_iter, s_conv_iter, s_restarts;
    tr::TRParams trp;
    RealType gt, DELTA_STALL;
    int ni, ncd, nmd, ndof, mi;
    // Single/double hybrid for the multipole term: relax in single precision
    // until the scaled gradient is under switch_tol (or the slot looks converged
    // or stalled), then switch to double for good and re-evaluate. Only double
    // precision can certify convergence.
    SView1D<int, Space> s_prec, s_reeval;
    RealType switch_tol, DELTA0;
    bool hybrid;

    KOKKOS_INLINE_FUNCTION bool operator()(const int b, const RealType Etr) const {
        const bool finite = (Etr == Etr) && Etr < RealType(1e300) && Etr > RealType(-1e300);
        if (slot_iter(b) == 0 || s_reeval(b)) {
            // Init, or the re-evaluation after a precision switch: adopt the
            // current point as accepted. Hessian and radius carry over.
            s_reeval(b) = 0;
            RealType* g = &sv.mg(b,0);
            tr::pack_scaled_gradient(b, ni, ncd, nmd, dgt, dgr, dgs, basis, Dsc, g);
            for (int i = 0; i < ndof; ++i) gacc(b,i) = g[i];
            // infinity(), not a large finite value: pack_results drops
            // non-finite energies, so a slot that never accepted is not stored.
            Eacc(b) = finite ? Etr : Kokkos::Experimental::infinity_v<RealType>;
            tr::gradient_norms(g, ndof, s_gmax(b), s_grms(b));
        } else {
            tr::advance_step(sv, b, Etr, trp,
                             (hybrid && s_prec(b) == 1) ? RealType(1.2e-7) : trp.precision);
        }
        const bool single = hybrid && s_prec(b) == 1;
        if (!single && s_conv_iter(b) < 0 && gt > 0 && s_gmax(b) < gt)
            s_conv_iter(b) = slot_iter(b);
        slot_iter(b) += 1;
        // Finished: converged (gmax < tol), stalled (radius still collapsed
        // after advance_step's restart budget), or at the iteration cap.
        const bool converged = (gt > 0 && s_gmax(b) < gt);
        const bool stalled = delta(b) < DELTA_STALL
                          && s_restarts(b) >= trp.max_restarts;
        if (single) {
            // Nothing is final in single precision: switch to double and continue
            // from the same point with the radius and restart budget restored.
            if (converged || stalled || s_gmax(b) < switch_tol) {
                s_prec(b) = 0;
                s_reeval(b) = 1;
                if (stalled) {
                    delta(b) = Kokkos::fmax(delta(b), RealType(0.1) * DELTA0);
                    s_restarts(b) = 0;
                }
            }
            return slot_iter(b) >= mi;
        }
        return converged || stalled || slot_iter(b) >= mi;
    }
};

// --- harvest, device-to-device into a downstream pool ---------------------
template <typename Space>
struct TrHarvestPool {
    SView3D<RealType, Space> o_cell, o_cinv, o_tr, o_rv, o_H;
    SView1D<RealType, Space> o_kappa, o_vol, o_s1g;
    SView2D<int, Space> o_hkl, o_shift;
    SView2D<RealType, Space> o_cp;
    SView1D<int, Space> o_s1i, o_seeds;
    SView3D<RealType, Space> cellv, b_cinv, acc_t, acc_r;
    tr::StepViews<Space> sv;
    SView1D<RealType, Space> b_kappa, b_vol, s_gmax;
    SView2D<int, Space> b_hkl, b_shift;
    SView2D<RealType, Space> b_cp;
    SView1D<int, Space> s_conv_iter, slot_entry, p_seeds;
    int ni, nmd;

    KOKKOS_INLINE_FUNCTION void operator()(const int b, const int o) const {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                o_cell(o,r,c) = cellv(b,r,c);
                o_cinv(o,r,c) = b_cinv(b,r,c);
            }
        o_kappa(o) = b_kappa(b);
        o_vol(o)   = b_vol(b);
        for (int d = 0; d < 3; ++d) { o_hkl(o,d) = b_hkl(b,d); o_shift(o,d) = b_shift(b,d); }
        for (int d = 0; d < 6; ++d) o_cp(o,d) = b_cp(b,d);
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                o_tr(o,k,d) = acc_t(b,k,d);
                o_rv(o,k,d) = acc_r(b,k,d);
            }
        // The model's diagonal, stored unscaled, warm-starts the next stage.
        tr::model_unscaled_diagonal(sv, b);
        for (int i = 0; i < nmd; ++i)
            for (int jj = 0; jj < nmd; ++jj) o_H(o,i,jj) = (i == jj) ? sv.mnw(b,i) : RealType(0);
        o_seeds(o) = p_seeds(slot_entry(b));
        o_s1g(o) = s_gmax(b);
        o_s1i(o) = s_conv_iter(b);
    }
};

// --- harvest, into the device-side staging buffers that go home to the host --
template <typename Space>
struct TrHarvestHost {
    SView1D<RealType, Space> d_hE;
    SView3D<RealType, Space> d_ht, d_hr, d_hcell;
    SView1D<int, Space> d_hseed;
    SView1D<RealType, Space> d_hgmax, d_hgrms, d_hdelta, d_hs1g;
    SView1D<int, Space> d_hiters, d_hrestarts, d_hs1i;
    SView1D<RealType, Space> Eacc, s_gmax, s_grms, delta, p_s1g;
    SView3D<RealType, Space> acc_t, acc_r, cellv;
    SView1D<int, Space> s_conv_iter, s_restarts, slot_entry, p_seeds, p_s1i;
    int ni;
    bool diag;

    KOKKOS_INLINE_FUNCTION void operator()(const int b, const int o) const {
        d_hE(o) = Eacc(b);
        for (int k = 0; k < ni; ++k)
            for (int d = 0; d < 3; ++d) {
                d_ht(o,k,d) = acc_t(b,k,d);
                d_hr(o,k,d) = acc_r(b,k,d);
            }
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) d_hcell(o,r,c) = cellv(b,r,c);
        d_hseed(o) = p_seeds(slot_entry(b));
        if (diag) { d_hgmax(o) = s_gmax(b); d_hgrms(o) = s_grms(b); d_hiters(o) = s_conv_iter(b);
                    d_hdelta(o) = delta(b); d_hrestarts(o) = s_restarts(b);
                    d_hs1g(o) = p_s1g(slot_entry(b)); d_hs1i(o) = p_s1i(slot_entry(b)); }
    }
};

// --- two small setup/bookkeeping kernels, as functors so no extended lambda is
//     defined inside a member function ----------------------------------------
template <typename Space>
struct TrSafeCell {
    SView3D<RealType, Space> cellv, cell_ref;
    SView2D<RealType, Space> strain;
    SView1D<int, Space> slot_iter;
    RealType SAFE_CELL;
    KOKKOS_INLINE_FUNCTION void operator()(const int b) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                cellv(b,i,j) = (i==j) ? SAFE_CELL : RealType(0);
                cell_ref(b,i,j) = (i==j) ? SAFE_CELL : RealType(0);
            }
        for (int s = 0; s < 6; ++s) strain(b,s) = 0;
        slot_iter(b) = 0;
    }
};

template <typename Space>
struct TrMarkNew {
    SView1D<int, Space> d_hmask, state, slot_iter;
    KOKKOS_INLINE_FUNCTION void operator()(const int b) const {
        d_hmask(b) = (state(b) == SLOT_LIVE && slot_iter(b) == 1) ? 1 : 0;
    }
};

// ===========================================================================
// The host half: allocation, the energy oracle, and the two hooks the engine
// calls between launches.
// ===========================================================================

template <int MaxRank, typename Space>
struct TrStepper {
    using V1 = SView1D<RealType, Space>;
    using V2 = SView2D<RealType, Space>;
    using V3 = SView3D<RealType, Space>;
    using Vi = SView1D<int, Space>;

    BatchWorkspaceT<Space>& bws;
    StreamPoolT<Space>& pool;
    StreamPoolT<Space>* out_pool;
    // es is the copy the energy pipeline runs with (energy_only forced off);
    // settings0 is the caller's, handed to the Hessian seeder.
    BatchOptimizeSettings settings0, es;
    ForceField ff;
    KernelTimings kt;

    int cap, ni, ns, ncd, nmd, ndof, lattice, max_iters;
    bool to_pool, warm_mol, diag;
    RealType DELTA0, DELTA_STALL, grad_tol;
    tr::TRParams trp;

    V3 acc_t, acc_r;
    V2 gacc, pstep, Dsc, acc_s;
    V1 Eacc, delta, pred, s_gmax, s_grms;
    Vi slot_iter, slot_entry, s_conv_iter, s_restarts, s_wrapped;
    // Multipole-term precision hybrid: per-slot precision, re-evaluation latch,
    // switch level. See TrAccept.
    Vi s_prec, s_reeval;
    // Which slots batch_rebracket_ewald just moved to a new Ewald split, and
    // the cell volume past which a slot is abandoned. vol_abort <= 0 disables.
    Vi s_rebracket;
    RealType vol_abort = 0;
    RealType switch_tol = 0;
    bool hybrid = false;
    tr::StepViews<Space> sv;

    // Harvest staging (device), copied to the host StreamResult once at the end.
    int Pn = 0;
    V1 d_hE, d_hgmax, d_hgrms, d_hdelta, d_hs1g;
    V3 d_ht, d_hr, d_hcell;
    Vi d_hseed, d_hiters, d_hrestarts, d_hs1i;
    int out_base0 = 0;

    // The engine's slot lifecycle, used for the energy mask and Hessian seeding.
    Vi state;
    // Host-known (refills happen only at checkpoints), avoiding a device
    // reduction and its sync every iteration.
    bool pending_seed = false;

    TrStepper(BatchWorkspaceT<Space>& b, StreamPoolT<Space>& p, StreamPoolT<Space>* op,
              const BatchOptimizeSettings& settings)
        : bws(b), pool(p), out_pool(op),
          settings0(settings), es(settings),
          ff(make_force_field(b.tables, settings.model)) {
        es.energy_only = false;
        to_pool = out_pool != nullptr;
        cap = bws.geom.n_batch;               // full slot capacity
        ni  = bws.tables.n_indep;
        ns  = bws.tables.n_sites_per_mol;
        ncd = bws.dofs.n_cell_dof;
        lattice = bws.dofs.cell_dof_lattice;
        nmd = ni * 6;
        ndof = nmd + ncd;
        max_iters = settings.max_iters_per_slot > 0
                  ? settings.max_iters_per_slot : settings.n_iterations;
        DELTA0 = trp.delta0;
        // Stall harvest: trust radius (Bohr) below which the slot is at its local
        // minimum to achievable precision and is harvested.
        DELTA_STALL = RealType(5e-3);
        grad_tol = settings.grad_tol;
        diag = settings.store_diagnostics;
        warm_mol = settings.reuse_hessian;

        tr::allocate_model(sv, cap, ndof);
        gacc = sv.gacc; pstep = sv.pstep;
        Dsc = V2("st_Dsc", cap, ndof);
        Eacc = V1("st_Eacc", cap); delta = V1("st_delta", cap); pred = V1("st_pred", cap);
        acc_t = V3("st_acc_t", cap, ni, 3); acc_r = V3("st_acc_r", cap, ni, 3);
        acc_s = V2("st_acc_s", cap, 6);
        s_gmax = V1("st_gmax", cap); s_grms = V1("st_grms", cap);
        slot_iter = Vi("st_iter", cap); slot_entry = Vi("st_entry", cap);
        s_conv_iter = Vi("st_conv", cap);
        s_restarts = Vi("st_restarts", cap);
        s_wrapped = Vi("st_wrapped", cap);
        s_prec = Vi("st_prec", cap); s_reeval = Vi("st_reeval", cap);
        s_rebracket = Vi("st_rebracket", cap);
        hybrid = es.model.mp_single && MaxRank >= 2;
        switch_tol = settings.mp_single_switch_tol;

        sv.Dsc = Dsc;
        sv.acc_t = acc_t; sv.acc_r = acc_r; sv.acc_s = acc_s;
        sv.trans = bws.dofs.translations; sv.rots = bws.dofs.rot_vectors;
        sv.strain = bws.dofs.strain;
        sv.cell = bws.geom.cell; sv.cell_ref = bws.dofs.cell_ref;
        sv.basis = bws.dofs.cell_dof_basis;
        sv.dgt = bws.dofs.d_gt; sv.dgr = bws.dofs.d_gr; sv.dgs = bws.dofs.d_g_strain;
        sv.Eacc = Eacc; sv.delta = delta; sv.pred = pred;
        sv.gmax = s_gmax; sv.grms = s_grms;
        sv.restarts = s_restarts; sv.wrapped = s_wrapped;
        // Allocate seeding scratch before binding sv (a View is a handle), sized by
        // the workspace's allocated capacity as batch_seed_hessian_blocks expects,
        // so the seeder never reallocates underneath sv.
        const int hess_cap = static_cast<int>(bws.geom.energies.extent(0));
        if ((int)bws.tr.fc.extent(0) != hess_cap || (int)bws.tr.fc.extent(1) != bws.geom.n_atoms
            || (int)bws.tr.fc.extent(2) != 12)
            bws.tr.fc = decltype(bws.tr.fc)("tr_fc", hess_cap, bws.geom.n_atoms, 12);
        if ((int)bws.tr.strain_hess.extent(0) != hess_cap)
            bws.tr.strain_hess = decltype(bws.tr.strain_hess)("tr_strain_hess", hess_cap, 6, 6);
        sv.strain_hess = bws.tr.strain_hess;
        sv.ni = ni; sv.ncd = ncd; sv.nmd = nmd; sv.ndof = ndof;

        // Seed the Hessian of every slot at its init iteration; bws.tr.hess_mask
        // restricts the work to those slots.
        if (bws.tr.hess_mask.extent(0) != (size_t)cap)
            bws.tr.hess_mask = Kokkos::View<int*, Space>("st_hess_mask", cap);

        // Harvest destination sizing: what the upstream stage built, capped by
        // the destination capacity.
        Pn = to_pool ? Kokkos::min(pool.filled, out_pool->P - out_pool->filled)
                     : pool.filled;
        const int Ph = to_pool ? 1 : Pn;   // host buffers unused when harvesting to a pool
        d_hE = V1("h_E", Ph);
        d_ht = V3("h_t", Ph, ni, 3); d_hr = V3("h_r", Ph, ni, 3);
        d_hcell = V3("h_cell", Ph, 3, 3);
        d_hseed = Vi("h_seed", Ph);
        if (diag && !to_pool) {
            d_hgmax = V1("h_gm", Pn); d_hgrms = V1("h_gr", Pn); d_hiters = Vi("h_it", Pn);
            d_hdelta = V1("h_delta", Pn); d_hrestarts = Vi("h_rs", Pn);
            d_hs1g = V1("h_s1g", Pn); d_hs1i = Vi("h_s1i", Pn);
        }
        out_base0 = to_pool ? out_pool->filled : 0;
    }

    // Only in-flight slots go through the energy pipeline; the mask is restored
    // when the run ends.
    ~TrStepper() {
        bws.geom.slot_mask = decltype(bws.geom.slot_mask)();
        bws.geom.mp_precision = decltype(bws.geom.mp_precision)();
    }

    TrStepper(const TrStepper&) = delete;
    TrStepper& operator=(const TrStepper&) = delete;

    // --- the Stepper concept, host side ---------------------------------

    void bind(Vi slot_state) {
        state = slot_state;
        // Give every slot a valid cell so empty slots cannot produce NaNs. The edge
        // must be realistic: the lattice-shift box is derived from the cell.
        const RealType SAFE_CELL = RealType(20);   // Bohr, ~10.6 A
        Kokkos::parallel_for("st_cell_safe", Kokkos::RangePolicy<Space>(0, cap),
            TrSafeCell<Space>{bws.geom.cell, bws.dofs.cell_ref, bws.dofs.strain,
                              slot_iter, SAFE_CELL});
        // Only SLOT_LIVE slots go through the energy pipeline.
        bws.geom.slot_mask = state;
        // With the hybrid on, the per-slot multipole precision; the pair pass
        // launches once per precision.
        if (hybrid) bws.geom.mp_precision = s_prec;
    }

    V1 energies() const { return bws.geom.energies; }

    void evaluate(int round) {
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);
        batch_expand_asymmetric_unit<MaxRank, Space>(bws.tables, bws.geom, bws.dofs);
        batch_energy_gradient_asymunit<MaxRank>(bws.tables, bws.geom, bws.dofs, es, round, &kt);
    }

    // A refill's init eval lands in the next round, so its Hessian can only be
    // seeded in after_round(), after that round's accept.
    void on_filled(int) { pending_seed = true; }

    void after_round() {
        // Slots refilled at the last checkpoint have just taken their init eval,
        // so their geometry is expanded and their Hessian can be seeded.
        if (pending_seed) {
            pending_seed = false;
            auto d_hmask = bws.tr.hess_mask;
            Kokkos::parallel_for("st_mark_new", Kokkos::RangePolicy<Space>(0, cap),
                TrMarkNew<Space>{d_hmask, state, slot_iter});
            // Same seeder as the lockstep path.
            batch_seed_hessian_blocks<MaxRank, Space>(bws, settings0, sv, cap, warm_mol);
        }
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);

        // Between rounds only: evaluate() updates the cell from the trial strain,
        // so a kappa reset there would make kappa depend on the strain being
        // differentiated. See klasp/batch.hpp.
        batch_rebracket_ewald<Space>(ff, bws.geom, state, s_rebracket);
        {
            auto reb = s_rebracket; auto reev = s_reeval;
            auto st_l = state; auto vol = bws.geom.volume;
            const RealType vmax = vol_abort;
            Kokkos::parallel_for("st_after_rebracket",
                Kokkos::RangePolicy<Space>(0, cap),
                KOKKOS_LAMBDA(const int b) {
                    if (st_l(b) != SLOT_LIVE) return;
                    // Far past the volume budget: finish it to free the slot.
                    if (vmax > RealType(0) && vol(b) > vmax) {
                        st_l(b) = SLOT_DONE;
                        return;
                    }
                    // Geometry unchanged, so Hessian and radius stand, but energy
                    // and gradient need re-evaluating on the new split.
                    if (reb(b)) reev(b) = 1;
                });
        }
    }

    void harvest(Vi idx, int n, int base) {
        if (to_pool) {
            // Device-to-device into the destination pool. cell_params is stale
            // after the round, so derive it first; the next fill re-anchors
            // cell_ref and zeroes strain.
            cell_matrix_to_params_device<Space>(bws.geom, cap);
            stream::for_slot_pairs<Space>("st_harvest_pool", idx, n, out_base0 + base,
                TrHarvestPool<Space>{
                    out_pool->cells.cell, out_pool->cells.cell_inv,
                    out_pool->translations, out_pool->rot_vectors, out_pool->hess_mol,
                    out_pool->cells.kappa, out_pool->cells.volume, out_pool->s1_grad,
                    out_pool->cells.max_hkl, out_pool->cells.max_shift,
                    out_pool->cells.cell_params, out_pool->s1_iters, out_pool->seeds,
                    bws.geom.cell, bws.geom.cell_inv, acc_t, acc_r, sv,
                    bws.geom.kappa, bws.geom.volume, s_gmax,
                    bws.geom.max_hkl, bws.geom.max_shift,
                    bws.geom.cell_params,
                    s_conv_iter, slot_entry, pool.seeds, ni, nmd});
        } else {
            stream::for_slot_pairs<Space>("st_harvest", idx, n, base,
                TrHarvestHost<Space>{
                    d_hE, d_ht, d_hr, d_hcell, d_hseed,
                    d_hgmax, d_hgrms, d_hdelta, d_hs1g,
                    d_hiters, d_hrestarts, d_hs1i,
                    Eacc, s_gmax, s_grms, delta, pool.s1_grad,
                    acc_t, acc_r, bws.geom.cell,
                    s_conv_iter, s_restarts, slot_entry, pool.seeds, pool.s1_iters,
                    ni, diag});
        }
    }

    // --- the Stepper concept, device functors ---------------------------

    TrFill<Space> filler() const {
        return TrFill<Space>{
            bws.geom.cell, bws.dofs.cell_ref, bws.geom.cell_inv,
            bws.geom.kappa, bws.geom.volume, bws.geom.max_hkl, bws.geom.max_shift,
            bws.geom.cell_params, bws.dofs.strain,
            bws.dofs.translations, bws.dofs.rot_vectors, acc_t, acc_r, acc_s,
            bws.dofs.cell_dof_basis, Dsc,
            bws.tables.body_positions, bws.tables.mol_site_count,
            sv, delta, pred,
            s_restarts, slot_iter, s_conv_iter, s_wrapped, slot_entry,
            pool.cells.cell, pool.cells.cell_inv, pool.translations, pool.rot_vectors,
            pool.hess_mol, pool.cells.kappa, pool.cells.volume,
            pool.cells.max_hkl, pool.cells.max_shift, pool.cells.cell_params,
            ni, ns, nmd, ncd, ndof, lattice, warm_mol, DELTA0,
            s_prec, s_reeval, hybrid ? 1 : 0};
    }

    TrPropose<Space> proposer() const {
        return TrPropose<Space>{sv, gacc, pstep, delta, pred,
                                slot_iter, s_wrapped, ndof, s_reeval};
    }

    TrAccept<Space> accepter() const {
        return TrAccept<Space>{sv, bws.dofs.d_gt, bws.dofs.d_gr, bws.dofs.cell_dof_basis,
                               bws.dofs.d_g_strain, Dsc, gacc,
                               Eacc, s_gmax, s_grms, delta,
                               slot_iter, s_conv_iter, s_restarts,
                               trp, grad_tol, DELTA_STALL,
                               ni, ncd, nmd, ndof, max_iters,
                               s_prec, s_reeval, switch_tol, DELTA0, hybrid};
    }
};

// ---------------------------------------------------------------------------

template <int MaxRank, typename Space>
StreamResult batch_optimize_stream_tr(BatchWorkspaceT<Space>& bws,
                                      StreamPoolT<Space>& pool,
                                      const BatchOptimizeSettings& settings,
                                      StreamPoolT<Space>* out_pool) {
    Kokkos::Timer timer;
    // Harvest destination: a downstream pool or host buffers. See batch_stream.hpp.
    const bool to_pool = out_pool != nullptr;
    if (to_pool && out_pool == &pool)
        Kokkos::abort("batch_optimize_stream_tr: source and destination pool "
                      "must differ (refill reads the head, harvest writes the tail)");
    // The harvest cannot grow the destination; overflow is an error.
    if (to_pool && out_pool->P - out_pool->filled < pool.filled)
        Kokkos::abort("batch_optimize_stream_tr: destination pool has less free "
                      "space than the source holds; every harvested structure "
                      "needs somewhere to go");

    TrStepper<MaxRank, Space> st(bws, pool, out_pool, settings);
    st.vol_abort = settings.max_cell_volume;

    stream::Config cfg;
    cfg.capacity = st.cap;
    cfg.refill_interval = settings.refill_interval > 0 ? settings.refill_interval : 10;
    cfg.source_base = pool.consumed;
    cfg.source_count = pool.filled - pool.consumed;
    cfg.harvest_capacity = st.Pn;
    cfg.abort = settings.hooks.abort;

    const stream::Result r = stream::schedule<Space>(cfg, st);
    pool.consumed += r.n_consumed;
    const int n_harvested = r.n_harvested;
    const int ni = st.ni;

    StreamResult out;
    out.n = n_harvested;
    if (to_pool) {
        // Everything stayed on the device; the caller reads the pool. Counts only.
        out_pool->filled = st.out_base0 + n_harvested;
        out.wall_time_s = timer.seconds();
        out.timings = st.kt;
        out.total_iters = r.total_iters;
        return out;
    }

    // Single D2H: device harvest buffers -> host StreamResult.
    out.energies     = Kokkos::View<RealType*, Kokkos::HostSpace>("o_E", n_harvested);
    out.translations = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>("o_t", n_harvested, ni, 3);
    out.rotations    = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>("o_r", n_harvested, ni, 3);
    out.cell         = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>("o_cell", n_harvested, 3, 3);
    using P2 = std::pair<int,int>;
    P2 hr(0, n_harvested);
    Kokkos::deep_copy(out.energies,     Kokkos::subview(st.d_hE, hr));
    Kokkos::deep_copy(out.translations, Kokkos::subview(st.d_ht, hr, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(out.rotations,    Kokkos::subview(st.d_hr, hr, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(out.cell,         Kokkos::subview(st.d_hcell, hr, Kokkos::ALL, Kokkos::ALL));
    {
        auto h_seed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hseed, hr));
        out.seeds.resize(n_harvested);
        for (int j = 0; j < n_harvested; ++j) out.seeds[j] = h_seed(j);
    }
    if (st.diag) {
        out.diagnostics.grad_max = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hgmax, hr));
        out.diagnostics.grad_rms = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hgrms, hr));
        out.diagnostics.iters_to_converge = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hiters, hr));
        out.diagnostics.final_delta = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hdelta, hr));
        out.diagnostics.restarts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hrestarts, hr));
        out.s1_diagnostics.grad_max = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hs1g, hr));
        out.s1_diagnostics.grad_rms = out.s1_diagnostics.grad_max;   // unused, keeps valid() true
        out.s1_diagnostics.iters_to_converge = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(st.d_hs1i, hr));
    }
    out.wall_time_s = timer.seconds();
    out.timings = st.kt;
    out.total_iters = r.total_iters;
    return out;
}

// Explicit instantiations.
template StreamPool allocate_stream_pool<ExecSpace>(int, int);
template void stream_pool_append<ExecSpace>(StreamPool&, const BatchWorkspace&, int, const std::vector<int>&, const BatchDiagnostics&, bool);
template StreamResult batch_optimize_stream_tr<0, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
template StreamResult batch_optimize_stream_tr<1, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
template StreamResult batch_optimize_stream_tr<2, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
template StreamResult batch_optimize_stream_tr<3, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
template StreamResult batch_optimize_stream_tr<4, ExecSpace>(BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);

}  // namespace klasp
