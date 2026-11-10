// The device-resident relaxer, shared by every energy backend. Must build
// without KLASP_ENABLE_PET.

#include "klasp/device_relax.hpp"

#include <algorithm>
#include <cstdlib>

#include "klasp/cell.hpp"      // matrix_to_cell_params
#include "klasp/constants.hpp"
#include "klasp/crystal_dof.hpp"
#include "klasp/device_optim.hpp"
#include "klasp/pet_relax.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace klasp {

namespace {

using devopt::kHist;

// Restore-to-accepted-point kernel for the round-budget exit. A functor because
// nvcc rejects an extended lambda inside the generic driver lambda.
struct RoundBudgetRestore {
    devopt::RView1D x, xbase, energy, ebase;
    devopt::IView1D active, dofoff, nasym;

    KOKKOS_INLINE_FUNCTION void operator()(const int b) const {
        if (!active(b)) return;
        const int off = dofoff(b), nd = 3 * nasym(b) + 6;
        for (int k = 0; k < nd; ++k) x(off + k) = xbase(off + k);
        energy(b) = ebase(b);
        active(b) = 0;
    }
};

}  // namespace

// Backend-agnostic device-resident relaxer (declared in device_relax.hpp). The
// evaluation is `backend.eval_fn`, or `backend.force_fn` wrapped in the DOF
// policy's fold when the backend speaks per-atom forces.
std::vector<PetRelaxResult> device_relax(
    const std::vector<CrystalGeometry>& inits, const PetRelaxOptions& opts,
    const PetSymBatch* symb, DeviceBackend& be) {
    using devopt::IView1D;
    using devopt::RView1D;
    using devopt::RView2D;
    using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
    const int B = static_cast<int>(inits.size());
    const bool sym = (symb != nullptr);
    std::vector<PetRelaxResult> res(B);
    for (int b = 0; b < B; ++b) res[b].structure = inits[b];

    // The dense model-Hessian preconditioner is built only up to this DOF count;
    // larger structures use the scalar L-BFGS scaling.
    constexpr int kMaxPrecondDof = 180;  // N <= 58 atoms
    const bool precond = opts.precondition;

    // h_N = network atoms (full cell); h_na = DOF atoms (asymmetric unit for sym,
    // h_N for free); ndof = 3*h_na + 6.
    std::vector<int> h_N(B), h_na(B), h_dofoff(B), h_atomoff(B), h_histoff(B), h_poff(B, -1), h_sopoff(B, 0);
    std::vector<int> h_indoff(B, 0);
    int D = 0, Ntot = 0, Hd = 0, NopTot = 0, NindepTot = 0;
    std::size_t Lcap = 0;
    for (int b = 0; b < B; ++b) {
        const int N = static_cast<int>(inits[b].atomic_numbers.size());
        // Real asymmetric-unit atoms, not the padded stride.
        const int na = sym ? symb->asym_off[b].back() : N;
        h_N[b] = N; h_na[b] = na;
        h_atomoff[b] = Ntot; h_dofoff[b] = D; h_histoff[b] = Hd; h_sopoff[b] = NopTot;
        h_indoff[b] = NindepTot;
        const int ndof = 3 * na + 6;
        if (precond && ndof <= kMaxPrecondDof) { h_poff[b] = (int)Lcap; Lcap += (std::size_t)ndof * ndof; }
        Ntot += N; D += ndof; Hd += ndof * kHist;
        if (sym) { NopTot += symb->n_op[b]; NindepTot += symb->n_indep[b]; }
    }

    RView1D x("pet_dx", D), grad("pet_dg", D), gprev("pet_dgp", D), d("pet_dd", D),
            xbase("pet_dxb", D), last("pet_dl", D), q("pet_dq", D);
    RView1D Sd("pet_dS", Hd), Yd("pet_dY", Hd), rho("pet_drho", (size_t)B * kHist);
    RView2D defv("pet_ddef", B, 9);
    RView2D& h0pet = be.h0;                  // reference cell, CellKernel frame (backend units)
    RView1D volv("pet_dvol", B), energy("pet_de", B), ebase("pet_deb", B), gd("pet_dgd", B),
            maxg("pet_dmg", B), maxs("pet_dms", B), cf("pet_dcf", B), gpm("pet_dgpm", B);
    // This round's evaluation, kept apart from the accepted point's
    // energy/maxg/maxs because a backtrack stages an unaccepted trial.
    RView1D r_energy("pet_dre", B), r_maxg("pet_drg", B), r_maxs("pet_drs", B);
    IView1D active("pet_dac", B), conv("pet_dcv", B), iters("pet_dit", B),
            hh("pet_dhh", B), hc("pet_dhc", B), haveprev("pet_dhp", B), stallc("pet_dstc", B);
    IView1D Nb("pet_dN", B), dofoff("pet_ddo", B), atomoff("pet_dao", B), histoff("pet_dho", B);
    // Per-structure periodicity, so an isolated molecule is not wrapped.
    IView1D per("pet_dper", B);
    // Preconditioner Cholesky factors and solve scratch.
    RView1D Lmat("pet_dL", std::max<std::size_t>(Lcap, 1)), py("pet_dpy", D);
    IView1D poff("pet_dpoff", B), hasP("pet_dhasP", B);
    RView2D hbcell("pet_dhb", B, 9);
    // Symmetry-constrained data (flat symops indexed by sopoff); sized 1 in free mode.
    IView1D nasym("pet_dnas", B), nopv("pet_dnop", B), sopoff("pet_dsop", B);
    IView1D nindepv("pet_dnid", B);
    // Ragged symmetry layout, flat with per-structure bases. asym_off and full_off
    // carry one extra prefix-sum sentinel per structure.
    IView1D indoff("pet_dind", B), atomoff2("pet_dao2", B);
    IView1D sitesall("pet_dsit", std::max(NindepTot, 1));
    IView1D asymoffall("pet_daof", std::max(NindepTot + B, 1));
    IView1D fulloffall("pet_dfof", std::max(NopTot + B, 1));
    IView1D opofatom("pet_dooa", std::max(Ntot, 1));
    RView1D M0all("pet_dM0", std::max(NopTot, 1) * 9), tcall("pet_dtc", std::max(NopTot, 1) * 3),
            Rall("pet_dR", std::max(NopTot, 1) * 9);
    IView1D indepall("pet_dind", std::max(NopTot, 1));

    // Trust-region state (devopt::TrDevStep); the accepted point and its gradient
    // live in xbase / gprev, the step in d, the Newton step in last.
    const bool tr = opts.trust_region && !std::getenv("KLASP_PET_LBFGS");
    const std::size_t Dtr = tr ? (std::size_t) D : 0, Btr = tr ? (std::size_t) B : 0;
    RView1D bgv("pet_dbg", std::max<std::size_t>(Dtr, 1));
    RView1D Str("pet_dtrS", std::max<std::size_t>(Dtr * devopt::kTrMem, 1)),
            Ytr("pet_dtrY", std::max<std::size_t>(Dtr * devopt::kTrMem, 1)),
            Atr("pet_dtrA", std::max<std::size_t>(Dtr * devopt::kTrMem, 1)),
            PStr("pet_dtrPS", std::max<std::size_t>(Dtr * devopt::kTrMem, 1));
    RView1D Pmat("pet_dP", tr ? std::max<std::size_t>(Lcap, 1) : 1);   // shapes, L L^T
    IView1D ndofv("pet_dndof", B);
    RView1D ystr("pet_dtrys", std::max<std::size_t>(Btr * devopt::kTrMem, 1)),
            satr("pet_dtrsa", std::max<std::size_t>(Btr * devopt::kTrMem, 1)),
            altr("pet_dtralpha", std::max<std::size_t>(Btr * devopt::kTrMem, 1));
    RView1D deltav("pet_dtrdelta", std::max<std::size_t>(Btr, 1)), gammav("pet_dtrgamma", std::max<std::size_t>(Btr, 1)),
            predv("pet_dtrpred", std::max<std::size_t>(Btr, 1));
    IView1D restartv("pet_dtrrst", std::max<std::size_t>(Btr, 1)), freshv("pet_dtrfresh", std::max<std::size_t>(Btr, 1));
    if (tr) Kokkos::deep_copy(freshv, 1);
    // How many structures run at once; waiting ones take a finished one's place.
    // The line search needs them all at once.
    const int cap = (tr && opts.max_active > 0) ? std::min(opts.max_active, B) : B;
    std::vector<char> started(B, cap >= B ? 1 : 0);

    // Geometry staging Views, sized to the full batch; each round only the active
    // prefix is populated.
    devopt::DeviceGeom& geom = be.geom;      // pre-allocated by the backend
    IView1D& spec_full = be.spec_full;       // pre-staged by the backend (PET species / unused)
    IView1D actv("pet_dact", B), caoffv("pet_dcaoff", B), csid("pet_dcsid", Ntot);

    auto hm = [](auto v){ return Kokkos::create_mirror_view(v); };
    auto hN = hm(Nb), hdo = hm(dofoff), hao = hm(atomoff), hho = hm(histoff);
    auto hac = hm(active), hcv = hm(conv), hit = hm(iters), hhh = hm(hh), hhc = hm(hc), hhp = hm(haveprev);
    auto hcf = hm(cf), hx = hm(x);
    auto hact = hm(actv), hcaoff = hm(caoffv), hcsid = hm(csid);
    auto hpoff = hm(poff);
    auto hhb = hm(hbcell);
    auto hnas = hm(nasym);
    auto hper = hm(per);
    auto hnop = hm(nopv);
    auto hnid = hm(nindepv);
    auto hindo = hm(indoff); auto hao2 = hm(atomoff2);
    auto hsit = hm(sitesall); auto haof = hm(asymoffall);
    auto hfof = hm(fulloffall); auto hooa = hm(opofatom);
    auto hsop = hm(sopoff);
    auto hM0 = hm(M0all);
    auto htc = hm(tcall);
    auto hR = hm(Rall);
    auto hind = hm(indepall);
    for (int b = 0; b < B; ++b) {
        const auto& s = inits[b];
        const int N = h_N[b], na = h_na[b], off = h_dofoff[b];
        const int ndof = 3 * na + 6;
        hN(b) = N; hdo(b) = off; hao(b) = h_atomoff[b]; hho(b) = h_histoff[b];
        hac(b) = 1; hcv(b) = 0; hit(b) = 0; hhh(b) = 0; hhc(b) = 0; hhp(b) = 0;
        hcf(b) = static_cast<double>(N > 0 ? N : 1);  // cell_factor = full-cell atoms (both modes)
        hpoff(b) = h_poff[b];
        hnas(b) = na;
        hper(b) = inits[b].periodic ? 7 : 0;  // all three axes, or none
        // Initial Bohr cell, row-major [r][c] (columns = lattice vectors).
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) hhb(b, i * 3 + j) = s.cell[i][j];
        if (!sym) {
            for (int i = 0; i < N; ++i)
                for (int c = 0; c < 3; ++c) hx(off + 3 * i + c) = s.positions[i][c];  // Bohr
            for (int k = 0; k < 6; ++k) hx(off + 3 * N + k) = 0.0;
        } else {
            const auto& xi = symb->x_init[b];  // [3*na + 6] asym coords (Bohr) + zero strain
            for (int t = 0; t < ndof; ++t) hx(off + t) = xi[t];
            hnop(b) = symb->n_op[b]; hsop(b) = h_sopoff[b];
            hnid(b) = symb->n_indep[b];
            hindo(b) = h_indoff[b]; hao2(b) = h_atomoff[b];
            const int io = h_indoff[b];
            for (int k = 0; k < symb->n_indep[b]; ++k) hsit(io + k) = symb->sites_of_indep[b][k];
            for (int k = 0; k <= symb->n_indep[b]; ++k) haof(io + b + k) = symb->asym_off[b][k];
            for (int op = 0; op <= symb->n_op[b]; ++op) hfof(h_sopoff[b] + b + op) = symb->full_off[b][op];
            for (int i = 0; i < N; ++i) hooa(h_atomoff[b] + i) = symb->op_of_atom[b][i];
            const int so = h_sopoff[b];
            for (int op = 0; op < symb->n_op[b]; ++op) {
                hind(so + op) = symb->indep_of_op[b][op];
                for (int e = 0; e < 9; ++e) { hM0((so+op)*9+e) = symb->M0[b][op*9+e]; hR((so+op)*9+e) = symb->R[b][op*9+e]; }
                for (int e = 0; e < 3; ++e) htc((so+op)*3+e) = symb->tc[b][op*3+e];
            }
        }
    }
    Kokkos::deep_copy(Nb, hN); Kokkos::deep_copy(dofoff, hdo); Kokkos::deep_copy(atomoff, hao);
    Kokkos::deep_copy(histoff, hho); Kokkos::deep_copy(active, hac); Kokkos::deep_copy(conv, hcv);
    Kokkos::deep_copy(iters, hit); Kokkos::deep_copy(hh, hhh); Kokkos::deep_copy(hc, hhc);
    Kokkos::deep_copy(haveprev, hhp); Kokkos::deep_copy(cf, hcf); Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(poff, hpoff); Kokkos::deep_copy(hbcell, hhb); Kokkos::deep_copy(nasym, hnas);
    Kokkos::deep_copy(per, hper);
    if (sym) {
        Kokkos::deep_copy(nopv, hnop); Kokkos::deep_copy(sopoff, hsop);
        Kokkos::deep_copy(nindepv, hnid);
        Kokkos::deep_copy(indoff, hindo); Kokkos::deep_copy(atomoff2, hao2);
        Kokkos::deep_copy(sitesall, hsit); Kokkos::deep_copy(asymoffall, haof);
        Kokkos::deep_copy(fulloffall, hfof); Kokkos::deep_copy(opofatom, hooa);
        Kokkos::deep_copy(M0all, hM0); Kokkos::deep_copy(tcall, htc); Kokkos::deep_copy(Rall, hR);
        Kokkos::deep_copy(indepall, hind);
    }

    // Optimizer state is in Bohr/Hartree; the backend converts with gunit, f2hb, e2h.
    const double ftol = opts.force_tol;          // Ha/Bohr
    const double stol = opts.stress_tol;         // Ha/Bohr^3
    const double max_step = opts.max_step;       // Bohr
    const double max_strain = opts.max_strain_step;
    const double bohr3 = be.volunit;             // cell determinant -> Bohr^3

    // ===================================================================
    // Optimizer driver, generic over the DOF policy (free / symmetric). Fills `res`.
    // ===================================================================
    auto run = [&](auto dof) {
        using Dof = decltype(dof);

        // Model-Hessian preconditioner, built once from the initial geometry.
        // hasP(b) = 1 when usable.
        if (precond) {
            Kokkos::deep_copy(hasP, 0);
            devopt::PrecondKernel<Dof> pk{x, Lmat, dofoff, poff, hasP, hbcell, dof};
            Kokkos::parallel_for("pet_dev_precond", RangePolicy(0, B), pk);
            if (tr) {
                auto hnd = Kokkos::create_mirror_view(ndofv);
                for (int b = 0; b < B; ++b) hnd(b) = 3 * h_na[b] + 6;
                Kokkos::deep_copy(ndofv, hnd);
                devopt::TrShapeKernel sk{Lmat, Pmat, poff, hasP, dofoff, ndofv};
                Kokkos::parallel_for("pet_dev_tr_shape", devopt::TrPolicy(B, 1, devopt::kTrLanes), sk);
            }
        }

        // Geometry-build kernels over the compact active set (Kc structures, Mc atoms).
        auto build_geom = [&](int Kc, int Mc) {
            devopt::CellKernel<Dof> ck{actv, dofoff, caoffv, Nb, per, x, cf, volv, defv, h0pet, geom, bohr3, dof};
            Kokkos::parallel_for("pet_dev_cell", RangePolicy(0, Kc), ck);
            devopt::PosKernel<Dof> pk{csid, actv, dofoff, caoffv, atomoff, spec_full, x, defv, geom, be.gunit, dof};
            Kokkos::parallel_for("pet_dev_pos", RangePolicy(0, Mc), pk);
        };

        auto run_step = [&](int Kc, bool initial) {
            if (tr) {
                devopt::TrDevStep<Dof> f;
                f.x = x; f.grad = grad; f.xacc = xbase; f.gacc = gprev; f.p = d; f.q = q; f.nw = last;
                f.bg = bgv; f.tmp = py;
                f.S = Str; f.Y = Ytr; f.A = Atr; f.PS = PStr; f.ys = ystr; f.sa = satr; f.alpha = altr;
                f.delta = deltav; f.energy = energy; f.maxg = maxg; f.maxs = maxs; f.gamma = gammav;
                f.pred = predv; f.ebase = ebase;
                f.restarts = restartv; f.hc = hc; f.hh = hh; f.iters = iters; f.fresh = freshv;
                f.active = active; f.conv = conv;
                f.dofoff = dofoff; f.act = actv; f.Lmat = Lmat; f.Pmat = Pmat; f.poff = poff; f.hasP = hasP; f.dof = dof;
                f.r_energy = r_energy; f.r_maxg = r_maxg; f.r_maxs = r_maxs;
                // Trust radii (0.2, 1, 1e-4 Angstrom) and starting curvature
                // (identity, 1 eV/Angstrom^2) as in chmpy, in Bohr and Hartree.
                constexpr double bohr = 1.0 / 0.52917721067;
                f.tune.ftol = ftol; f.tune.stol = stol;
                f.tune.delta0 = 0.2 * bohr; f.tune.delta_max = 1.0 * bohr; f.tune.delta_min = 1e-4 * bohr;
                f.tune.gamma0 = (1.0 / 27.211386245988) / (bohr * bohr);
                f.tune.precision = opts.energy_precision;
                f.tune.maxiter = opts.max_iter_full;
                Kokkos::parallel_for("pet_dev_tr_step", devopt::TrPolicy(Kc, 1, devopt::kTrLanes), f);
                return;
            }
            devopt::DevStep<Dof> f;
            f.x=x; f.grad=grad; f.gprev=gprev; f.d=d; f.xbase=xbase; f.last=last; f.q=q;
            f.Sd=Sd; f.Yd=Yd; f.rho=rho; f.defv=defv; f.volv=volv; f.gpm=gpm; f.stallc=stallc;
            f.energy=energy; f.ebase=ebase; f.gd=gd; f.maxg=maxg; f.maxs=maxs;
            f.active=active; f.conv=conv; f.iters=iters; f.hh=hh; f.hc=hc; f.haveprev=haveprev;
            f.dofoff=dofoff; f.histoff=histoff; f.act=actv; f.cf=cf;
            f.Lmat=Lmat; f.py=py; f.poff=poff; f.hasP=hasP; f.dof=dof;
            f.r_energy=r_energy; f.r_maxg=r_maxg; f.r_maxs=r_maxs;
            f.ftol=ftol; f.stol=stol; f.max_step=max_step; f.max_strain=max_strain;
            f.maxiter=opts.max_iter_full; f.initial=initial;
            Kokkos::parallel_for("pet_dev_step", RangePolicy(0, Kc), f);
        };

        // The evaluation seam: eval_fn directly, or force_fn with the policy's fold.
        const DofEval out{grad, r_energy, r_maxg, r_maxs, actv, caoffv, dofoff};
        std::function<void(int, int)> evaluate;
        if (be.eval_fn) {
            evaluate = [&be, out](int Kc, int Mc) { be.eval_fn(Kc, Mc, out); };
        } else {
            evaluate = [&, out, dof](int Kc, int Mc) {
                const BackendForces br = be.force_fn(Kc, Mc);
                devopt::FoldKernel<Dof> fk{br.forces, br.virial, br.energy, defv, volv, cf,
                                           actv, caoffv, dofoff, grad,
                                           r_energy, r_maxg, r_maxs, be.f2hb, be.e2h, dof};
                Kokkos::parallel_for("pet_dev_fold", RangePolicy(0, Kc), fk);
            };
        }

        // Pack the active list (act/caoff/csid) on the host from the active flags;
        // returns Kc and sets Mc to the active atom count.
        int Mc = 0;
        auto compact = [&]() {
            Kokkos::deep_copy(hac, active);
            int Kc = 0; Mc = 0;
            int room = cap;
            for (int b = 0; b < B; ++b) room -= (hac(b) && started[b]) ? 1 : 0;
            for (int b = 0; b < B; ++b) {
                if (!hac(b)) continue;
                if (!started[b]) {
                    if (room <= 0) continue;
                    started[b] = 1, --room;
                }
                hact(Kc) = b; hcaoff(Kc) = Mc;
                for (int i = 0; i < h_N[b]; ++i) hcsid(Mc + i) = Kc;
                Mc += h_N[b]; ++Kc;
            }
            if (Kc > 0) {
                // Copy only the active prefix.
                Kokkos::deep_copy(Kokkos::subview(actv, std::make_pair(0, Kc)),
                                  Kokkos::subview(hact, std::make_pair(0, Kc)));
                Kokkos::deep_copy(Kokkos::subview(caoffv, std::make_pair(0, Kc)),
                                  Kokkos::subview(hcaoff, std::make_pair(0, Kc)));
                Kokkos::deep_copy(Kokkos::subview(csid, std::make_pair(0, Mc)),
                                  Kokkos::subview(hcsid, std::make_pair(0, Mc)));
            }
            return Kc;
        };

        auto do_round = [&](int kc, bool initial) {
            build_geom(kc, Mc);
            evaluate(kc, Mc);
            run_step(kc, initial);
        };

        // Round 0 proposes; later rounds accept/backtrack and harvest converged
        // structures.
        int Kc = compact();
        do_round(Kc, /*initial=*/true);
        // With a window, each structure takes at most max_iter_full + 1
        // evaluations and B / cap of them share a place.
        const int waves = (B + cap - 1) / cap;
        const int max_rounds = tr ? (opts.max_iter_full + 2) * waves + 50 : opts.max_iter_full * 10 + 50;
        for (int round = 0; round < max_rounds; ++round) {
            Kc = compact();
            if (Kc == 0) break;
            do_round(Kc, /*initial=*/false);
            }
        // Structures still active hit the round budget with `x` at an unaccepted
        // trial; restore them to the accepted point.
        {
        // nasym is the per-structure DOF atom count.
            RoundBudgetRestore rb{x, xbase, energy, ebase, active, dofoff, nasym};
            Kokkos::parallel_for("pet_dev_round_budget_restore",
                Kokkos::RangePolicy<ExecSpace>(0, B), rb);
            Kokkos::fence();
        }
        // Rebuild relaxed geometry (Bohr) and final scalars.
        auto fx = Kokkos::create_mirror_view(x); Kokkos::deep_copy(fx, x);
        auto fe = Kokkos::create_mirror_view(energy); Kokkos::deep_copy(fe, energy);
        auto fmg = Kokkos::create_mirror_view(maxg); Kokkos::deep_copy(fmg, maxg);
        auto fms = Kokkos::create_mirror_view(maxs); Kokkos::deep_copy(fms, maxs);
        auto fit = Kokkos::create_mirror_view(iters); Kokkos::deep_copy(fit, iters);
        auto fcv = Kokkos::create_mirror_view(conv); Kokkos::deep_copy(fcv, conv);
        for (int b = 0; b < B; ++b) {
            const int N = h_N[b], na = h_na[b], off = h_dofoff[b];
            double eps[6]; for (int k = 0; k < 6; ++k) eps[k] = fx(off + 3 * na + k) / (double)(N > 0 ? N : 1);
            double def[9]; crystdof::strain_to_def(eps, def);
            CrystalGeometry s;
            s.periodic = inits[b].periodic;
            s.atomic_numbers = inits[b].atomic_numbers;
            s.positions.resize(N);
            if constexpr (std::is_same_v<Dof, devopt::SymDof>) {
                // r_{op,site} = def*(M0_op q_a + tc_op); must match sym_expand.
                const int no = symb->n_op[b];
                for (int op = 0; op < no; ++op) {
                    const int ki = symb->indep_of_op[b][op];
                    const int ns = symb->sites_of_indep[b][ki];
                    const double* M0op = &symb->M0[b][op * 9];
                    const double* tcop = &symb->tc[b][op * 3];
                    for (int site = 0; site < ns; ++site) {
                        const int a = symb->asym_off[b][ki] + site;
                        const int il = symb->full_off[b][op] + site;
                        const double q[3] = {fx(off + 3 * a), fx(off + 3 * a + 1), fx(off + 3 * a + 2)};
                        double base[3]; crystdof::mat3_vec(M0op, q, base);
                        for (int c = 0; c < 3; ++c) base[c] += tcop[c];
                        double r[3]; crystdof::mat3_vec(def, base, r);
                        s.positions[il] = {r[0], r[1], r[2]};
                    }
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    const double p[3] = {fx(off + 3 * i), fx(off + 3 * i + 1), fx(off + 3 * i + 2)};  // Bohr
                    double r[3]; crystdof::mat3_vec(def, p, r);
                    s.positions[i] = {r[0], r[1], r[2]};
                }
            }
            // Cell columns (Bohr): column c = def * (reference cell column c).
            for (int c = 0; c < 3; ++c) {
                const double v[3] = {hhb(b, c), hhb(b, 3+c), hhb(b, 6+c)};
                double rv[3]; crystdof::mat3_vec(def, v, rv);
                for (int r = 0; r < 3; ++r) s.cell[r][c] = rv[r];
            }
            res[b].structure = s;
            res[b].energy = fe(b);             // already Hartree
            res[b].max_force = fmg(b);         // already Ha/Bohr
            res[b].max_stress = fms(b);        // already Ha/Bohr^3
            res[b].full_iterations = fit(b);
            res[b].converged = fcv(b) != 0;
        }
    };

    // Single DOF-policy dispatch: free vs symmetry-constrained.
    if (sym) run(devopt::SymDof{nasym, nopv, sopoff, indepall, nindepv,
                               indoff, atomoff2, sitesall, asymoffall, fulloffall, opofatom,
                               M0all, tcall, Rall, hbcell});
    else     run(devopt::FreeDof{Nb});
    return res;
}

}  // namespace klasp
