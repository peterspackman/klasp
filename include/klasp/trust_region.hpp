#pragma once
// Trust-region quasi-Newton primitives for the per-structure rigid-body DOF set
// (translation, rotation, cell strain); host/device callable. The dense helpers
// (cholesky, dogleg, bfgs_update) use fixed scratch with n_dof <= MAX_TR_DOF.

#include <Kokkos_Core.hpp>
#include "klasp/types.hpp"
#include "klasp/strain_basis.hpp"
#include "klasp/lbfgs_tr.hpp"

namespace klasp::tr {

// n_indep*6 + n_cell_dof, for Z' <= 8.
constexpr int MAX_TR_DOF = 54;

KOKKOS_INLINE_FUNCTION
RealType dot(const RealType* a, const RealType* b, int n) {
    RealType s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

KOKKOS_INLINE_FUNCTION
RealType norm(const RealType* a, int n) { return Kokkos::sqrt(dot(a, a, n)); }

// y = A x, A symmetric n x n (row-major).
KOKKOS_INLINE_FUNCTION
void matvec(const RealType* A, const RealType* x, RealType* y, int n) {
    for (int i = 0; i < n; ++i) {
        RealType s = 0;
        for (int j = 0; j < n; ++j) s += A[i * n + j] * x[j];
        y[i] = s;
    }
}

// Radius of gyration of n points (interleaved xyz) about their centroid; the
// rotation-DOF scale.
KOKKOS_INLINE_FUNCTION
RealType radius_of_gyration(const RealType* pos, int n) {
    if (n <= 0) return 0;
    RealType c[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i) { c[0] += pos[3*i]; c[1] += pos[3*i+1]; c[2] += pos[3*i+2]; }
    const RealType inv = RealType(1) / n;
    c[0] *= inv; c[1] *= inv; c[2] *= inv;
    RealType s = 0;
    for (int i = 0; i < n; ++i) {
        const RealType dx = pos[3*i]-c[0], dy = pos[3*i+1]-c[1], dz = pos[3*i+2]-c[2];
        s += dx*dx + dy*dy + dz*dz;
    }
    return Kokkos::sqrt(s * inv);
}

// Lower-triangular Cholesky factor L of SPD A (A = L Lᵀ), both n x n row-major.
// Returns false if A is not positive definite.
KOKKOS_INLINE_FUNCTION
bool cholesky(const RealType* A, RealType* L, int n) {
    for (int i = 0; i < n * n; ++i) L[i] = 0;
    for (int j = 0; j < n; ++j) {
        RealType d = A[j*n + j];
        for (int k = 0; k < j; ++k) d -= L[j*n + k] * L[j*n + k];
        if (d <= 0) return false;
        const RealType ljj = Kokkos::sqrt(d);
        L[j*n + j] = ljj;
        for (int i = j + 1; i < n; ++i) {
            RealType s = A[i*n + j];
            for (int k = 0; k < j; ++k) s -= L[i*n + k] * L[j*n + k];
            L[i*n + j] = s / ljj;
        }
    }
    return true;
}

// Solve A x = b given the Cholesky factor L (A = L Lᵀ).
KOKKOS_INLINE_FUNCTION
void chol_solve(const RealType* L, const RealType* b, RealType* x, int n) {
    RealType y[MAX_TR_DOF];
    for (int i = 0; i < n; ++i) {                  // forward: L y = b
        RealType s = b[i];
        for (int k = 0; k < i; ++k) s -= L[i*n + k] * y[k];
        y[i] = s / L[i*n + i];
    }
    for (int i = n - 1; i >= 0; --i) {             // back: Lᵀ x = y
        RealType s = y[i];
        for (int k = i + 1; k < n; ++k) s -= L[k*n + i] * x[k];
        x[i] = s / L[i*n + i];
    }
}

// Predicted reduction of the quadratic model m(p) = g·p + ½ p·B·p, i.e. -m(p).
KOKKOS_INLINE_FUNCTION
RealType predicted_reduction(const RealType* B, const RealType* g,
                             const RealType* p, int n) {
    RealType Bp[MAX_TR_DOF];
    matvec(B, p, Bp, n);
    return -(dot(g, p, n) + RealType(0.5) * dot(p, Bp, n));
}

// Dogleg solution of the trust-region subproblem:
//   min_p g·p + ½ p·B·p   s.t. ‖p‖ ≤ delta,   B symmetric (ideally SPD).
// Writes p (length n) and returns the predicted reduction. Falls back to steepest
// descent to the boundary if B is not positive definite. `Lscratch` is a
// caller-provided n*n buffer for the Cholesky factor.
KOKKOS_INLINE_FUNCTION
RealType dogleg(const RealType* B, const RealType* g, RealType delta,
                int n, RealType* p, RealType* Lscratch) {
    const RealType gnorm = norm(g, n);
    if (gnorm <= RealType(1e-30)) { for (int i = 0; i < n; ++i) p[i] = 0; return 0; }

    RealType* L = Lscratch;
    RealType pN[MAX_TR_DOF];
    const bool pd = cholesky(B, L, n);
    if (pd) {                                       // full Newton step
        RealType negg[MAX_TR_DOF];
        for (int i = 0; i < n; ++i) negg[i] = -g[i];
        chol_solve(L, negg, pN, n);
        if (norm(pN, n) <= delta) {
            for (int i = 0; i < n; ++i) p[i] = pN[i];
            return predicted_reduction(B, g, p, n);
        }
    }

    // Cauchy point pU = -(g·g)/(g·B·g) g (steepest-descent minimiser along -g).
    RealType Bg[MAX_TR_DOF];
    matvec(B, g, Bg, n);
    const RealType gBg = dot(g, Bg, n);
    auto to_boundary = [&]() {                       // -g scaled to ‖p‖ = delta
        const RealType s = delta / gnorm;
        for (int i = 0; i < n; ++i) p[i] = -s * g[i];
    };
    if (gBg <= 0 || !pd) { to_boundary(); return predicted_reduction(B, g, p, n); }

    const RealType tau_u = dot(g, g, n) / gBg;
    RealType pU[MAX_TR_DOF];
    for (int i = 0; i < n; ++i) pU[i] = -tau_u * g[i];
    if (norm(pU, n) >= delta) { to_boundary(); return predicted_reduction(B, g, p, n); }

    // Interpolate pU -> pN, intersect ‖pU + τ(pN-pU)‖ = delta, τ ∈ [0,1].
    RealType d[MAX_TR_DOF];
    for (int i = 0; i < n; ++i) d[i] = pN[i] - pU[i];
    const RealType a = dot(d, d, n);
    const RealType b = 2 * dot(pU, d, n);
    const RealType c = dot(pU, pU, n) - delta * delta;
    const RealType disc = b * b - 4 * a * c;
    RealType tau = (a > 0 && disc > 0) ? (-b + Kokkos::sqrt(disc)) / (2 * a) : 1;
    tau = tau < 0 ? 0 : (tau > 1 ? 1 : tau);
    for (int i = 0; i < n; ++i) p[i] = pU[i] + tau * d[i];
    return predicted_reduction(B, g, p, n);
}

// Powell-damped BFGS update of B (n x n row-major) with pair (s, y):
// B += ŷ ŷᵀ/(ŷᵀs) − (Bs)(Bs)ᵀ/(sᵀBs), with y damped so ŷᵀs >= 0.2 sᵀBs and B
// stays SPD. Returns false only when B is not positive definite along s.
KOKKOS_INLINE_FUNCTION
bool bfgs_update(RealType* B, const RealType* s, const RealType* y, int n) {
    RealType Bs[MAX_TR_DOF];
    matvec(B, s, Bs, n);
    const RealType sBs = dot(s, Bs, n);
    if (!(sBs > RealType(0))) return false;

    const RealType ys = dot(y, s, n);
    RealType yd[MAX_TR_DOF];
    if (ys >= RealType(0.2) * sBs) {
        for (int i = 0; i < n; ++i) yd[i] = y[i];
    } else {
        // θ chosen so that ŷᵀs comes out at exactly 0.2 sᵀBs.
        const RealType theta = RealType(0.8) * sBs / (sBs - ys);
        for (int i = 0; i < n; ++i)
            yd[i] = theta * y[i] + (RealType(1) - theta) * Bs[i];
    }
    const RealType yds = dot(yd, s, n);
    if (!(yds > RealType(1e-30))) return false;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[i*n + j] += yd[i]*yd[j]/yds - Bs[i]*Bs[j]/sBs;
    return true;
}

// Trust-radius update after a rejected step: the radius only shrinks on
// rejection (not Nocedal & Wright Alg. 4.1). On reaching the floor the caller
// restarts from the accepted point with a fresh model; after `max_restarts` it
// stays at the floor.
struct RadiusUpdate {
    RealType delta;    // radius for the next iteration
    bool restart;      // caller should reset the Hessian approximation
};

KOKKOS_INLINE_FUNCTION
RadiusUpdate shrink_radius(RealType delta, bool collapsed, int restarts,
                           RealType delta0, RealType delta_min, int max_restarts) {
    const RealType shrunk = delta * (collapsed ? RealType(0.25) : RealType(0.5));
    if (shrunk >= delta_min)     return {shrunk, false};
    if (restarts < max_restarts) return {delta0, true};
    return {delta_min, false};
}


// Per-structure steps shared by the lockstep and streaming optimisers.

// Per-DOF scaling D so the trust radius is Bohr of atomic displacement: rotation
// by the molecule's radius of gyration, strain by the longest cell edge,
// translation (fractional) by its lattice vector length. `cellv` is (batch, 3, 3)
// with lattice vectors as columns; `body` is (n_indep*n_sites, 3) body frame;
// `nsites` is each molecule's real site count and `ns` the array stride.
template <typename CellV, typename Body, typename DscV, typename NsV>
KOKKOS_INLINE_FUNCTION
void compute_dof_scaling(int b, int ni, int ns, int nmd, int ncd,
                         const CellV& cellv, const Body& body, const DscV& Dsc,
                         const NsV& nsites) {
    RealType axis[3], L = 0;
    for (int d = 0; d < 3; ++d) {
        axis[d] = Kokkos::sqrt(cellv(b,0,d)*cellv(b,0,d) + cellv(b,1,d)*cellv(b,1,d)
                             + cellv(b,2,d)*cellv(b,2,d));
        if (axis[d] > L) L = axis[d];
    }
    for (int k = 0; k < ni; ++k) {
        const int nr = nsites(k) > 0 ? nsites(k) : ns;
        RealType c[3] = {0,0,0};
        for (int s = 0; s < nr; ++s) {
            c[0] += body(k*ns+s,0); c[1] += body(k*ns+s,1); c[2] += body(k*ns+s,2);
        }
        const RealType invn = RealType(1)/nr;
        c[0] *= invn; c[1] *= invn; c[2] *= invn;
        RealType sq = 0;
        for (int s = 0; s < nr; ++s) {
            const RealType dx = body(k*ns+s,0)-c[0], dy = body(k*ns+s,1)-c[1],
                           dz = body(k*ns+s,2)-c[2];
            sq += dx*dx + dy*dy + dz*dz;
        }
        RealType rg = Kokkos::sqrt(sq*invn);
        if (rg < RealType(1e-3)) rg = RealType(1e-3);
        for (int d = 0; d < 3; ++d) {
            Dsc(b, k*3+d)        = (axis[d] > RealType(1e-3)) ? axis[d] : RealType(1);
            Dsc(b, ni*3+k*3+d)   = rg;
        }
    }
    for (int s = 0; s < ncd; ++s)
        Dsc(b, nmd+s) = (L > RealType(1e-3)) ? L : RealType(1);
}

// Gather raw gradients into one scaled DOF vector [translations | rotations |
// strain]. The strain part is projected onto the symmetry-adapted basis:
// dE/dq_s = sum_v basis[s][v] dE/deps_v.
template <typename DGT, typename DGR, typename DGS, typename BasisV, typename DscV>
KOKKOS_INLINE_FUNCTION
void pack_scaled_gradient(int b, int ni, int ncd, int nmd,
                          const DGT& dgt, const DGR& dgr, const DGS& dgs,
                          const BasisV& basis, const DscV& Dsc, RealType* out) {
    for (int k = 0; k < ni; ++k)
        for (int d = 0; d < 3; ++d) {
            out[k*3+d]        = dgt(b,k,d) / Dsc(b, k*3+d);
            out[ni*3+k*3+d]   = dgr(b,k,d) / Dsc(b, ni*3+k*3+d);
        }
    if (ncd > 0) {
        RealType geps[6], gq[6];
        for (int v = 0; v < 6; ++v) geps[v] = dgs(b, v);
        basis_project(&basis(b,0,0), geps, ncd, gq);
        for (int s = 0; s < ncd; ++s) out[nmd+s] = gq[s] / Dsc(b, nmd+s);
    }
}

// Largest |gradient| over the scaled DOFs, where every component is Hartree/Bohr.
KOKKOS_INLINE_FUNCTION
void gradient_norms(const RealType* g, int n, RealType& gmax, RealType& grms) {
    gmax = 0;
    RealType sq = 0;
    for (int i = 0; i < n; ++i) {
        const RealType a = Kokkos::fabs(g[i]);
        if (a > gmax) gmax = a;
        sq += g[i]*g[i];
    }
    grms = n > 0 ? Kokkos::sqrt(sq / n) : RealType(0);
}

// Reduce an angle-axis vector to |r| <= pi without changing the rotation. Needed
// because the map is singular at |r| = 2pi. Returns true if r changed.
KOKKOS_INLINE_FUNCTION
bool wrap_angle_axis(RealType* r) {
    const RealType TWO_PI = RealType(2) * RealType(M_PI);
    const RealType theta = Kokkos::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (theta <= RealType(M_PI)) return false;
    RealType tm = Kokkos::fmod(theta, TWO_PI);
    if (tm > RealType(M_PI)) tm -= TWO_PI;
    const RealType s = tm / theta;
    for (int d = 0; d < 3; ++d) r[d] *= s;
    return true;
}

// Wrap fractional translations into [0, 1). Energy and gradient are invariant,
// but the neighbour shift box is fixed about the home cell.
KOKKOS_INLINE_FUNCTION
void wrap_fractional(RealType* t, int n) {
    for (int i = 0; i < n; ++i) t[i] -= Kokkos::floor(t[i]);
}

// Per-structure trust-region state, shared by the lockstep and streaming optimisers.
template <typename Space>
struct StepViews {
    using V2 = SView2D<RealType, Space>;
    using V3 = SView3D<RealType, Space>;
    // Quadratic model (klasp/lbfgs_tr.hpp): seeded diagonal B0 plus the last `mem`
    // curvature pairs.
    V2 B0;                         // (slot, ndof) diagonal, scaled coordinates
    V2 mS, mY, mA, mPS;            // (slot, mem * ndof)
    V2 mys, msa, malpha;           // (slot, mem)
    V2 mq, mbg, mnw, mg;           // (slot, ndof) scratch; mbg holds B pstep between propose and accept
    SView1D<RealType, Space> mgamma;
    SView1D<int, Space> mhc, mhh;
    V2 gacc, pstep;                // accepted gradient, step
    V2 Dsc;                        // per-DOF scaling
    V3 acc_t, acc_r;               // accepted translations / rotations
    V2 acc_s;                      // accepted cell amplitudes (basis coords)
    V3 trans, rots;                // workspace DOFs the energy kernel reads
    V2 strain;                     // workspace Voigt strain
    V3 cell, cell_ref, basis;      // current cell, strain reference, DOF basis
    V3 dgt, dgr;                   // raw translation / rotation gradients
    V2 dgs;                        // raw Voigt strain gradient
    SView1D<RealType, Space> Eacc, delta, pred, gmax, grms;
    SView1D<int, Space> restarts, wrapped;
    // Analytic strain second derivative, Voigt (batch, 6, 6), from batch_force_constants.
    V3 strain_hess;
    int ni = 0, ncd = 0, nmd = 0, ndof = 0;
    int mem = 0;
};

// Curvature pairs kept by the force-field model. When full, the model folds into
// its diagonal instead of evicting the oldest pair.
constexpr int kFFMem = 64;

template <typename Space>
void allocate_model(StepViews<Space>& v, int cap, int ndof, int mem = kFFMem) {
    using V1 = SView1D<RealType, Space>;
    using V2 = SView2D<RealType, Space>;
    using Vi = SView1D<int, Space>;
    v.B0 = V2("tr_B0", cap, ndof);
    v.mem = mem;
    v.mS = V2("tr_S", cap, mem * ndof); v.mY = V2("tr_Y", cap, mem * ndof);
    v.mA = V2("tr_A", cap, mem * ndof); v.mPS = V2("tr_PS", cap, mem * ndof);
    v.mys = V2("tr_ys", cap, mem); v.msa = V2("tr_sa", cap, mem); v.malpha = V2("tr_alpha", cap, mem);
    v.mq = V2("tr_q", cap, ndof); v.mbg = V2("tr_bg", cap, ndof); v.mnw = V2("tr_nw", cap, ndof);
    v.mg = V2("tr_g", cap, ndof);
    v.gacc = V2("tr_gacc", cap, ndof); v.pstep = V2("tr_pstep", cap, ndof);
    v.mgamma = V1("tr_gamma", cap); v.mhc = Vi("tr_hc", cap); v.mhh = Vi("tr_hh", cap);
}

// Load slot b's model; model_store writes it back. gamma stays at one because
// the seeded diagonal already carries the scale.
template <typename Space>
KOKKOS_INLINE_FUNCTION Model<RealType> model_at(const StepViews<Space>& v, int b) {
    Model<RealType> m;
    m.S = &v.mS(b, 0); m.Y = &v.mY(b, 0); m.A = &v.mA(b, 0); m.PS = &v.mPS(b, 0);
    m.ys = &v.mys(b, 0); m.sa = &v.msa(b, 0); m.alpha = &v.malpha(b, 0);
    m.q = &v.mq(b, 0); m.bg = &v.mbg(b, 0); m.tmp = nullptr;
    m.D = &v.B0(b, 0);
    m.n = v.ndof;
    m.mem = v.mem;
    m.gamma = v.mgamma(b); m.hc = v.mhc(b); m.hh = v.mhh(b);
    m.rule = Gamma::Fixed;
    m.fold = true;
    return m;
}

template <typename Space>
KOKKOS_INLINE_FUNCTION void model_store(const StepViews<Space>& v, int b, const Model<RealType>& m) {
    v.mgamma(b) = m.gamma; v.mhc(b) = m.hc; v.mhh(b) = m.hh;
}

// Forget the pairs, keep the seeded diagonal.
template <typename Space>
KOKKOS_INLINE_FUNCTION void model_reset(const StepViews<Space>& v, int b) {
    v.mgamma(b) = RealType(1); v.mhc(b) = 0; v.mhh(b) = 0;
}

// The model's diagonal, unscaled (B / D_i^2), into mnw.
template <typename Space>
KOKKOS_INLINE_FUNCTION void model_unscaled_diagonal(const StepViews<Space>& v, int b) {
    const auto m = model_at(v, b);
    lb_diagonal(Serial{}, m, &v.mnw(b, 0));
    for (int i = 0; i < v.ndof; ++i) {
        const RealType d = v.Dsc(b, i);
        if (d > RealType(0)) v.mnw(b, i) /= d * d;
    }
}

// Dogleg step from the accepted gradient into pstep; returns the predicted reduction.
template <typename Space>
KOKKOS_INLINE_FUNCTION RealType model_dogleg(const StepViews<Space>& v, int b) {
    const auto m = model_at(v, b);
    return lb_dogleg(Serial{}, m, &v.gacc(b, 0), v.delta(b), &v.pstep(b, 0), &v.mnw(b, 0));
}

// Scale the proposed step by t (B pstep with it) and return its predicted reduction.
template <typename Space>
KOKKOS_INLINE_FUNCTION RealType model_scale_step(const StepViews<Space>& v, int b, RealType t) {
    for (int i = 0; i < v.ndof; ++i) { v.pstep(b, i) *= t; v.mbg(b, i) *= t; }
    const auto m = model_at(v, b);
    return lb_reduction(Serial{}, m, &v.gacc(b, 0), &v.pstep(b, 0));
}

// Seed slot b's strain block from the analytic Voigt Hessian, projected into the
// symmetry-adapted basis (B H B^T) and scaled. Only the diagonal is used, floored
// positive: the full block is often indefinite at the start, which defeats the
// Cholesky. The full block remains in bws.tr.strain_hess.
template <typename Space>
KOKKOS_INLINE_FUNCTION
void seed_strain_hessian(const StepViews<Space>& v, int b) {
    const int nmd = v.nmd, ncd = v.ncd;
    if (ncd <= 0) return;
    for (int s = 0; s < ncd; ++s) {
        // (B H B^T)_ss, then into scaled coordinates.
        RealType acc = 0;
        for (int p = 0; p < 6; ++p) {
            RealType row = 0;
            for (int q = 0; q < 6; ++q)
                row += v.strain_hess(b, p, q) * v.basis(b, s, q);
            acc += v.basis(b, s, p) * row;
        }
        const RealType d = v.Dsc(b, nmd + s);
        acc /= (d * d);
        v.B0(b, nmd + s) = (acc > RealType(1e-3)) ? acc : RealType(1e-3);
    }
}

// Trust-region tuning shared by both optimisers.
struct TRParams {
    RealType eta = RealType(0.10);        // min rho to accept
    RealType rho_max = RealType(10.0);    // above this the model is meaningless
    RealType delta0 = RealType(0.20);     // initial / post-restart radius
    RealType delta_max = RealType(1.00);
    RealType delta_min = RealType(1e-4);  // floor: below this the radius has collapsed
    int max_restarts = 3;
    // Relative energy precision below which differences are not evidence (lb_judge).
    RealType precision = RealType(1e-12);
};

struct AdvanceResult {
    RealType rho;
    bool accepted;
    bool restarted;      // the Hessian was reset from the accepted point
};

// Cell half of a proposed step: the fraction of it that keeps the strain inside its
// bounds. The caller scales the whole step by this and recomputes the predicted
// reduction. If the bound blocks progress, the strain reference is first
// re-anchored to the current cell with amplitudes reset to zero.
template <typename Space>
KOKKOS_INLINE_FUNCTION
RealType clip_cell_step(const StepViews<Space>& v, int b) {
    if (v.ncd <= 0) return RealType(1);
    RealType q0[6], dq[6];
    for (int s = 0; s < v.ncd; ++s) {
        q0[s] = v.acc_s(b, s);
        dq[s] = v.pstep(b, v.nmd + s) / v.Dsc(b, v.nmd + s);
    }
    RealType t = strain_step_fraction(&v.basis(b,0,0), q0, dq, v.ncd);
    if (t < RealType(0.5)) {
        // Re-anchor the strain origin to the current cell. Deformations compose
        // multiplicatively, so this step's pred and gacc are in the old frame and its rho
        // is slightly off; the next iteration corrects it.
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) v.cell_ref(b,r,c) = v.cell(b,r,c);
        for (int s = 0; s < v.ncd; ++s) { v.acc_s(b,s) = 0; q0[s] = 0; }
        t = strain_step_fraction(&v.basis(b,0,0), q0, dq, v.ncd);
    }
    return t;
}

// Write the accepted state plus `pstep` into the workspace DOFs. Returns true if a
// rotation was wrapped; the caller must then skip the BFGS update (rho stays valid).
template <typename Space>
KOKKOS_INLINE_FUNCTION
bool apply_step(const StepViews<Space>& v, int b) {
    bool wrapped = false;
    for (int k = 0; k < v.ni; ++k) {
        RealType tv[3], rv[3];
        for (int d = 0; d < 3; ++d) {
            tv[d] = v.acc_t(b,k,d) + v.pstep(b, k*3+d)            / v.Dsc(b, k*3+d);
            rv[d] = v.acc_r(b,k,d) + v.pstep(b, v.ni*3 + k*3 + d) / v.Dsc(b, v.ni*3 + k*3 + d);
        }
        wrap_fractional(tv, 3);
        if (wrap_angle_axis(rv)) wrapped = true;
        for (int d = 0; d < 3; ++d) { v.trans(b,k,d) = tv[d]; v.rots(b,k,d) = rv[d]; }
    }
    if (v.ncd > 0) {
        RealType q[6], eps[6];
        for (int s = 0; s < v.ncd; ++s)
            q[s] = v.acc_s(b,s) + v.pstep(b, v.nmd+s) / v.Dsc(b, v.nmd+s);
        basis_expand(&v.basis(b,0,0), q, v.ncd, eps);
        for (int e = 0; e < 6; ++e) v.strain(b,e) = eps[e];
    }
    return wrapped;
}

// Adopt the trial point. Translations and rotations come from the workspace views
// since apply_step may have wrapped them.
template <typename Space>
KOKKOS_INLINE_FUNCTION
void commit_step(const StepViews<Space>& v, int b) {
    for (int k = 0; k < v.ni; ++k)
        for (int d = 0; d < 3; ++d) {
            v.acc_t(b,k,d) = v.trans(b,k,d);
            v.acc_r(b,k,d) = v.rots(b,k,d);
        }
    for (int s = 0; s < v.ncd; ++s)
        v.acc_s(b,s) += v.pstep(b, v.nmd+s) / v.Dsc(b, v.nmd+s);
}

// Put the workspace DOFs back to the accepted point after a rejected step.
template <typename Space>
KOKKOS_INLINE_FUNCTION
void rollback_step(const StepViews<Space>& v, int b) {
    for (int k = 0; k < v.ni; ++k)
        for (int d = 0; d < 3; ++d) {
            v.trans(b,k,d) = v.acc_t(b,k,d);
            v.rots(b,k,d)  = v.acc_r(b,k,d);
        }
    if (v.ncd > 0) {
        RealType q[6], eps[6];
        for (int s = 0; s < v.ncd; ++s) q[s] = v.acc_s(b,s);
        basis_expand(&v.basis(b,0,0), q, v.ncd, eps);
        for (int e = 0; e < 6; ++e) v.strain(b,e) = eps[e];
    }
}

// Verdict on a trial step from the model-vs-actual reduction ratio.
struct StepVerdict {
    RealType rho;
    bool finite;      // trial energy usable at all
    bool collapsed;   // non-finite, or rho so large the model is meaningless
    bool accept;
};

KOKKOS_INLINE_FUNCTION
StepVerdict judge_step(RealType E_trial, RealType E_accepted, RealType pred,
                       RealType eta, RealType rho_max) {
    StepVerdict v{};
    v.finite = (E_trial == E_trial) && E_trial < RealType(1e300)
                                    && E_trial > RealType(-1e300);
    const bool has_pred = pred > RealType(1e-30);
    v.rho = has_pred ? (E_accepted - E_trial) / pred : RealType(-1);
    v.collapsed = !v.finite || v.rho > rho_max;
    v.accept = v.finite && !v.collapsed && v.rho > eta && has_pred;
    return v;
}

// One accept/reject decision: judge the trial, update the model, then commit or
// roll back and shrink. `wrapped` suppresses the BFGS update after a rotation wrap.
template <typename Space>
KOKKOS_INLINE_FUNCTION
AdvanceResult advance_step(const StepViews<Space>& v, int b, RealType E_trial,
                           const TRParams& p, RealType precision) {
    const int ndof = v.ndof;
    StepVerdict verdict{};
    verdict.finite = Kokkos::isfinite(E_trial);
    {
        const RealType noise = precision * Kokkos::fmax(Kokkos::fabs(v.Eacc(b)), RealType(1));
        bool unusable = false;
        lb_judge(E_trial, v.Eacc(b), v.pred(b), p.eta, p.rho_max, noise,
                 verdict.rho, verdict.accept, unusable);
        verdict.collapsed = unusable;
    }
    AdvanceResult out{verdict.rho, verdict.accept, false};

    // Learn from the step whether or not it is accepted.
    RealType* gtr = &v.mg(b,0);
    if (verdict.finite) {
        pack_scaled_gradient(b, v.ni, v.ncd, v.nmd, v.dgt, v.dgr, v.dgs,
                             v.basis, v.Dsc, gtr);
        if (!v.wrapped(b)) {
            auto m = model_at(v, b);
            // mbg holds B pstep from the propose.
            lb_update(Serial{}, m, &v.pstep(b,0), gtr, &v.gacc(b,0), true);
            model_store(v, b, m);
        }
    }

    if (verdict.accept) {
        for (int i = 0; i < ndof; ++i) v.gacc(b,i) = gtr[i];
        v.Eacc(b) = E_trial;
        commit_step(v, b);
        RealType pnorm = 0;
        for (int i = 0; i < ndof; ++i) pnorm += v.pstep(b,i)*v.pstep(b,i);
        if (verdict.rho > RealType(0.75)
            && Kokkos::sqrt(pnorm) > RealType(0.8)*v.delta(b))
            v.delta(b) = Kokkos::fmin(RealType(2)*v.delta(b), p.delta_max);
    } else {
        rollback_step(v, b);
        const auto ru = shrink_radius(v.delta(b), verdict.collapsed, v.restarts(b),
                                      p.delta0, p.delta_min, p.max_restarts);
        if (ru.restart) {
            v.restarts(b) += 1;
            out.restarted = true;
            model_reset(v, b);
        }
        v.delta(b) = ru.delta;
    }
    // Convergence is measured at the accepted point.
    gradient_norms(&v.gacc(b,0), ndof, v.gmax(b), v.grms(b));
    return out;
}


}  // namespace klasp::tr
