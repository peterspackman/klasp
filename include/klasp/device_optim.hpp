#pragma once
// Device-side kernels for a batched, preconditioned L-BFGS (and trust-region)
// crystal relaxer, independent of the energy backend and the DOF
// parameterization.
//
// The backend is injected as a host callback that yields a DOF gradient and an
// energy (device_relax.hpp); FoldKernel adapts backends that produce per-atom
// forces. The DOF parameterization is a compile-time policy (FreeDof / SymDof).
// The step itself (lbfgs_propose / lbfgs_accept / step_cap) is raw-pointer
// free functions shared with the host relaxer in flex_relax.hpp.
//
// Optimizer state is in atomic units (Bohr / Hartree); see crystal_dof.hpp.
#include "klasp/crystal_dof.hpp"
#include "klasp/lbfgs_tr.hpp"
#include "klasp/types.hpp"

#include <Kokkos_Core.hpp>

namespace klasp {
namespace devopt {

// The optimizer's own View types, so builds without PET need nothing of the
// network's. A backend hands over views of exactly these types.
using RView1D = SView1D<RealType, ExecSpace>;
using RView2D = SView2D<RealType, ExecSpace>;
using IView1D = SView1D<int, ExecSpace>;
using IView2D = SView2D<int, ExecSpace>;

// L-BFGS history depth.
constexpr int kHist = 8;
// If max|grad| fails to drop below 99% of its value for this many consecutive
// accepted steps, reset the history and take a steepest-descent step.
constexpr int kStallReset = 5;

// Model-Hessian preconditioner parameters (Bohr / Hartree). Stretch-model force
// constants k(r) = k0*exp(-A*(r/r0 - 1)) over min-image pairs within rcut, plus
// a soft-mode floor (kP_reg) on the atomic diagonal and a strain-DOF diagonal
// (kP_cell). Only the relative shape matters: H0 is mean-diagonal normalized.
constexpr double kP_k0 = 0.40, kP_A = 3.0, kP_r0 = 2.85, kP_rcut = 6.0;
constexpr double kP_reg = 0.30, kP_cell = 0.50;

// ===================================================================
// Generic dense linear-algebra kernels (per thread, small n).
// ===================================================================

// In-place Cholesky of a dense symmetric [n,n] (row-major) -> lower factor L in
// the lower triangle. Returns false if not positive-definite.
KOKKOS_INLINE_FUNCTION bool chol_dense(double* A, int n) {
    for (int j = 0; j < n; ++j) {
        double d = A[j * n + j];
        for (int k = 0; k < j; ++k) d -= A[j * n + k] * A[j * n + k];
        if (d <= 1e-12) return false;
        d = Kokkos::sqrt(d);
        A[j * n + j] = d;
        const double invd = 1.0 / d;
        for (int i = j + 1; i < n; ++i) {
            double s = A[i * n + j];
            for (int k = 0; k < j; ++k) s -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s * invd;
        }
    }
    return true;
}

// Solve (L L^T) x = b in place (x starts as b). y is scratch of length n.
KOKKOS_INLINE_FUNCTION void chol_solve_dense(const double* L, double* x, double* y, int n) {
    for (int i = 0; i < n; ++i) {           // forward: L y = b
        double s = x[i];
        for (int k = 0; k < i; ++k) s -= L[i * n + k] * y[k];
        y[i] = s / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {       // backward: L^T x = y
        double s = y[i];
        for (int k = i + 1; k < n; ++k) s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
}

// ===================================================================
// DOF kernels over raw pointers. Offsets are pre-applied by the caller
// (e.g. xb = &x(dofoff(b))).
// ===================================================================

// --- Free (Cartesian-in-reference-cell) parameterization: r = def * p. ---

KOKKOS_INLINE_FUNCTION void free_expand(const double def[9], const double* xb, int il, double r[3]) {
    const double p[3] = {xb[3 * il], xb[3 * il + 1], xb[3 * il + 2]};
    crystdof::cart_from_ref(def, p, r);
}

// Fold atomic forces (backend units, scaled by f2hb to Ha/Bohr) onto the
// reference coords: grad_p = -def^T F. Returns max |grad component|.
KOKKOS_INLINE_FUNCTION double free_fold_atoms(const double def[9], const double* Fraw,
                                              double f2hb, int N, double* grad) {
    double mx = 0;
    for (int i = 0; i < N; ++i) {
        const double F[3] = {Fraw[3*i]*f2hb, Fraw[3*i+1]*f2hb, Fraw[3*i+2]*f2hb};  // Ha/Bohr
        crystdof::atom_grad_cart(def, F, &grad[3 * i]);
        for (int c = 0; c < 3; ++c) { const double a = Kokkos::fabs(grad[3*i+c]); mx = a > mx ? a : mx; }
    }
    return mx;
}

// Step-cap metric: max |d| over the nf atomic coords.
KOKKOS_INLINE_FUNCTION double free_cap(const double* db, int nf) {
    double mc = 0;
    for (int k = 0; k < nf; ++k) { const double a = Kokkos::fabs(db[k]); mc = a > mc ? a : mc; }
    return mc;
}

// Stretch-model preconditioner over the atomic DOFs (min-image pairs). Fills
// the atomic blocks of P[ndof*ndof]; the caller adds the floors, normalizes and
// factorizes.
KOKKOS_INLINE_FUNCTION void free_precond_pairs(const double* xb, int N, int ndof,
                                               const double cell[9], const double hi[9],
                                               bool inv_ok, double* P) {
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx[3] = {xb[3*j]-xb[3*i], xb[3*j+1]-xb[3*i+1], xb[3*j+2]-xb[3*i+2]};
            if (inv_ok) {  // minimum image: wrap fractional displacement to [-0.5,0.5]
                double f[3]; crystdof::mat3_vec(hi, dx, f);
                for (int c = 0; c < 3; ++c) f[c] -= Kokkos::round(f[c]);
                crystdof::mat3_vec(cell, f, dx);
            }
            const double r = Kokkos::sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
            if (r < 1e-6 || r > kP_rcut) continue;
            const double k = kP_k0 * Kokkos::exp(-kP_A * (r / kP_r0 - 1.0));
            const double u[3] = {dx[0]/r, dx[1]/r, dx[2]/r};
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c) {
                    const double v = k * u[a] * u[c];
                    P[(3*i+a)*ndof + (3*i+c)] += v;
                    P[(3*j+a)*ndof + (3*j+c)] += v;
                    P[(3*i+a)*ndof + (3*j+c)] -= v;
                    P[(3*j+a)*ndof + (3*i+c)] -= v;
                }
        }
    }
}

// Finish a model-Hessian preconditioner: soft-mode floor on the 3N atomic
// diagonal, strain-DOF diagonal stiffness, mean-diagonal normalization (so the
// absolute scale drops out), then in-place Cholesky. Returns PD success.
KOKKOS_INLINE_FUNCTION bool precond_finish(double* P, int N, int ndof) {
    for (int t = 0; t < 3 * N; ++t) P[t * ndof + t] += kP_reg;
    for (int c = 0; c < 6; ++c) P[(3*N+c)*ndof + (3*N+c)] += kP_cell;
    double tr = 0;
    for (int t = 0; t < ndof; ++t) tr += P[t * ndof + t];
    if (tr > 0) { const double sc = ndof / tr; for (int t = 0; t < ndof * ndof; ++t) P[t] *= sc; }
    return chol_dense(P, ndof);
}

// --- Symmetric parameterization: the DOFs are the asymmetric unit (q_a) +
//     symmetrized strain; the full cell atom (op,site) = def*(M0_op q_a + tc_op).
//     M0_op = h0 R_op h0^-1 (Cartesian); Rall holds the fractional rotation. ---

// Where each molecule's atoms live in the compacted (padding-free) atom list.
// Components may differ in atom count, so molecules are not at a uniform stride.
struct SymLayout {
    int n_indep;                // independent molecules (Z')
    const int* sites_of_indep;  // [n_indep]   real atoms of each independent molecule
    const int* asym_off;        // [n_indep+1] asym atom  = asym_off[k] + site
    const int* full_off;        // [n_op+1]    full atom  = full_off[op] + site
    const int* op_of_atom;      // [n_atoms]   operator owning a flat full-cell atom

    KOKKOS_INLINE_FUNCTION int sites_of_op(int op, const int* indep) const {
        return sites_of_indep[indep[op]];
    }
    KOKKOS_INLINE_FUNCTION int n_full(int n_op) const { return full_off[n_op]; }
};

// Expand full-cell atom il from the asym DOFs: r = def*(M0_op q + tc).

KOKKOS_INLINE_FUNCTION void sym_expand(const double def[9], const double* xb, int il,
                                       const SymLayout& L,
                                       const double* M0, const double* tc, const int* indep,
                                       double r[3]) {
    const int op = L.op_of_atom[il];
    const int site = il - L.full_off[op];
    const int a = L.asym_off[indep[op]] + site;
    const double* M0op = &M0[op * 9];
    const double q[3] = {xb[3*a], xb[3*a+1], xb[3*a+2]};
    double base[3]; crystdof::mat3_vec(M0op, q, base);              // M0 q
    for (int c = 0; c < 3; ++c) base[c] += tc[op * 3 + c];        // + tc
    crystdof::mat3_vec(def, base, r);                               // (I+E)(M0 q + tc)
}

// Fold full-cell forces onto the asym DOFs and symmetrize the strain gradient.
// grad[0..3*nasym) gets g_qa = -sum_op (def M0_op)^T F_{op,site}; grad[nf..nf+6)
// gets the point-group-symmetrized virial. h0 is the reference Bohr cell
// (row-major). Returns max per-atom |grad| in maxF and max stress in ms.
KOKKOS_INLINE_FUNCTION void sym_fold(const double def[9], const double* Fraw, double f2hb,
                                     const double* Wraw, double e2h, int no, const SymLayout& L,
                                     const double* M0, const double* tc, const double* R,
                                     const int* indep, const double h0[9], int nf, double cellfac,
                                     double* grad, double& maxF, double& ms) {
    for (int t = 0; t < nf; ++t) grad[t] = 0.0;
    for (int op = 0; op < no; ++op) {
        const int ki = indep[op];
        const double* M0op = &M0[op * 9];
        const int nsop = L.sites_of_indep[ki];
        for (int site = 0; site < nsop; ++site) {
            const int il = L.full_off[op] + site, a = L.asym_off[ki] + site;
            const double F[3] = {Fraw[3*il]*f2hb, Fraw[3*il+1]*f2hb, Fraw[3*il+2]*f2hb};
            double dTF[3], gq[3];
            crystdof::mat3T_vec(def, F, dTF);     // def^T F
            crystdof::mat3T_vec(M0op, dTF, gq);   // M0^T (def^T F)
            for (int c = 0; c < 3; ++c) grad[3*a + c] -= gq[c];
        }
    }
    // The gradient sums over each atom's orbit; the reported maxF is divided by
    // the orbit size so force_tol means a per-atom force, as on the free path.
    maxF = 0;
    for (int ki = 0; ki < L.n_indep; ++ki) {
        int orbit = 0;
        for (int op = 0; op < no; ++op) orbit += (indep[op] == ki) ? 1 : 0;
        const double inv = orbit > 0 ? 1.0 / orbit : 1.0;
        const int lo = 3 * L.asym_off[ki], hi = 3 * (L.asym_off[ki] + L.sites_of_indep[ki]);
        for (int t = lo; t < hi && t < nf; ++t) {
            const double a = Kokkos::fabs(grad[t]) * inv;
            maxF = a > maxF ? a : maxF;
        }
    }
    // strain: Wsym = sum_op (h R_op h^-1) Wm (h R_op h^-1)^T / n_op, h = def h0.
    double h_[9]; crystdof::mat3_mul(def, h0, h_);
    // Identity if the cell is singular (mat3_inv leaves its output untouched),
    // which makes the symmetrization the identity map.
    double hinv[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const bool h_invertible = crystdof::mat3_inv(h_, hinv);
    double Wm[9];
    { const double Wv[6] = {Wraw[0]*e2h, Wraw[1]*e2h, Wraw[2]*e2h, Wraw[3]*e2h, Wraw[4]*e2h, Wraw[5]*e2h};
      // Cumulative-strain DOF needs dE/dE = W * def^-T; correct before the
      // point-group symmetrization (identity at def=I).
      double Wc[6];
      crystdof::strain_correct_virial(Wv, def, Wc);
      // Decode to the actual tensor before rotating (see voigt_full_to_tensor).
      crystdof::voigt_full_to_tensor(Wc, Wm); }
    double Wsym[9] = {0,0,0,0,0,0,0,0,0};
    if (!h_invertible) {
        for (int t = 0; t < 9; ++t) Wsym[t] = Wm[t] * no;   // undone by invn below
    } else {
        for (int op = 0; op < no; ++op) {
            const double* Rop = &R[op * 9];
            double tmp[9], M[9], Mt[9], MW[9], MWMt[9];
            crystdof::mat3_mul(h_, Rop, tmp); crystdof::mat3_mul(tmp, hinv, M);  // M = h R h^-1
            crystdof::mat3_transpose(M, Mt);
            crystdof::mat3_mul(M, Wm, MW); crystdof::mat3_mul(MW, Mt, MWMt);
            for (int t = 0; t < 9; ++t) Wsym[t] += MWMt[t];
        }
    }
    ms = 0;
    const double invn = 1.0 / no, vol = Kokkos::fabs(crystdof::mat3_det(h_));
    double Wavg[9];
    for (int t = 0; t < 9; ++t) Wavg[t] = Wsym[t] * invn;
    for (int k = 0; k < 6; ++k) {
        // Back to full-tensor Voigt, as the free path's strain_grad produces.
        const double w = crystdof::tensor_to_voigt_full(Wavg, k);
        grad[nf + k] = w / cellfac;
        const double s = Kokkos::fabs(w) / (vol > 0 ? vol : 1.0);
        ms = s > ms ? s : ms;
    }
}

// Step-cap metric for the asym DOFs: max per-atom displacement norm (SymState).
KOKKOS_INLINE_FUNCTION double sym_cap(const double* db, int ndofatoms) {
    double mc = 0;
    for (int a = 0; a < ndofatoms; ++a) {
        const double dx = db[3*a], dy = db[3*a+1], dz = db[3*a+2];
        const double r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
        mc = r > mc ? r : mc;
    }
    return mc;
}

// --- Rigid-body parameterization: each group carries a translation and a
//     rotation vector, and the cell is fixed. The DOFs of group g are
//     x[6g..6g+3) = t_g (Bohr) and x[6g+3..6g+6) = w_g (angle-axis, radians),
//     and atom i of that group sits at
//
//       r_i = c_g + R(w_g) (p0_i - c_g) + t_g,   c_g = centroid of the reference
//
//     with p0 the geometry the relaxation started from. ---

// Rodrigues rotation matrix from an angle-axis vector w (|w| = angle, radians).
KOKKOS_INLINE_FUNCTION void rigid_rotation(const double w[3], double R[9]) {
    const double th = Kokkos::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if (th < 1e-12) {
        R[0]=1; R[1]=0; R[2]=0; R[3]=0; R[4]=1; R[5]=0; R[6]=0; R[7]=0; R[8]=1;
        return;
    }
    const double kx = w[0]/th, ky = w[1]/th, kz = w[2]/th;
    const double c = Kokkos::cos(th), s = Kokkos::sin(th), C = 1.0 - c;
    R[0] = c + kx*kx*C;    R[1] = kx*ky*C - kz*s; R[2] = kx*kz*C + ky*s;
    R[3] = ky*kx*C + kz*s; R[4] = c + ky*ky*C;    R[5] = ky*kz*C - kx*s;
    R[6] = kz*kx*C - ky*s; R[7] = kz*ky*C + kx*s; R[8] = c + kz*kz*C;
}

// Place one rigid group's atoms. p0 is the group's reference coordinates (Bohr,
// 3*n), c its reference centroid; xg points at the group's six DOFs.
KOKKOS_INLINE_FUNCTION void rigid_expand(const double* xg, const double* p0, const double c[3],
                                         int n, double* r) {
    double R[9];
    rigid_rotation(&xg[3], R);
    for (int i = 0; i < n; ++i) {
        const double d[3] = {p0[3*i] - c[0], p0[3*i+1] - c[1], p0[3*i+2] - c[2]};
        double q[3]; crystdof::mat3_vec(R, d, q);
        for (int k = 0; k < 3; ++k) r[3*i + k] = c[k] + q[k] + xg[k];
    }
}

// Fold one group's atomic forces onto its six DOFs.
//
// dE/dt is minus the net force. dE/dw is the exact derivative of the Rodrigues
// map (Gallego & Yezzi), not the torque. With A = sum_i q_i F_i^T (q_i = R d_i)
// and v(A) = (A23-A32, A31-A13, A12-A21), the chain rule through
// dR/dw_k = (w_k [w]x + [w x (I-R) e_k]x) R / |w|^2 reduces to two dot products
// per component, since tr([a]x A) = a . v(A). At |w| -> 0 it is -torque_k.
KOKKOS_INLINE_FUNCTION void rigid_fold(const double* xg, const double* p0, const double c[3],
                                       const double* F, int n, double* g) {
    double R[9];
    rigid_rotation(&xg[3], R);
    double A[9] = {0,0,0,0,0,0,0,0,0}, Fnet[3] = {0,0,0};
    for (int i = 0; i < n; ++i) {
        const double d[3] = {p0[3*i] - c[0], p0[3*i+1] - c[1], p0[3*i+2] - c[2]};
        double q[3]; crystdof::mat3_vec(R, d, q);
        for (int a = 0; a < 3; ++a) {
            Fnet[a] += F[3*i + a];
            for (int b = 0; b < 3; ++b) A[a*3 + b] += q[a] * F[3*i + b];
        }
    }
    for (int k = 0; k < 3; ++k) g[k] = -Fnet[k];
    const double v[3] = {A[5] - A[7], A[6] - A[2], A[1] - A[3]};
    const double* w = &xg[3];
    const double th2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    if (th2 < 1e-24) {                       // R = I: the bracket is [e_k]x
        for (int k = 0; k < 3; ++k) g[3 + k] = -v[k];
        return;
    }
    const double wv = w[0]*v[0] + w[1]*v[1] + w[2]*v[2];
    const double inv = 1.0 / th2;
    for (int k = 0; k < 3; ++k) {
        // u = (I - R) e_k, column k of I - R.
        const double u[3] = {(k == 0 ? 1.0 : 0.0) - R[k],
                             (k == 1 ? 1.0 : 0.0) - R[3 + k],
                             (k == 2 ? 1.0 : 0.0) - R[6 + k]};
        const double wu[3] = {w[1]*u[2] - w[2]*u[1],
                              w[2]*u[0] - w[0]*u[2],
                              w[0]*u[1] - w[1]*u[0]};
        g[3 + k] = -(w[k] * wv + wu[0]*v[0] + wu[1]*v[1] + wu[2]*v[2]) * inv;
    }
}

// Coordinate metrics for the line search. Free atoms bound the largest
// component; an asymmetric unit or rigid body bounds the largest per-atom
// displacement.
struct FreeCap {
    KOKKOS_INLINE_FUNCTION double operator()(const double* db, int nfree, int) const {
        return free_cap(db, nfree);
    }
};
struct SymCap {
    KOKKOS_INLINE_FUNCTION double operator()(const double* db, int, int ndofatoms) const {
        return sym_cap(db, ndofatoms);
    }
};

// The trust-region cap. `coord` is the parameterization's coordinate metric
// (free_cap or sym_cap); the strain block, stored scaled by cellfac, is
// de-scaled before being bounded. `nstrain` is 6 for a relaxing cell and 0 for
// a fixed one. Returns the factor in (0,1] to scale the step by.
KOKKOS_INLINE_FUNCTION double step_cap(double coord, const double* db, int nfree, int nstrain,
                                       double cellfac, double max_step, double max_strain) {
    double msn = 0;
    for (int k = 0; k < nstrain; ++k) {
        const double a = Kokkos::fabs(db[nfree + k]) / cellfac;
        msn = a > msn ? a : msn;
    }
    double sc = 1.0;
    if (coord > max_step) sc = Kokkos::fmin(sc, max_step / coord);
    if (nstrain > 0 && msn > max_strain) sc = Kokkos::fmin(sc, max_strain / msn);
    return sc;
}

// Stretch-model preconditioner folded from the FULL expanded cell onto the asym
// DOFs via the symop Jacobian J = d r_{op,site}/d q_a = M0_op (strain = 0 at
// build => def = I), P_asym = J^T H_full J. Each physical pair appears once per
// op image, so the folded block is scaled by 1/n_op to match a single cell.
KOKKOS_INLINE_FUNCTION void sym_precond_pairs(const double* xb, int ndof, int no,
                                              const SymLayout& L,
                                              const double* M0, const double* tc, const int* indep,
                                              const double cell[9], const double hi[9],
                                              bool inv_ok, double* P) {
    const int Nfull = L.full_off[no];
    const double invnop = no > 0 ? 1.0 / no : 1.0;
    for (int il = 0; il < Nfull; ++il) {
        const int opi = L.op_of_atom[il], sitei = il - L.full_off[opi];
        const int ai = L.asym_off[indep[opi]] + sitei;   // asym DOF atom
        const double* M0i = &M0[opi * 9];
        const double qi[3] = {xb[3*ai], xb[3*ai+1], xb[3*ai+2]};
        double ri[3]; crystdof::mat3_vec(M0i, qi, ri);
        for (int c = 0; c < 3; ++c) ri[c] += tc[opi*3 + c];
        for (int jl = il + 1; jl < Nfull; ++jl) {
            const int opj = L.op_of_atom[jl], sitej = jl - L.full_off[opj];
            const int aj = L.asym_off[indep[opj]] + sitej;
            const double* M0j = &M0[opj * 9];
            const double qj[3] = {xb[3*aj], xb[3*aj+1], xb[3*aj+2]};
            double rj[3]; crystdof::mat3_vec(M0j, qj, rj);
            for (int c = 0; c < 3; ++c) rj[c] += tc[opj*3 + c];
            double dx[3] = {rj[0]-ri[0], rj[1]-ri[1], rj[2]-ri[2]};
            if (inv_ok) {
                double f[3]; crystdof::mat3_vec(hi, dx, f);
                for (int c = 0; c < 3; ++c) f[c] -= Kokkos::round(f[c]);
                crystdof::mat3_vec(cell, f, dx);
            }
            const double r = Kokkos::sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
            if (r < 1e-6 || r > kP_rcut) continue;
            const double k = invnop * kP_k0 * Kokkos::exp(-kP_A * (r / kP_r0 - 1.0));
            const double u[3] = {dx[0]/r, dx[1]/r, dx[2]/r};
            double Bm[9];  // B = k u u^T (pair stiffness in Cartesian)
            for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) Bm[a*3+c] = k*u[a]*u[c];
            // fold: P(ai,ai)+=M0i^T B M0i, P(aj,aj)+=M0j^T B M0j,
            //       P(ai,aj)-=M0i^T B M0j, P(aj,ai)-=M0j^T B M0i.
            double BMi[9], BMj[9], MiT[9], MjT[9], Sii[9], Sjj[9], Sij[9], Sji[9];
            crystdof::mat3_mul(Bm, M0i, BMi); crystdof::mat3_mul(Bm, M0j, BMj);
            crystdof::mat3_transpose(M0i, MiT); crystdof::mat3_transpose(M0j, MjT);
            crystdof::mat3_mul(MiT, BMi, Sii); crystdof::mat3_mul(MjT, BMj, Sjj);
            crystdof::mat3_mul(MiT, BMj, Sij); crystdof::mat3_mul(MjT, BMi, Sji);
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c) {
                    P[(3*ai+a)*ndof + (3*ai+c)] += Sii[a*3+c];
                    P[(3*aj+a)*ndof + (3*aj+c)] += Sjj[a*3+c];
                    P[(3*ai+a)*ndof + (3*aj+c)] -= Sij[a*3+c];
                    P[(3*aj+a)*ndof + (3*ai+c)] -= Sji[a*3+c];
                }
        }
    }
}

// ===================================================================
// DOF policies: Kokkos-View adapters over the kernels above, selected at
// compile time. Each method takes the packed state by pre-applied offset
// (xb=&x(off), grad=&grad(off), ...).
// ===================================================================

// Free (CellState) parameterization: every atom carries DOFs; r = def * p.
struct FreeDof {
    using Cap = FreeCap;         // the line search's coordinate metric
    IView1D Nb;  // [B] full-cell atom count = DOF-atom count

    KOKKOS_INLINE_FUNCTION int ndofatoms(int b) const { return Nb(b); }

    KOKKOS_INLINE_FUNCTION void expand(int b, int il, const double def[9],
                                       const double* xb, double r[3]) const {
        free_expand(def, xb, il, r);
    }
    KOKKOS_INLINE_FUNCTION void fold(int b, int nf, const double def[9], const double* Fraw,
                                     const double* Wraw, double f2hb, double e2h, double cellfac,
                                     double vol, double* grad, double& maxF, double& ms) const {
        maxF = free_fold_atoms(def, Fraw, f2hb, nf / 3, grad);
        const double W[6] = {Wraw[0]*e2h, Wraw[1]*e2h, Wraw[2]*e2h, Wraw[3]*e2h, Wraw[4]*e2h, Wraw[5]*e2h};
        // Cumulative-strain DOF needs dE/dE = W * def^-T, not the raw virial.
        double Wc[6];
        crystdof::strain_correct_virial(W, def, Wc);
        double g6[6];
        ms = crystdof::strain_grad(Wc, cellfac, vol, g6);
        for (int k = 0; k < 6; ++k) grad[nf + k] = g6[k];
    }
    KOKKOS_INLINE_FUNCTION bool build_precond(int b, int N, int ndof, const double* xb,
                                              const double cell[9], const double hi[9],
                                              bool inv_ok, double* P) const {
        free_precond_pairs(xb, N, ndof, cell, hi, inv_ok, P);
        return precond_finish(P, N, ndof);
    }
};

// Symmetry-constrained (SymState) parameterization: the DOFs are the asymmetric
// unit + symmetrized strain; the full cell is regenerated from the operators.
struct SymDof {
    using Cap = SymCap;          // the line search's coordinate metric
    IView1D nasym, nop, sopoff, indepall;         // [B] / flat symop index
    IView1D nindepv;                              // [B] independent molecules (Z')
    // Ragged layout, flat with per-structure bases: indoff(b) into the per-indep
    // arrays, sopoff(b) into the per-op array, atomoff(b) into the per-atom one.
    IView1D indoff, atomoff2, sitesall, asymoffall, fulloffall, opofatom;
    RView1D M0all, tcall, Rall;                   // flat [NopTot*9 / *3 / *9]
    RView2D hb;                                   // [B,9] reference Bohr cell (row-major)

    KOKKOS_INLINE_FUNCTION int ndofatoms(int b) const { return nasym(b); }

    KOKKOS_INLINE_FUNCTION SymLayout layout(int b) const {
        const int io = indoff(b), so = sopoff(b), ao = atomoff2(b);
        return SymLayout{nindepv(b), &sitesall(io), &asymoffall(io + b), &fulloffall(so + b),
                         &opofatom(ao)};
    }
    KOKKOS_INLINE_FUNCTION void expand(int b, int il, const double def[9],
                                       const double* xb, double r[3]) const {
        const int so = sopoff(b);
        sym_expand(def, xb, il, layout(b), &M0all(so*9), &tcall(so*3), &indepall(so), r);
    }
    KOKKOS_INLINE_FUNCTION void fold(int b, int nf, const double def[9], const double* Fraw,
                                     const double* Wraw, double f2hb, double e2h, double cellfac,
                                     double vol, double* grad, double& maxF, double& ms) const {
        const int so = sopoff(b);
        double h0[9]; for (int e = 0; e < 9; ++e) h0[e] = hb(b, e);
        sym_fold(def, Fraw, f2hb, Wraw, e2h, nop(b), layout(b), &M0all(so*9), &tcall(so*3),
                 &Rall(so*9), &indepall(so), h0, nf, cellfac, grad, maxF, ms);
    }
    KOKKOS_INLINE_FUNCTION bool build_precond(int b, int N, int ndof, const double* xb,
                                              const double cell[9], const double hi[9],
                                              bool inv_ok, double* P) const {
        const int so = sopoff(b);
        sym_precond_pairs(xb, ndof, nop(b), layout(b), &M0all(so*9), &tcall(so*3),
                          &indepall(so), cell, hi, inv_ok, P);
        return precond_finish(P, N, ndof);
    }
};

// ===================================================================
// The L-BFGS step over raw pointers, shared by the device functor (DevStep) and
// the host batched relaxer (flexopt::LbfgsBase). Computing the DOF gradient is
// the caller's job.
// ===================================================================

// One structure's optimizer state. The device points these at slices of packed
// batch Views; the host at std::vector data. S and Y are [kHist * ndof] ring
// buffers and rho is [kHist], indexed by hh/hc.
struct LbfgsState {
    double *x, *grad, *gprev, *d, *xbase, *last, *q;
    double *S, *Y, *rho;
    const double* L = nullptr;   // model-Hessian Cholesky factor; null = scalar gamma only
    double* py = nullptr;        // [ndof] solve scratch, required iff L != nullptr
    int ndof = 0;                // nfree + nstrain
    int nfree = 0;               // coordinate DOFs
    int nstrain = 0;             // 6 for a relaxing cell, 0 for a fixed one
    int ndofatoms = 0;           // atoms carrying coordinate DOFs (nfree = 3 * this)
    double cellfac = 1.0;
    double *energy, *ebase, *gd, *maxg, *maxs, *gpm;
    int *active, *conv, *iters, *hh, *hc, *haveprev, *stallc;
};

struct LbfgsTuning {
    double ftol, stol, max_step, max_strain;
    int maxiter;
};

// Take a step from the point just evaluated: `s.grad` holds its DOF gradient and
// (e, maxF, ms) its energy and the two physical residuals. Tests convergence,
// pushes the curvature pair, runs the two-loop recursion through the
// preconditioner, caps the step and moves x to the first trial point.
//
// `cap` is the coordinate metric of the parameterization (FreeCap / SymCap).
template <class Cap>
KOKKOS_INLINE_FUNCTION void lbfgs_propose(const LbfgsState& s, const LbfgsTuning& t,
                                          const Cap& cap, double e, double maxF, double ms) {
    const int ndof = s.ndof, nf = s.nfree;
    *s.energy = e; *s.maxg = maxF; *s.maxs = ms;
    if (maxF < t.ftol && ms < t.stol) { *s.active = 0; *s.conv = 1; return; }
    // Stall detection (see kStallReset).
    const bool stalled = *s.haveprev && (maxF >= 0.99 * (*s.gpm));
    *s.stallc = stalled ? *s.stallc + 1 : 0;
    *s.gpm = maxF;
    const bool reset = *s.stallc >= kStallReset;
    if (reset) { *s.hc = 0; *s.hh = 0; *s.stallc = 0; }
    if (*s.haveprev && !reset) {
        double ys = 0;
        for (int k = 0; k < ndof; ++k) ys += (s.grad[k] - s.gprev[k]) * s.last[k];
        if (ys > 1e-12) {
            int pos;
            if (*s.hc < kHist) { pos = (*s.hh + *s.hc) % kHist; *s.hc += 1; }
            else { pos = *s.hh; *s.hh = (*s.hh + 1) % kHist; }
            const int sb = pos * ndof;
            for (int k = 0; k < ndof; ++k) {
                s.S[sb + k] = s.last[k];
                s.Y[sb + k] = s.grad[k] - s.gprev[k];
            }
            s.rho[pos] = 1.0 / ys;
        }
    }
    if (reset) {
        // Steepest descent, no history or preconditioner, capped like a normal step.
        for (int k = 0; k < ndof; ++k) s.d[k] = -s.grad[k];
    } else {
        const int cnt = *s.hc, head = *s.hh;
        for (int k = 0; k < ndof; ++k) s.q[k] = s.grad[k];
        double alpha[kHist];
        for (int ii = cnt - 1; ii >= 0; --ii) {
            const int sb = ((head + ii) % kHist) * ndof;
            double a = 0;
            for (int k = 0; k < ndof; ++k) a += s.S[sb + k] * s.q[k];
            a *= s.rho[(head + ii) % kHist]; alpha[ii] = a;
            for (int k = 0; k < ndof; ++k) s.q[k] -= a * s.Y[sb + k];
        }
        double gamma = 1.0;
        if (cnt > 0) {
            const int sb = ((head + cnt - 1) % kHist) * ndof;
            double ys = 0, yy = 0;
            for (int k = 0; k < ndof; ++k) { ys += s.Y[sb + k] * s.S[sb + k]; yy += s.Y[sb + k] * s.Y[sb + k]; }
            if (yy > 0) gamma = ys / yy;
        }
        if (s.L) chol_solve_dense(s.L, s.q, s.py, ndof);   // q <- Pn^-1 q
        for (int k = 0; k < ndof; ++k) s.q[k] *= gamma;
        for (int ii = 0; ii < cnt; ++ii) {
            const int sb = ((head + ii) % kHist) * ndof;
            double bb = 0;
            for (int k = 0; k < ndof; ++k) bb += s.Y[sb + k] * s.q[k];
            bb *= s.rho[(head + ii) % kHist];
            for (int k = 0; k < ndof; ++k) s.q[k] += (alpha[ii] - bb) * s.S[sb + k];
        }
        for (int k = 0; k < ndof; ++k) s.d[k] = -s.q[k];
    }
    const double sc = step_cap(cap(s.d, nf, s.ndofatoms), s.d, nf, s.nstrain,
                               s.cellfac, t.max_step, t.max_strain);
    for (int k = 0; k < ndof; ++k) s.xbase[k] = s.x[k];
    *s.ebase = e;
    double gdot = 0;
    for (int k = 0; k < ndof; ++k) {
        s.d[k] *= sc;
        gdot += s.grad[k] * s.d[k];
        s.x[k] = s.xbase[k] + s.d[k];
    }
    *s.gd = gdot;
    for (int k = 0; k < ndof; ++k) s.gprev[k] = s.grad[k];
    *s.haveprev = 1;
}

// Judge the trial point just evaluated: commit the step (and propose the next
// from here) or halve it and ask for another evaluation. (e, maxF, ms) describe
// the trial, not the accepted point.
template <class Cap>
KOKKOS_INLINE_FUNCTION void lbfgs_accept(const LbfgsState& s, const LbfgsTuning& t,
                                         const Cap& cap, double e, double maxF, double ms) {
    const int ndof = s.ndof;
    if (e <= *s.ebase + 1e-4 * (*s.gd)) {              // Armijo (gd < 0 by construction)
        for (int k = 0; k < ndof; ++k) s.last[k] = s.d[k];
        *s.iters += 1;
        if (*s.iters >= t.maxiter) {
            // Report the point we stop on, including its residuals.
            *s.energy = e; *s.maxg = maxF; *s.maxs = ms;
            if (maxF < t.ftol && ms < t.stol) *s.conv = 1;
            *s.active = 0;
            return;
        }
        lbfgs_propose(s, t, cap, e, maxF, ms);         // next direction from here
    } else {                                           // backtrack: halve and re-evaluate
        double nd = 0;
        for (int k = 0; k < ndof; ++k) {
            s.d[k] *= 0.5;
            s.x[k] = s.xbase[k] + s.d[k];
            nd += s.d[k] * s.d[k];
        }
        *s.gd *= 0.5;
        if (Kokkos::sqrt(nd) < 1e-9) {                 // stuck
            for (int k = 0; k < ndof; ++k) s.x[k] = s.xbase[k];
            *s.energy = *s.ebase;
            *s.active = 0;
        }
    }
}

// ===================================================================
// The trust-region step: klasp/lbfgs_tr.hpp's dogleg over a limited-memory BFGS
// model, one structure per team, its vectors across the lanes. B0 is gamma times
// the stretch-model preconditioner (mean diagonal one) or the identity.
//
// The radius shrinks only on rejection; at its floor the model is reset from the
// accepted point, a few times, after which the radius stays at the floor.
// Differences smaller than the energy's own precision are not read as evidence
// (lb_judge).
// ===================================================================

constexpr int kTrMem = 12;
// Vector lanes on one structure's step: a warp on NVIDIA, a wavefront on AMD.
#if defined(KOKKOS_ENABLE_HIP)
constexpr int kTrLanes = 64;
#else
constexpr int kTrLanes = 32;
#endif
using TrPolicy = Kokkos::TeamPolicy<ExecSpace>;
using TrMember = TrPolicy::member_type;

struct TrTuning {
    double ftol, stol;              // convergence: max force (Ha/Bohr), max stress (Ha/Bohr^3)
    double eta = 0.10, rho_max = 10.0;
    double delta0, delta_max, delta_min;  // Bohr of displacement
    double gamma0;                  // starting curvature scale, Ha/Bohr^2
    double precision;               // relative precision of the energy (float32: ~1.2e-7)
    int max_restarts = 3;
    int maxiter;                    // evaluations, accepted or not
};

// One structure's trust-region state: vectors as raw pointers into packed batch
// slices, scalars as values kept identical across lanes and written back once at
// the end of the round.
struct TrState {
    double *x, *xacc, *gacc, *grad;   // trial (what was just evaluated), accepted point and its gradient, this round's gradient
    double *p, *nw;                   // step and [n] scratch
    tr::Model<double> m;              // the curvature model (gamma, pairs, shape)
    int n = 0;
    double delta, eacc, maxg, maxs, pred;
    int restarts, iters, fresh, active, conv;
};

// One round for one structure, after its point x has been evaluated: grad,
// (e, maxF, ms) are that point's. Adopts it (a fresh structure), or judges it as
// the trial of the last step; then proposes the next trial into x, or finishes.
KOKKOS_INLINE_FUNCTION void tr_round(const TrMember& t, TrState& s, const TrTuning& tu, double e, double maxF,
                                     double ms) {
    const tr::Team<TrMember> c{t};
    const int n = s.n;
    auto adopt = [&]() {
        c.each(n, [&](int k) { s.xacc[k] = s.x[k]; s.gacc[k] = s.grad[k]; });
        s.eacc = e; s.maxg = maxF; s.maxs = ms;
    };
    auto finish = [&](bool converged) {
        c.each(n, [&](int k) { s.x[k] = s.xacc[k]; });
        s.conv = converged ? 1 : 0; s.active = 0;
    };
    if (s.fresh) {
        s.fresh = 0;
        adopt();
        s.delta = tu.delta0; s.restarts = 0;
        tr::lb_reset(s.m, tu.gamma0);
        if (maxF < tu.ftol && ms < tu.stol) { finish(true); return; }
    } else {
        s.iters += 1;
        const double noise = tu.precision * Kokkos::fmax(Kokkos::fabs(s.eacc), 1.0);
        double rho; bool accept, unusable;
        tr::lb_judge(e, s.eacc, s.pred, tu.eta, tu.rho_max, noise, rho, accept, unusable);
        // Both outcomes update the curvature model; bg still holds B p from the dogleg.
        if (Kokkos::isfinite(e)) tr::lb_update(c, s.m, s.p, s.grad, s.gacc, true);
        const double pn = Kokkos::sqrt(tr::lb_dot(c, s.m, s.p, s.p));
        if (accept) {
            adopt();
            if (rho > 0.75 && pn > 0.8 * s.delta) s.delta = Kokkos::fmin(2.0 * s.delta, tu.delta_max);
            if (maxF < tu.ftol && ms < tu.stol) { finish(true); return; }
        } else {
            const double shrunk = s.delta * (unusable ? 0.25 : 0.5);
            if (shrunk >= tu.delta_min) s.delta = shrunk;
            else if (s.restarts < tu.max_restarts) {
                s.delta = tu.delta0; s.restarts += 1;
                tr::lb_reset(s.m, tu.gamma0);
            } else s.delta = tu.delta_min;
        }
        if (s.iters >= tu.maxiter) { finish(false); return; }
    }
    s.pred = tr::lb_dogleg(c, s.m, s.gacc, s.delta, s.p, s.nw);
    if (s.pred <= 0 && Kokkos::sqrt(tr::lb_dot(c, s.m, s.p, s.p)) <= 1e-14) { finish(false); return; }  // no move left
    c.each(n, [&](int k) { s.x[k] = s.xacc[k] + s.p[k]; });
}

// ===================================================================
// Device-resident batched optimizer kernels, templated on the DOF policy. The
// injected evaluation deposits a DOF gradient plus an energy (DevStep::grad /
// r_energy); FoldKernel produces that from per-atom forces.
// ===================================================================

// Geometry staging views, active structures packed to the front; the layout of
// pet::DeviceGeom.
struct DeviceGeom {
    RView2D pos;    // [Ntot,3]
    IView1D sid;    // [Ntot] owning (compact) structure
    IView1D spec;   // [Ntot] species index
    IView1D soff;   // [Kc] atom offset per active structure
    IView1D scnt;   // [Kc] atom count per active structure
    IView1D sper;   // [Kc] periodic axes, bit d for lattice vector d; 0 for a molecule
    RView2D scell;  // [Kc,9] cell rows = lattice vectors (backend units)
    IView1D staged; // [Kc] the structure each active slot holds (its index in the batch)
    int Ntot = 0, B = 0;
};

// Evaluation seam for a backend that produces per-atom forces and a virial: one
// thread per active structure folds them onto the DOFs through the policy.
// Writes `grad` every round, including rejected trials; the accepted point's
// gradient lives in gprev.
template <class Dof>
struct FoldKernel {
    RView2D fF;                 // [Mc,3] per-atom forces, backend units
    RView2D fW;                 // [Kc,6] virial, backend units, full-tensor Voigt
    RView1D fE;                 // [Kc] energy, backend units
    RView2D defv;               // [B,9] deformation I+E, written by CellKernel this round
    RView1D volv, cf;           // [B] cell volume / cell_factor
    IView1D act, caoff, dofoff; // [Kc] / [Kc] / [B]
    RView1D grad;               // [D] out: packed DOF gradient
    RView1D r_energy, r_maxg, r_maxs;  // [B] out: this round's energy and residuals
    double f2hb, e2h;           // backend force/energy -> Ha/Bohr, Hartree
    Dof dof;

    KOKKOS_INLINE_FUNCTION void operator()(int kc) const {
        const int b = act(kc), cao = caoff(kc), off = dofoff(b);
        const int nf = 3 * dof.ndofatoms(b);
        double def[9];
        for (int e = 0; e < 9; ++e) def[e] = defv(b, e);
        double maxF = 0, ms = 0;
        dof.fold(b, nf, def, &fF(cao, 0), &fW(kc, 0), f2hb, e2h, cf(b), volv(b),
                 &grad(off), maxF, ms);
        r_energy(b) = fE(kc) * e2h;
        r_maxg(b) = maxF;
        r_maxs(b) = ms;
    }
};

// Per-structure L-BFGS step, one thread per structure. r_energy/r_maxg/r_maxs
// are this round's evaluation; energy/maxg/maxs describe the last accepted
// point and must not be overwritten by a rejected trial.
template <class Dof>
struct DevStep {
    using RV = RView1D;
    using RV2 = RView2D;
    using IV = IView1D;
    RV x, grad, gprev, d, xbase, last, q;
    RV Sd, Yd, rho;             // history (Sd/Yd length D*kHist; rho length B*kHist)
    RV gpm;                     // [B] previous max|grad| (stall detection)
    IV stallc;                  // [B] consecutive non-decreasing-grad step count
    RV2 defv;                   // [B,9] current deformation I+E
    RV volv;                    // [B] current cell volume
    RV energy, ebase, gd, maxg, maxs;  // [B]
    IV active, conv, iters, hh, hc, haveprev;  // [B]
    IV dofoff, histoff;         // [B] metadata
    IV act;                     // [Kc] compact slot -> original structure index
    RV cf;                      // [B] cell_factor
    RV Lmat, py;                // model-Hessian Cholesky factors + solve scratch
    IV poff, hasP;              // [B] per-struct factor offset / preconditioner valid
    Dof dof;                    // DOF policy (free / symmetric)
    RV r_energy, r_maxg, r_maxs;  // [B] this round's evaluation (the seam's output)
    double ftol, stol, max_step, max_strain;
    int maxiter;
    bool initial;

    // The per-structure state as raw pointers into the packed batch views.
    KOKKOS_INLINE_FUNCTION LbfgsState slot(int b) const {
        const int off = dofoff(b), ho = histoff(b);
        const int na = dof.ndofatoms(b), nf = 3 * na;
        LbfgsState s;
        s.x = &x(off); s.grad = &grad(off); s.gprev = &gprev(off); s.d = &d(off);
        s.xbase = &xbase(off); s.last = &last(off); s.q = &q(off);
        s.S = &Sd(ho); s.Y = &Yd(ho); s.rho = &rho(b * kHist);
        if (hasP(b)) { s.L = &Lmat(poff(b)); s.py = &py(off); }
        s.ndof = nf + 6; s.nfree = nf; s.nstrain = 6; s.ndofatoms = na;
        s.cellfac = cf(b);
        s.energy = &energy(b); s.ebase = &ebase(b); s.gd = &gd(b);
        s.maxg = &maxg(b); s.maxs = &maxs(b); s.gpm = &gpm(b);
        s.active = &active(b); s.conv = &conv(b); s.iters = &iters(b);
        s.hh = &hh(b); s.hc = &hc(b); s.haveprev = &haveprev(b); s.stallc = &stallc(b);
        return s;
    }

    KOKKOS_INLINE_FUNCTION LbfgsTuning tuning() const {
        return LbfgsTuning{ftol, stol, max_step, max_strain, maxiter};
    }

    KOKKOS_INLINE_FUNCTION void operator()(int kc) const {
        const int b = act(kc);
        if (!active(b)) return;
        const typename Dof::Cap cap{};
        if (initial) lbfgs_propose(slot(b), tuning(), cap, r_energy(b), r_maxg(b), r_maxs(b));
        else         lbfgs_accept(slot(b), tuning(), cap, r_energy(b), r_maxg(b), r_maxs(b));
    }
};

// Per-structure trust-region round (one thread per active structure). x is the
// point evaluated and comes back as the next trial, or, for a finished
// structure, as its accepted point. energy/maxg/maxs are the accepted point's;
// ebase mirrors energy for the round-budget exit.
template <class Dof>
struct TrDevStep {
    using RV = RView1D;
    using IV = IView1D;
    RV x, grad, xacc, gacc, p, q, nw, bg, tmp;
    RV S, Y, A, PS, ys, sa, alpha;          // pairs: [D*kTrMem] x4, [B*kTrMem] x3
    RV delta, energy, maxg, maxs, gamma, pred, ebase;   // [B]
    IV restarts, hc, hh, iters, fresh, active, conv;    // [B]
    IV dofoff, act;
    RV Lmat, Pmat;                          // shape factors, shapes
    IV poff, hasP;
    Dof dof;
    RV r_energy, r_maxg, r_maxs;            // [B] this round's evaluation
    TrTuning tune;

    KOKKOS_INLINE_FUNCTION void operator()(const TrMember& t) const {
        const int b = act(t.league_rank());
        if (!active(b)) return;
        const int off = dofoff(b), n = 3 * dof.ndofatoms(b) + 6;
        TrState s;
        s.x = &x(off); s.grad = &grad(off); s.xacc = &xacc(off); s.gacc = &gacc(off);
        s.p = &p(off); s.nw = &nw(off);
        auto& m = s.m;
        m.q = &q(off); m.bg = &bg(off); m.tmp = &tmp(off);
        m.S = &S(off * kTrMem); m.Y = &Y(off * kTrMem); m.A = &A(off * kTrMem); m.PS = &PS(off * kTrMem);
        m.ys = &ys(b * kTrMem); m.sa = &sa(b * kTrMem); m.alpha = &alpha(b * kTrMem);
        if (hasP(b)) { m.L = &Lmat(poff(b)); m.P = &Pmat(poff(b)); }
        m.n = s.n = n;
        m.mem = kTrMem;
        m.gamma = gamma(b); m.hc = hc(b); m.hh = hh(b);
        s.delta = delta(b); s.eacc = energy(b); s.maxg = maxg(b); s.maxs = maxs(b);
        s.pred = pred(b);
        s.restarts = restarts(b); s.iters = iters(b);
        s.fresh = fresh(b); s.active = active(b); s.conv = conv(b);
        tr_round(t, s, tune, r_energy(b), r_maxg(b), r_maxs(b));
        Kokkos::single(Kokkos::PerTeam(t), [&] {
            delta(b) = s.delta; energy(b) = s.eacc; ebase(b) = s.eacc; maxg(b) = s.maxg; maxs(b) = s.maxs;
            gamma(b) = s.m.gamma; pred(b) = s.pred;
            restarts(b) = s.restarts; hc(b) = s.m.hc; hh(b) = s.m.hh; iters(b) = s.iters;
            fresh(b) = s.fresh; active(b) = s.active; conv(b) = s.conv;
        });
    }
};

// The shape B0 is gamma times: P = L L^T from a preconditioner's Cholesky
// factor, once per structure, so the trust region's products are one pass.
struct TrShapeKernel {
    RView1D Lmat, Pmat;
    IView1D poff, hasP, dofoff, nfull;   // nfull(b): ndof
    KOKKOS_INLINE_FUNCTION void operator()(const TrMember& t) const {
        const int b = t.league_rank();
        if (!hasP(b)) return;
        const int n = nfull(b);
        const double* L = &Lmat(poff(b));
        double* P = &Pmat(poff(b));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, n * n), [&](int ij) {
            const int i = ij / n, j = ij % n, m = i < j ? i : j;
            double u = 0;
            for (int k = 0; k <= m; ++k) u += L[i * n + k] * L[j * n + k];
            P[ij] = u;
        });
    }
};

// Build the model-Hessian preconditioner factor for structure b (policy owns the
// stretch-model pair enumeration; this wraps cell inverse + dispatch).
template <class Dof>
struct PrecondKernel {
    RView1D x, Lmat;
    IView1D dofoff, poff, hasP;
    RView2D hbcell;
    Dof dof;
    KOKKOS_INLINE_FUNCTION void operator()(int b) const {
        const int N = dof.ndofatoms(b), off = dofoff(b), ndof = 3 * N + 6;
        if (poff(b) < 0) return;  // exceeds the DOF cap -> scalar fallback
        double* P = &Lmat(poff(b));
        for (int t = 0; t < ndof * ndof; ++t) P[t] = 0.0;
        double hi[9];
        const double cell[9] = {hbcell(b,0),hbcell(b,1),hbcell(b,2),hbcell(b,3),
                                hbcell(b,4),hbcell(b,5),hbcell(b,6),hbcell(b,7),hbcell(b,8)};
        const bool inv_ok = crystdof::mat3_inv(cell, hi);
        hasP(b) = dof.build_precond(b, N, ndof, &x(off), cell, hi, inv_ok, P) ? 1 : 0;
    }
};

// Per-active-structure: deformation I+E, cell rows (scaled by gunit -> backend
// length units), volume. nf = 3 * (#DOF atoms).
template <class Dof>
struct CellKernel {
    IView1D actv, dofoff, caoffv, Nb, per;   // per: [B] as DeviceGeom::sper
    RView1D x, cf, volv;
    RView2D defv, h0;       // h0: reference cell rows (backend units)
    DeviceGeom geom;
    double volunit;              // converts the cell determinant to the optimizer's volume unit
    Dof dof;
    KOKKOS_INLINE_FUNCTION void operator()(int kc) const {
        const int b = actv(kc), off = dofoff(b), nf = 3 * dof.ndofatoms(b);
        double eps[6];
        for (int k = 0; k < 6; ++k) eps[k] = x(off + nf + k) / cf(b);
        double def[9]; crystdof::strain_to_def(eps, def);
        for (int e = 0; e < 9; ++e) defv(b, e) = def[e];
        double cell[9];
        for (int r = 0; r < 3; ++r) {            // scell row r = def * (h0 row r)
            const double v[3] = {h0(b, r * 3), h0(b, r * 3 + 1), h0(b, r * 3 + 2)};
            double rv[3]; crystdof::mat3_vec(def, v, rv);
            for (int c = 0; c < 3; ++c) { cell[r * 3 + c] = rv[c]; geom.scell(kc, r * 3 + c) = rv[c]; }
        }
        volv(b) = Kokkos::fabs(crystdof::mat3_det(cell)) / volunit;
        geom.sper(kc) = per(b);
        geom.soff(kc) = caoffv(kc); geom.scnt(kc) = Nb(b);
        if (geom.staged.extent(0)) geom.staged(kc) = b;
    }
};

// Per-active-atom: full-cell Cartesian (Bohr) -> backend length units via the
// policy expand (free: r = def*p; symmetric: r = def*(M0_op q_a + tc_op)).
template <class Dof>
struct PosKernel {
    IView1D csid, actv, dofoff, caoffv, atomoff, spec_full;
    RView1D x;
    RView2D defv;
    DeviceGeom geom;
    double gunit;  // Bohr -> backend length unit (Angstrom/Bohr for PET, 1 for a.u.)
    Dof dof;
    KOKKOS_INLINE_FUNCTION void operator()(int gc) const {
        const int kc = csid(gc), b = actv(kc), off = dofoff(b), il = gc - caoffv(kc);
        double def[9]; for (int e = 0; e < 9; ++e) def[e] = defv(b, e);
        double r[3]; dof.expand(b, il, def, &x(off), r);
        for (int c = 0; c < 3; ++c) geom.pos(gc, c) = r[c] * gunit;
        geom.sid(gc) = kc;
        geom.spec(gc) = spec_full(atomoff(b) + il);
    }
};

}  // namespace devopt
}  // namespace klasp
