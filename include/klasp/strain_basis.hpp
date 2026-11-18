#pragma once
// Symmetry-adapted cell strain degrees of freedom. The optimiser's cell
// variables are amplitudes q along a basis of allowed Voigt strain directions:
//
//     eps_v = sum_s q_s * basis[s][v]        (basis_expand)
//     q_s   = sum_v basis[s][v] * eps_v      (basis_project, rows orthonormal)
//     dE/dq_s = sum_v basis[s][v] * dE/deps_v
//
// The basis spans the strains every rotation of the point group leaves
// unchanged (e.g. eps_xx = eps_yy under a 4-fold about z), so the metric keeps
// the symmetry of the operations.
//
// Voigt order matches batch_shared.cpp: 0,1,2 = xx,yy,zz; 3,4,5 = yz,xz,xy,
// with the engineering factor of 2 on the shear components.

#include <Kokkos_Core.hpp>
#include "klasp/mat3.hpp"
#include "klasp/types.hpp"

#include <array>
#include <vector>

namespace klasp {

// Absolute bounds on the linearised strain, keeping (I + eps) a small
// deformation and the cell angles away from degeneracy.
constexpr RealType STRAIN_MAX_DIAG  = 0.4;
constexpr RealType STRAIN_MAX_SHEAR = 0.3;

KOKKOS_INLINE_FUNCTION
RealType strain_bound(int voigt) { return voigt < 3 ? STRAIN_MAX_DIAG : STRAIN_MAX_SHEAR; }

// eps (6) from amplitudes q (ndof). `basis` is 6x6 row-major; only the first
// ndof rows are read.
KOKKOS_INLINE_FUNCTION
void basis_expand(const RealType* basis, const RealType* q, int ndof, RealType* eps) {
    for (int v = 0; v < 6; ++v) eps[v] = 0;
    for (int s = 0; s < ndof; ++s) {
        const RealType qs = q[s];
        for (int v = 0; v < 6; ++v) eps[v] += qs * basis[s*6 + v];
    }
}

// Amplitudes q (ndof) from a 6-vector: q_s = basis[s] . x. Also the gradient
// contraction dE/dq = B dE/deps.
KOKKOS_INLINE_FUNCTION
void basis_project(const RealType* basis, const RealType* x, int ndof, RealType* q) {
    for (int s = 0; s < ndof; ++s) {
        RealType acc = 0;
        for (int v = 0; v < 6; ++v) acc += basis[s*6 + v] * x[v];
        q[s] = acc;
    }
}

// Largest t in [0, 1] with eps(q0 + t*dq) inside the strain bounds. The whole
// step is scaled, never clamped per component, so it stays in the symmetry
// subspace and the predicted reduction and secant remain consistent.
KOKKOS_INLINE_FUNCTION
RealType strain_step_fraction(const RealType* basis, const RealType* q0,
                              const RealType* dq, int ndof) {
    RealType e0[6], de[6];
    basis_expand(basis, q0, ndof, e0);
    basis_expand(basis, dq, ndof, de);
    RealType t = 1;
    for (int v = 0; v < 6; ++v) {
        const RealType lim = strain_bound(v);
        const RealType e1 = e0[v] + de[v];
        if (e1 > lim && de[v] > RealType(1e-30)) {
            const RealType tv = (lim - e0[v]) / de[v];
            if (tv < t) t = tv;
        } else if (e1 < -lim && de[v] < RealType(-1e-30)) {
            const RealType tv = (-lim - e0[v]) / de[v];
            if (tv < t) t = tv;
        }
    }
    return t < RealType(0) ? RealType(0) : t;
}

namespace strain_detail {

KOKKOS_INLINE_FUNCTION
constexpr RealType machine_eps() {
    return sizeof(RealType) == 4 ? RealType(1.1920929e-7) : RealType(2.220446049250313e-16);
}

KOKKOS_INLINE_FUNCTION
RealType dot6(const RealType* a, const RealType* b) {
    RealType s = 0;
    for (int v = 0; v < 6; ++v) s += a[v] * b[v];
    return s;
}

// Removes from v its components along the orthonormal rows[0..k), twice for
// stability, and returns the residual norm.
KOKKOS_INLINE_FUNCTION
RealType orthogonalise(const RealType* rows, int k, RealType* v) {
    for (int pass = 0; pass < 2; ++pass)
        for (int r = 0; r < k; ++r) {
            const RealType d = dot6(rows + r*6, v);
            for (int i = 0; i < 6; ++i) v[i] -= d * rows[r*6 + i];
        }
    return Kokkos::sqrt(dot6(v, v));
}

// Appends to rows[count..] the candidates (m x 6) that are independent of the
// rows so far, in order, keeping those whose residual exceeds tol. If that
// leaves fewer than `target` rows, adds the largest remaining residual until it
// does. Returns the new row count.
KOKKOS_INLINE_FUNCTION
int select_rows(RealType* rows, int count, int target, const RealType* cands, int m,
                RealType tol) {
    for (int c = 0; c < m; ++c) {
        RealType v[6];
        for (int i = 0; i < 6; ++i) v[i] = cands[c*6 + i];
        const RealType n = orthogonalise(rows, count, v);
        if (n > tol) {
            for (int i = 0; i < 6; ++i) rows[count*6 + i] = v[i] / n;
            ++count;
        }
    }
    while (count < target) {
        RealType best[6] = {}, best_n = 0;
        for (int c = 0; c < m; ++c) {
            RealType v[6];
            for (int i = 0; i < 6; ++i) v[i] = cands[c*6 + i];
            const RealType n = orthogonalise(rows, count, v);
            if (n > best_n) {
                best_n = n;
                for (int i = 0; i < 6; ++i) best[i] = v[i];
            }
        }
        if (!(best_n > RealType(0))) break;
        for (int i = 0; i < 6; ++i) rows[count*6 + i] = best[i] / best_n;
        ++count;
    }
    return count;
}

// Zeroes round-off, sets a row whose non-zero entries share one magnitude to
// exactly +-1/sqrt(m), and makes the first non-zero entry positive.
KOKKOS_INLINE_FUNCTION
void snap_row(RealType* row) {
    const RealType zero_tol = RealType(100) * machine_eps();
    const RealType equal_tol = RealType(1e4) * machine_eps();
    int m = 0;
    RealType ref = 0;
    bool equal = true;
    for (int v = 0; v < 6; ++v) {
        if (Kokkos::fabs(row[v]) < zero_tol) { row[v] = 0; continue; }
        const RealType a = Kokkos::fabs(row[v]);
        if (m == 0) ref = a;
        else if (Kokkos::fabs(a - ref) > equal_tol) equal = false;
        ++m;
    }
    if (m > 0 && equal) {
        const RealType mag = m == 1 ? RealType(1)
                           : m == 2 ? RealType(1) / Kokkos::sqrt(RealType(2))
                           : m == 3 ? RealType(1) / Kokkos::sqrt(RealType(3))
                                    : RealType(1) / Kokkos::sqrt(RealType(m));
        for (int v = 0; v < 6; ++v)
            if (row[v] != RealType(0)) row[v] = row[v] > 0 ? mag : -mag;
    }
    for (int v = 0; v < 6; ++v)
        if (row[v] != RealType(0)) {
            if (row[v] < 0)
                for (int w = 0; w < 6; ++w) row[w] = -row[w];
            break;
        }
}

}  // namespace strain_detail

// Allowed strain directions from the point group, device-callable.
//
// `cell` is 3x3 row-major with the lattice vectors as columns. `W` holds the n
// distinct fractional rotation parts (n x 9, row-major) of a closed point
// group; `dim_S` is the dimension of its symmetric-strain space (see
// PointGroup::strain_dim). A Voigt strain eps is allowed when R eps R^T = eps
// for every Cartesian rotation R = C W C^-1.
//
// Rows come lengths first: the allowed directions with no shear (Gram-Schmidt
// of xx, yy, zz projected onto them), then the rest. `diag_only` returns only
// the first group. Writes `basis` (6x6, unused rows zero) and returns the row
// count, or -1 if the cell is singular or the subspace found does not have
// dimension dim_S. `orth_err`, when given, receives max |R^T R - I| before
// clean-up: large values mean the cell's metric does not fit the operations.
KOKKOS_INLINE_FUNCTION
int build_symmetric_strain_basis(const RealType* cell, const RealType* W, int n, int dim_S,
                                 bool diag_only, RealType* basis,
                                 RealType* orth_err = nullptr) {
    using namespace strain_detail;
    for (int i = 0; i < 36; ++i) basis[i] = 0;
    if (orth_err) *orth_err = 0;
    RealType cinv[9];
    if (n < 1 || !mat3_inv(cell, cinv)) return -1;

    // P = (1/n) sum_k M_k, with M_k the Voigt matrix of eps -> R_k eps R_k^T.
    RealType P[36] = {};
    RealType worst = 0;
    for (int k = 0; k < n; ++k) {
        RealType tmp[9], R[9];
        mat3_mul(W + k*9, cinv, tmp);
        mat3_mul(cell, tmp, R);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                RealType s = 0;
                for (int l = 0; l < 3; ++l) s += R[l*3 + i] * R[l*3 + j];
                const RealType e = Kokkos::fabs(s - (i == j ? RealType(1) : RealType(0)));
                if (e > worst) worst = e;
            }
        for (int it = 0; it < 2; ++it) {                 // R <- (R + R^-T) / 2
            RealType Ri[9];
            if (!mat3_inv(R, Ri)) return -1;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R[i*3 + j] = RealType(0.5) * (R[i*3 + j] + Ri[j*3 + i]);
        }
        for (int v = 0; v < 6; ++v) {
            RealType e[9] = {};
            if (v < 3) e[v*3 + v] = 1;
            else {
                const int p = v == 3 ? 1 : 0, q = v == 5 ? 1 : 2;
                e[p*3 + q] = e[q*3 + p] = RealType(0.5);
            }
            RealType re[9], t[9];                         // t = R e R^T
            mat3_mul(R, e, re);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    RealType s = 0;
                    for (int l = 0; l < 3; ++l) s += re[i*3 + l] * R[j*3 + l];
                    t[i*3 + j] = s;
                }
            const RealType out[6] = {t[0], t[4], t[8], 2*t[5], 2*t[2], 2*t[1]};
            for (int w = 0; w < 6; ++w) P[w*6 + v] += out[w] / RealType(n);
        }
    }
    if (orth_err) *orth_err = worst;

    // S: Gram-Schmidt over the columns P e_0..P e_5.
    RealType cols[36], scale = 0;
    for (int v = 0; v < 6; ++v) {
        for (int w = 0; w < 6; ++w) cols[v*6 + w] = P[w*6 + v];
        const RealType nv = Kokkos::sqrt(dot6(cols + v*6, cols + v*6));
        if (nv > scale) scale = nv;
    }
    const RealType tol = RealType(1e-3);
    RealType Q[36] = {};
    if (select_rows(Q, 0, dim_S, cols, 6, tol * scale) != dim_S) return -1;

    // D = S with no shear: c in R^dim_S with sum_i c_i Q_i[3..5] = 0. Its
    // complement in coefficient space is spanned by the shear columns of Q.
    RealType shear[18] = {}, coef[36] = {};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < dim_S; ++i) shear[j*6 + i] = Q[i*6 + 3 + j];
    const int rank = select_rows(coef, 0, 0, shear, 3, tol);
    RealType unit[36] = {};
    for (int i = 0; i < dim_S; ++i) unit[i*6 + i] = 1;
    const int dim_D = dim_S - rank;
    if (select_rows(coef, rank, dim_S, unit, dim_S, tol) != dim_S) return -1;
    RealType Dr[36] = {};
    for (int d = 0; d < dim_D; ++d)
        for (int i = 0; i < dim_S; ++i)
            for (int w = 0; w < 6; ++w) Dr[d*6 + w] += coef[(rank + d)*6 + i] * Q[i*6 + w];

    // Tier 1: e_0..e_2 projected onto D.
    RealType cand[36] = {};
    for (int k = 0; k < 3; ++k)
        for (int d = 0; d < dim_D; ++d)
            for (int w = 0; w < 6; ++w) cand[k*6 + w] += Dr[d*6 + k] * Dr[d*6 + w];
    int count = select_rows(basis, 0, dim_D, cand, 3, tol);
    if (count != dim_D) return -1;

    // Tier 2: e_0..e_5 projected onto S, orthogonal to tier 1.
    if (!diag_only) {
        for (int i = 0; i < 36; ++i) cand[i] = 0;
        for (int k = 0; k < 6; ++k)
            for (int i = 0; i < dim_S; ++i)
                for (int w = 0; w < 6; ++w) cand[k*6 + w] += Q[i*6 + k] * Q[i*6 + w];
        count = select_rows(basis, count, dim_S, cand, 6, tol);
        if (count != dim_S) return -1;
    }
    for (int r = 0; r < count; ++r) snap_row(basis + r*6);
    return count;
}

// ---------------------------------------------------------------------------
// Host-side basis construction (src/strain_basis.cpp).
// ---------------------------------------------------------------------------

// The distinct rotation parts of a set of operations, closed under
// multiplication, and the dimension of the symmetric strains they leave
// unchanged.
struct PointGroup {
    std::vector<std::array<int, 9>> rotations;   // fractional, row-major
    int strain_dim = 0;
};

// From n fractional rotations (n x 9, row-major). Throws if a rotation is not
// an integer matrix or the closure exceeds 48 elements.
PointGroup point_group_of(const RealType* rotations, int n);

// build_symmetric_strain_basis for one cell on the host, warning when the
// cell's metric does not fit the operations.
int symmetric_strain_basis(const PointGroup& pg, const RealType cell[9], bool diag_only,
                           RealType basis[36]);

}  // namespace klasp
