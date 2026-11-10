#pragma once

#include "klasp/types.hpp"

namespace klasp {

// Device-side L-BFGS math on raw pointers, for use inside a per-structure kernel.

KOKKOS_INLINE_FUNCTION
RealType vec_dot(const RealType* a, const RealType* b, int n) {
    RealType s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

KOKKOS_INLINE_FUNCTION
RealType vec_norm_sq(const RealType* v, int n) {
    return vec_dot(v, v, n);
}

KOKKOS_INLINE_FUNCTION
void vec_axpy(RealType alpha, const RealType* x, RealType* y, int n) {
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

KOKKOS_INLINE_FUNCTION
void vec_copy(const RealType* src, RealType* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

KOKKOS_INLINE_FUNCTION
void vec_scale(RealType alpha, RealType* v, int n) {
    for (int i = 0; i < n; ++i) v[i] *= alpha;
}

// L-BFGS two-loop recursion: d = -H_k g with H_k the L-BFGS inverse Hessian.
//
// s_hist, y_hist: (memory, n_dof) stored as row-major (s_hist[k*n + i])
// rho:            (memory,) = 1 / (y_k^T s_k)
// alpha_scratch:  (memory,) workspace
// g:              (n_dof,) current gradient
// d:              (n_dof,) output search direction
// num_stored:     number of valid correction pairs
// memory:         max correction pairs
// n:              number of DOFs
KOKKOS_INLINE_FUNCTION
void lbfgs_direction(const RealType* s_hist, const RealType* y_hist,
                     const RealType* rho, RealType* alpha_scratch,
                     const RealType* g, RealType* d, int num_stored,
                     int memory, int n)
{
    // d holds q until the final negation.
    vec_copy(g, d, n);

    if (num_stored == 0) {
        vec_scale(-1.0, d, n);
        return;
    }

    for (int i = num_stored - 1; i >= 0; --i) {
        const RealType* si = s_hist + i * n;
        const RealType* yi = y_hist + i * n;
        alpha_scratch[i] = rho[i] * vec_dot(si, d, n);
        vec_axpy(-alpha_scratch[i], yi, d, n);
    }

    // gamma = s_k^T y_k / (y_k^T y_k) for most recent pair
    int last = num_stored - 1;
    const RealType* s_last = s_hist + last * n;
    const RealType* y_last = y_hist + last * n;
    RealType gamma = vec_dot(s_last, y_last, n) / vec_norm_sq(y_last, n);

    vec_scale(gamma, d, n);

    for (int i = 0; i < num_stored; ++i) {
        const RealType* si = s_hist + i * n;
        const RealType* yi = y_hist + i * n;
        RealType beta = rho[i] * vec_dot(yi, d, n);
        vec_axpy(alpha_scratch[i] - beta, si, d, n);
    }

    vec_scale(-1.0, d, n);
}

// Push a correction pair (s, y) into the ring buffer at *cursor (advanced in
// place). Skips the pair if y.s <= 1e-15. Returns the new num_stored.
KOKKOS_INLINE_FUNCTION
int lbfgs_push(RealType* s_hist, RealType* y_hist, RealType* rho,
               const RealType* s, const RealType* y,
               int num_stored, int* cursor, int memory, int n)
{
    RealType ys = vec_dot(y, s, n);
    if (ys <= 1e-15) return num_stored;

    int pos = *cursor;
    vec_copy(s, s_hist + pos * n, n);
    vec_copy(y, y_hist + pos * n, n);
    rho[pos] = 1.0 / ys;

    *cursor = (pos + 1) % memory;
    return (num_stored < memory) ? num_stored + 1 : memory;
}

}  // namespace klasp
