#pragma once
// 3x3 matrix primitives, device-callable, for both storage layouts in use:
// T[3][3] and row-major T[9].
//
// mat3_inv returns false and leaves `out` untouched when the determinant is
// exactly zero; the caller decides what a degenerate matrix means. A NaN
// determinant yields a NaN inverse.
//
// mat3_vec and mat3_mul use different expression forms per layout, which round
// differently under FMA; keep them distinct so existing energies stay bit-identical.

#include <Kokkos_Core.hpp>

namespace klasp {

// out = R v.
template <typename T>
KOKKOS_INLINE_FUNCTION
void mat3_vec(const T R[3][3], const T v[3], T out[3]) {
    for (int i = 0; i < 3; ++i) {
        out[i] = T(0);
        for (int j = 0; j < 3; ++j)
            out[i] += R[i][j] * v[j];
    }
}

// y = A x, A row-major 3x3.
template <typename T>
KOKKOS_INLINE_FUNCTION
void mat3_vec(const T A[9], const T x[3], T y[3]) {
    for (int i = 0; i < 3; ++i)
        y[i] = A[i * 3] * x[0] + A[i * 3 + 1] * x[1] + A[i * 3 + 2] * x[2];
}

// y = A^T x, A row-major 3x3.
template <typename T>
KOKKOS_INLINE_FUNCTION
void mat3T_vec(const T A[9], const T x[3], T y[3]) {
    for (int i = 0; i < 3; ++i)
        y[i] = A[0 * 3 + i] * x[0] + A[1 * 3 + i] * x[1] + A[2 * 3 + i] * x[2];
}

// C = A B. C must not alias A or B.
template <typename T>
KOKKOS_INLINE_FUNCTION
void mat3_mul(const T A[3][3], const T B[3][3], T C[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            C[i][j] = T(0);
            for (int k = 0; k < 3; ++k)
                C[i][j] += A[i][k] * B[k][j];
        }
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void mat3_mul(const T A[9], const T B[9], T C[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T s = T(0);
            for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = s;
        }
}

// out = A^T. out must not alias A.
template <typename T>
KOKKOS_INLINE_FUNCTION
void mat3_transpose(const T A[3][3], T out[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i][j] = A[j][i];
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void mat3_transpose(const T A[9], T out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i * 3 + j] = A[j * 3 + i];
}

// det M, expanded along the first row.
template <typename T>
KOKKOS_INLINE_FUNCTION
T mat3_det(const T M[3][3]) {
    return M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1])
         - M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0])
         + M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0]);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
T mat3_det(const T m[9]) {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// out = M^{-1}. Returns false and leaves `out` untouched if det M == 0.
template <typename T>
KOKKOS_INLINE_FUNCTION
bool mat3_inv(const T M[3][3], T out[3][3]) {
    const T det = mat3_det(M);
    if (det == T(0)) return false;
    const T id = T(1) / det;
    out[0][0] = (M[1][1]*M[2][2] - M[1][2]*M[2][1]) * id;
    out[0][1] = (M[0][2]*M[2][1] - M[0][1]*M[2][2]) * id;
    out[0][2] = (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * id;
    out[1][0] = (M[1][2]*M[2][0] - M[1][0]*M[2][2]) * id;
    out[1][1] = (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * id;
    out[1][2] = (M[0][2]*M[1][0] - M[0][0]*M[1][2]) * id;
    out[2][0] = (M[1][0]*M[2][1] - M[1][1]*M[2][0]) * id;
    out[2][1] = (M[0][1]*M[2][0] - M[0][0]*M[2][1]) * id;
    out[2][2] = (M[0][0]*M[1][1] - M[0][1]*M[1][0]) * id;
    return true;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
bool mat3_inv(const T m[9], T inv[9]) {
    const T d = mat3_det(m);
    if (d == T(0)) return false;
    const T id = T(1) / d;
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * id;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * id;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * id;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * id;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * id;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * id;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * id;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * id;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * id;
    return true;
}

}  // namespace klasp
