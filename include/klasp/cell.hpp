#pragma once

#include "klasp/mat3.hpp"
#include "klasp/types.hpp"

#include <cmath>

namespace klasp {

// Convert [a, b, c, alpha, beta, gamma] (Angstrom, degrees) to 3x3 cell matrix.
// a along x, b in xy-plane. out[row][col], columns = lattice vectors.
void cell_params_to_matrix(const RealType params[6], RealType out[3][3]);

// Cell volume = |a . (b x c)|.
RealType cell_volume(const RealType cell[3][3]);

// The helpers below are templated on the scalar because RealType may be float
// while the PET path is double throughout.

// Cell matrix (columns = lattice vectors) -> [a, b, c, alpha, beta, gamma], with
// lengths in the matrix's units and angles in degrees. Frame invariant.
template <typename T>
inline void matrix_to_cell_params(const T cell[3][3], T params[6]) {
    auto col = [&](int j, T v[3]) { v[0] = cell[0][j]; v[1] = cell[1][j]; v[2] = cell[2][j]; };
    auto nrm = [](const T v[3]) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); };
    auto dot = [](const T* u, const T* v) { return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]; };
    auto ang = [](T c) {
        return std::acos(c < T(-1) ? T(-1) : (c > T(1) ? T(1) : c)) * T(180) / T(M_PI);
    };
    T a[3], b[3], c[3];
    col(0, a); col(1, b); col(2, c);
    const T la = nrm(a), lb = nrm(b), lc = nrm(c);
    params[0] = la; params[1] = lb; params[2] = lc;
    params[3] = ang(dot(b, c) / (lb * lc));   // alpha
    params[4] = ang(dot(a, c) / (la * lc));   // beta
    params[5] = ang(dot(a, b) / (la * lb));   // gamma
}

// Rotate `n_indep` angle-axis orientations (in place) from the frame of `cell`
// into the canonical frame cell_params_to_matrix builds from `cell_params`.
// Returns false, changing nothing, for a singular cell.
bool align_orientations_to_canonical_cell(const RealType cell[3][3],
                                          const RealType cell_params[6],
                                          RealType* rot_vectors, int n_indep);

// Cell volume from [a, b, c, alpha, beta, gamma] (degrees), in the cube of the
// length unit. The radicand is clamped at zero so a degenerate cell gives 0, not NaN.
template <typename T>
inline T cell_volume_from_params(const T p[6]) {
    const T ca = std::cos(p[3] * T(M_PI) / T(180));
    const T cb = std::cos(p[4] * T(M_PI) / T(180));
    const T cg = std::cos(p[5] * T(M_PI) / T(180));
    const T rad = T(1) - ca*ca - cb*cb - cg*cg + T(2)*ca*cb*cg;
    return p[0] * p[1] * p[2] * std::sqrt(rad > T(0) ? rad : T(0));
}

// Fill workspace shifts + shift_cart from cell geometry and cutoff.
// Uses interplanar spacings for image range. Returns n_shifts written.
int fill_shifts(const RealType cell[3][3], RealType cutoff, Workspace& ws);

// Reciprocal-lattice index bound covering every g with |g| < gmax, for a cell
// whose columns are the lattice vectors: |h| < gmax |a_1| / 2 pi. The bound uses
// the lattice vector length, not the interplanar spacing, which truncates the
// sum on oblique cells.
template <typename T>
KOKKOS_INLINE_FUNCTION
void reciprocal_index_bounds(const T cell[3][3], T gmax, int out[3]) {
    const T scale = gmax / T(2 * M_PI);
    for (int i = 0; i < 3; ++i) {
        const T x = cell[0][i], y = cell[1][i], z = cell[2][i];
        const T len = Kokkos::sqrt(x * x + y * y + z * z);
        out[i] = static_cast<int>(Kokkos::ceil(scale * len)) + 1;
    }
}

}  // namespace klasp
