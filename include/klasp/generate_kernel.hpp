#pragma once
// GPU-resident structure generation kernel.

#include "klasp/types.hpp"
#include "klasp/kgf.hpp"
#include "klasp/rotation.hpp"
#include "klasp/sgdata.hpp"

namespace klasp {

// Map [0,1) to angle in [amin, amax] via arcsin
KOKKOS_INLINE_FUNCTION
RealType gpu_map_angle(RealType v, RealType amin, RealType amax) {
    RealType delta = amax - amin;
    RealType a = Kokkos::asin(2.0 * v - 1.0) / M_PI;
    return (0.5 + a) * delta + amin;
}

// Build upper-triangular cell matrix from parameters
KOKKOS_INLINE_FUNCTION
void gpu_build_cell(RealType a, RealType b, RealType c,
                    RealType ca, RealType cb, RealType cg, RealType sg, RealType sqrt_x,
                    RealType cell[3][3], RealType cell_inv[3][3]) {
    RealType v = a * b * c * sqrt_x;
    cell[0][0] = a;    cell[0][1] = b*cg;         cell[0][2] = c*cb;
    cell[1][0] = 0;    cell[1][1] = b*sg;          cell[1][2] = c*(ca - cb*cg)/sg;
    cell[2][0] = 0;    cell[2][1] = 0;             cell[2][2] = v / (a * b * sg);

    RealType det = cell[0][0] * cell[1][1] * cell[2][2];
    cell_inv[0][0] = cell[1][1]*cell[2][2] / det;
    cell_inv[0][1] = -cell[0][1]*cell[2][2] / det;
    cell_inv[0][2] = (cell[0][1]*cell[1][2] - cell[0][2]*cell[1][1]) / det;
    cell_inv[1][0] = 0;
    cell_inv[1][1] = cell[0][0]*cell[2][2] / det;
    cell_inv[1][2] = -cell[0][0]*cell[1][2] / det;
    cell_inv[2][0] = 0;
    cell_inv[2][1] = 0;
    cell_inv[2][2] = cell[0][0]*cell[1][1] / det;
}

// Build supercell positions from rotated body + COM + symops (all in Angstrom).
// `mol0_frac_buf` is caller-owned scratch of at least `n_sites * 3` doubles.
KOKKOS_INLINE_FUNCTION
void gpu_build_supercell(const RealType* rotated, int n_sites,
                          const RealType* trans, const RealType cell[3][3],
                          const RealType cell_inv[3][3],
                          const RealType* sym_rot, const RealType* sym_trans,
                          int n_symops, RealType* all_cart,
                          RealType* mol0_frac,
                          const RealType* nudge = nullptr) {
    RealType com[3] = {0, 0, 0};
    for (int d = 0; d < 3; ++d)
        for (int k = 0; k < 3; ++k)
            com[d] += trans[k] * cell[d][k];

    for (int i = 0; i < n_sites; ++i) {
        RealType cart[3];
        for (int d = 0; d < 3; ++d) {
            cart[d] = rotated[i*3+d] + com[d];
            if (nudge) cart[d] += nudge[d];
        }
        for (int d = 0; d < 3; ++d) {
            mol0_frac[i*3+d] = 0;
            for (int k = 0; k < 3; ++k)
                mol0_frac[i*3+d] += cell_inv[d][k] * cart[k];
        }
    }

    for (int op = 0; op < n_symops; ++op) {
        for (int i = 0; i < n_sites; ++i) {
            RealType frac_new[3] = {0, 0, 0};
            for (int d = 0; d < 3; ++d) {
                for (int k = 0; k < 3; ++k)
                    frac_new[d] += sym_rot[op*9 + d*3 + k] * mol0_frac[i*3+k];
                frac_new[d] += sym_trans[op*3 + d];
            }
            int atom = op * n_sites + i;
            for (int d = 0; d < 3; ++d) {
                all_cart[atom*3+d] = 0;
                for (int k = 0; k < 3; ++k)
                    all_cart[atom*3+d] += frac_new[k] * cell[d][k];
            }
        }
    }
}

}  // namespace klasp
