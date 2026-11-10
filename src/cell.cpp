#include "klasp/cell.hpp"
#include "klasp/rotation.hpp"
#include <cmath>
#include <algorithm>

namespace klasp {

bool align_orientations_to_canonical_cell(const RealType cell[3][3],
                                          const RealType cell_params[6],
                                          RealType* rot_vectors, int n_indep) {
    RealType canon[3][3];
    cell_params_to_matrix(cell_params, canon);
    RealType inv[3][3];
    if (!mat3_inv(cell, inv)) return false;
    RealType align[3][3];
    mat3_mul(canon, inv, align);
    for (int k = 0; k < n_indep; ++k) {
        RealType* aa = rot_vectors + 3 * k;
        RealType before[3][3], after[3][3];
        angle_axis_to_matrix(aa, before);
        mat3_mul(align, before, after);
        rotation_matrix_to_angle_axis(after, aa);
    }
    return true;
}

void cell_params_to_matrix(const RealType params[6], RealType out[3][3]) {
    const RealType a = params[0], b = params[1], c = params[2];
    const RealType alpha_deg = params[3], beta_deg = params[4], gamma_deg = params[5];

    constexpr RealType deg2rad = M_PI / 180.0;
    const RealType cos_alpha = std::cos(alpha_deg * deg2rad);
    const RealType cos_beta  = std::cos(beta_deg  * deg2rad);
    const RealType cos_gamma = std::cos(gamma_deg * deg2rad);
    const RealType sin_gamma = std::sin(gamma_deg * deg2rad);

    const RealType ax = a;
    const RealType bx = b * cos_gamma;
    const RealType by = b * sin_gamma;
    const RealType cx = c * cos_beta;
    const RealType cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    const RealType cz = std::sqrt(c * c - cx * cx - cy * cy);

    out[0][0] = ax;  out[0][1] = bx;  out[0][2] = cx;
    out[1][0] = 0.0; out[1][1] = by;  out[1][2] = cy;
    out[2][0] = 0.0; out[2][1] = 0.0; out[2][2] = cz;
}

RealType cell_volume(const RealType cell[3][3]) {
    // Columns are lattice vectors: a = col0, b = col1, c = col2
    const RealType ax = cell[0][0], ay = cell[1][0], az = cell[2][0];
    const RealType bx = cell[0][1], by = cell[1][1], bz = cell[2][1];
    const RealType cx = cell[0][2], cy = cell[1][2], cz = cell[2][2];

    // a . (b x c)
    return std::abs(
        ax * (by * cz - bz * cy) -
        ay * (bx * cz - bz * cx) +
        az * (bx * cy - by * cx)
    );
}

int fill_shifts(const RealType cell[3][3], RealType cutoff, Workspace& ws) {
    // Lattice vectors = columns of cell
    RealType a[3] = {cell[0][0], cell[1][0], cell[2][0]};
    RealType b[3] = {cell[0][1], cell[1][1], cell[2][1]};
    RealType c[3] = {cell[0][2], cell[1][2], cell[2][2]};

    RealType vol = cell_volume(cell);

    // Cross products for interplanar spacings
    auto cross = [](const RealType u[3], const RealType v[3], RealType w[3]) {
        w[0] = u[1] * v[2] - u[2] * v[1];
        w[1] = u[2] * v[0] - u[0] * v[2];
        w[2] = u[0] * v[1] - u[1] * v[0];
    };
    auto norm3 = [](const RealType v[3]) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    };

    RealType bc[3], ca[3], ab[3];
    cross(b, c, bc);
    cross(c, a, ca);
    cross(a, b, ab);

    RealType h_a = vol / norm3(bc);
    RealType h_b = vol / norm3(ca);
    RealType h_c = vol / norm3(ab);

    int na = static_cast<int>(std::ceil(cutoff / h_a)) + 1;
    int nb = static_cast<int>(std::ceil(cutoff / h_b)) + 1;
    int nc = static_cast<int>(std::ceil(cutoff / h_c)) + 1;

    // Count total shifts and fill host arrays
    int total = (2 * na + 1) * (2 * nb + 1) * (2 * nc + 1);

    // Fill shift_cart on host: shift_cart[s] = shifts[s] @ cell^T
    // cell^T[i][j] = cell[j][i], so shift_cart[s][i] = sum_j shifts[s][j] * cell[i][j]
    auto h_shifts    = Kokkos::create_mirror_view(ws.shifts);
    auto h_shiftcart = Kokkos::create_mirror_view(ws.shift_cart);

    int idx = 0;
    for (int ia = -na; ia <= na; ++ia) {
        for (int ib = -nb; ib <= nb; ++ib) {
            for (int ic = -nc; ic <= nc; ++ic) {
                h_shifts(idx, 0) = ia;
                h_shifts(idx, 1) = ib;
                h_shifts(idx, 2) = ic;
                for (int d = 0; d < 3; ++d) {
                    h_shiftcart(idx, d) =
                        static_cast<RealType>(ia) * cell[d][0] +
                        static_cast<RealType>(ib) * cell[d][1] +
                        static_cast<RealType>(ic) * cell[d][2];
                }
                ++idx;
            }
        }
    }

    Kokkos::deep_copy(ws.shifts, h_shifts);
    Kokkos::deep_copy(ws.shift_cart, h_shiftcart);

    ws.n_shifts = total;
    return total;
}

}  // namespace klasp
