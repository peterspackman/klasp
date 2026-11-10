#pragma once

#include <Kokkos_Core.hpp>
#include "klasp/mat3.hpp"
#include "klasp/types.hpp"

namespace klasp {

// Compute Cartesian symmetry operator from fractional: S_cart = cell @ S_frac @ cell_inv
KOKKOS_INLINE_FUNCTION
void frac_to_cart_symop(const RealType cell[3][3], const RealType cell_inv[3][3],
                         const RealType S_frac[3][3], RealType S_cart[3][3]) {
    RealType tmp[3][3];
    mat3_mul(S_frac, cell_inv, tmp);
    mat3_mul(cell, tmp, S_cart);
}

// Parity (±1) from determinant of rotation matrix
KOKKOS_INLINE_FUNCTION
int rotation_parity(const RealType R[3][3]) {
    return mat3_det(R) > 0 ? 1 : -1;
}

// Rodrigues formula: angle-axis (3,) -> 3x3 rotation matrix.
// R = I + k1*K + k2*K^2 where K is skew-symmetric [aa]_x
// k1 = sin(angle)/angle, k2 = (1 - cos(angle))/angle^2
KOKKOS_INLINE_FUNCTION
void angle_axis_to_matrix(const RealType aa[3], RealType R[3][3]) {
    const RealType angle_sq = aa[0]*aa[0] + aa[1]*aa[1] + aa[2]*aa[2];
    const RealType angle = Kokkos::sqrt(angle_sq + 1e-30);

    // sinc(angle/pi) = sin(angle)/angle, stable at zero
    RealType k1, k2;
    if (angle < 1e-10) {
        k1 = 1.0 - angle_sq / 6.0;
        k2 = 0.5 - angle_sq / 24.0;
    } else {
        k1 = Kokkos::sin(angle) / angle;
        k2 = (1.0 - Kokkos::cos(angle)) / angle_sq;
    }

    // Skew-symmetric K
    // K = [  0   -az   ay ]
    //     [  az   0   -ax ]
    //     [ -ay   ax   0  ]
    const RealType ax = aa[0], ay = aa[1], az = aa[2];

    // K^2[i][j] = sum_k K[i][k]*K[k][j]
    // K^2 = [ -(ay^2+az^2)   ax*ay         ax*az       ]
    //       [  ax*ay        -(ax^2+az^2)   ay*az       ]
    //       [  ax*az         ay*az        -(ax^2+ay^2) ]

    R[0][0] = 1.0 + k2 * (-(ay*ay + az*az));
    R[0][1] = k1 * (-az) + k2 * (ax * ay);
    R[0][2] = k1 * (ay) + k2 * (ax * az);

    R[1][0] = k1 * (az) + k2 * (ax * ay);
    R[1][1] = 1.0 + k2 * (-(ax*ax + az*az));
    R[1][2] = k1 * (-ax) + k2 * (ay * az);

    R[2][0] = k1 * (-ay) + k2 * (ax * az);
    R[2][1] = k1 * (ax) + k2 * (ay * az);
    R[2][2] = 1.0 + k2 * (-(ax*ax + ay*ay));
}

// Derivative of angle_axis_to_matrix w.r.t. aa[m], producing three 3x3 matrices.
// dR/d(aa_m) = dk1_m * K + k1 * dK_m + dk2_m * K^2 + k2 * (dK_m*K + K*dK_m)
// where dK_m is the m-th SO(3) generator.
KOKKOS_INLINE_FUNCTION
void angle_axis_to_matrix_deriv(const RealType aa[3],
    RealType dR0[3][3], RealType dR1[3][3], RealType dR2[3][3])
{
    const RealType ax = aa[0], ay = aa[1], az = aa[2];
    const RealType angle_sq = ax*ax + ay*ay + az*az;
    const RealType angle = Kokkos::sqrt(angle_sq + 1e-30);

    RealType k1, k2, dk1_coeff, dk2_coeff;
    if (angle < 1e-10) {
        k1 = 1.0 - angle_sq / 6.0;
        k2 = 0.5 - angle_sq / 24.0;
        // dk1/d(angle) = -angle/3, dk1/d(aa_m) = dk1/d(angle) * d(angle)/d(aa_m)
        // d(angle)/d(aa_m) = aa_m/angle
        // dk1/d(aa_m) = -aa_m/3
        dk1_coeff = -1.0 / 3.0;  // multiply by aa_m
        dk2_coeff = -1.0 / 12.0; // multiply by aa_m
    } else {
        const RealType s = Kokkos::sin(angle);
        const RealType c = Kokkos::cos(angle);
        k1 = s / angle;
        k2 = (1.0 - c) / angle_sq;
        // dk1/d(aa_m) = (angle*cos - sin)/angle^3 * aa_m
        dk1_coeff = (angle * c - s) / (angle_sq * angle);
        // dk2/d(aa_m) = (angle*sin - 2*(1-cos))/angle^4 * aa_m
        dk2_coeff = (angle * s - 2.0 * (1.0 - c)) / (angle_sq * angle_sq);
    }

    RealType K[3][3] = {
        { 0.0, -az,  ay},
        { az,   0.0, -ax},
        {-ay,   ax,   0.0}
    };

    RealType KK[3][3];
    KK[0][0] = -(ay*ay + az*az); KK[0][1] = ax*ay;             KK[0][2] = ax*az;
    KK[1][0] = ax*ay;            KK[1][1] = -(ax*ax + az*az);  KK[1][2] = ay*az;
    KK[2][0] = ax*az;            KK[2][1] = ay*az;             KK[2][2] = -(ax*ax + ay*ay);

    // SO(3) generators dK_m:
    // dK_0 = [[ 0, 0, 0], [ 0, 0,-1], [ 0, 1, 0]]
    // dK_1 = [[ 0, 0, 1], [ 0, 0, 0], [-1, 0, 0]]
    // dK_2 = [[ 0,-1, 0], [ 1, 0, 0], [ 0, 0, 0]]

    // For each m, compute:
    // dR_m = dk1_m * K + k1 * dK_m + dk2_m * K^2 + k2 * (dK_m*K + K*dK_m)
    // where dk1_m = dk1_coeff * aa[m], dk2_m = dk2_coeff * aa[m]

    RealType* dR[3] = {&dR0[0][0], &dR1[0][0], &dR2[0][0]};

    RealType gen[3][3][3] = {
        {{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},  // dK_0
        {{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}},  // dK_1
        {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}}   // dK_2
    };

    for (int m = 0; m < 3; ++m) {
        RealType dk1_m = dk1_coeff * aa[m];
        RealType dk2_m = dk2_coeff * aa[m];

        // dK_m * K + K * dK_m
        RealType dKK[3][3], KdK[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                dKK[i][j] = 0.0;
                KdK[i][j] = 0.0;
                for (int p = 0; p < 3; ++p) {
                    dKK[i][j] += gen[m][i][p] * K[p][j];
                    KdK[i][j] += K[i][p] * gen[m][p][j];
                }
            }

        RealType (*out)[3] = reinterpret_cast<RealType(*)[3]>(dR[m]);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                out[i][j] = dk1_m * K[i][j]
                           + k1 * gen[m][i][j]
                           + dk2_m * KK[i][j]
                           + k2 * (dKK[i][j] + KdK[i][j]);
            }
    }
}

// Map 3 values in [0,1) to a uniform random rotation matrix via quaternion.
KOKKOS_INLINE_FUNCTION
void vec2rot(const RealType v[3], RealType R[3][3]) {
    RealType v0 = v[0], v1 = v[1], v2 = v[2];
    RealType s1 = Kokkos::sin(2.0 * M_PI * v1);
    RealType c1 = Kokkos::cos(2.0 * M_PI * v1);
    RealType s2 = Kokkos::sin(2.0 * M_PI * v2);
    RealType c2 = Kokkos::cos(2.0 * M_PI * v2);
    RealType r0 = Kokkos::sqrt(1.0 - v0);
    RealType r1 = Kokkos::sqrt(v0);
    RealType x = r0 * s1, y = r0 * c1, z = r1 * s2, w = r1 * c2;

    R[0][0] = 1 - 2*y*y - 2*z*z;  R[0][1] = 2*x*y + 2*z*w;      R[0][2] = 2*x*z - 2*y*w;
    R[1][0] = 2*x*y - 2*z*w;      R[1][1] = 1 - 2*x*x - 2*z*z;  R[1][2] = 2*y*z + 2*x*w;
    R[2][0] = 2*x*z + 2*y*w;      R[2][1] = 2*y*z - 2*x*w;      R[2][2] = 1 - 2*x*x - 2*y*y;
}

// Convert rotation matrix to quaternion (w, x, y, z).
// Uses Shepperd's method for numerical stability at all angles.
KOKKOS_INLINE_FUNCTION
void rotation_matrix_to_quaternion(const RealType R[3][3], RealType q[4]) {
    // q[0]=w, q[1]=x, q[2]=y, q[3]=z
    RealType t;
    if (R[2][2] < 0) {
        if (R[0][0] > R[1][1]) {
            t = 1.0 + R[0][0] - R[1][1] - R[2][2];
            q[0] = R[2][1] - R[1][2];
            q[1] = t;
            q[2] = R[0][1] + R[1][0];
            q[3] = R[2][0] + R[0][2];
        } else {
            t = 1.0 - R[0][0] + R[1][1] - R[2][2];
            q[0] = R[0][2] - R[2][0];
            q[1] = R[0][1] + R[1][0];
            q[2] = t;
            q[3] = R[1][2] + R[2][1];
        }
    } else {
        if (R[0][0] < -R[1][1]) {
            t = 1.0 - R[0][0] - R[1][1] + R[2][2];
            q[0] = R[1][0] - R[0][1];
            q[1] = R[2][0] + R[0][2];
            q[2] = R[1][2] + R[2][1];
            q[3] = t;
        } else {
            t = 1.0 + R[0][0] + R[1][1] + R[2][2];
            q[0] = t;
            q[1] = R[2][1] - R[1][2];
            q[2] = R[0][2] - R[2][0];
            q[3] = R[1][0] - R[0][1];
        }
    }
    RealType s = 0.5 / Kokkos::sqrt(t);
    q[0] *= s; q[1] *= s; q[2] *= s; q[3] *= s;
    // Ensure w >= 0 for canonical form
    if (q[0] < 0) { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
}

// Convert quaternion to angle-axis vector.
KOKKOS_INLINE_FUNCTION
void quaternion_to_angle_axis(const RealType q[4], RealType aa[3]) {
    // q = [w, x, y, z], angle = 2*acos(w), axis = (x,y,z)/sin(angle/2)
    RealType w = q[0];
    if (w > 1.0) w = 1.0;
    if (w < -1.0) w = -1.0;
    RealType half_angle = Kokkos::acos(w);
    RealType angle = 2.0 * half_angle;
    RealType s = Kokkos::sin(half_angle);
    if (s < 1e-15) {
        aa[0] = aa[1] = aa[2] = 0.0;
    } else {
        aa[0] = q[1] / s * angle;
        aa[1] = q[2] / s * angle;
        aa[2] = q[3] / s * angle;
    }
}

// Convert rotation matrix to angle-axis vector via quaternion.
// Robust at all angles including near 0 and pi.
KOKKOS_INLINE_FUNCTION
void rotation_matrix_to_angle_axis(const RealType R[3][3], RealType aa[3]) {
    RealType q[4];
    rotation_matrix_to_quaternion(R, q);
    quaternion_to_angle_axis(q, aa);
}

}  // namespace klasp
