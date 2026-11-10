#pragma once
// Device-callable DOF transforms for the PET device-resident relaxer.
//
// The optimizer's parameterization is Cartesian coordinates in the reference cell
// plus cell strain. With eps the symmetric strain in Voigt order
// [xx,yy,zz,xy,xz,yz] and def = I + voigt(eps):
//
//   h = def h0            (h0 = reference cell, columns = lattice vectors)
//   r_i = def p_i         (p_i = reference-cell Cartesian coordinate)
//   g_{p,i} = dE/dp_i = -def^T F_i
//   g_{eps} = dE/deps = W * def^-T   (see strain_correct_virial)
//
// The fractional parameterization (precond_length, current_cell, cart_from_dof,
// atom_grad: u = lambda*s_i, r_i = h s_i, lambda = cbrt|det h0|) is not used by
// the optimizer.
//
// Cells are row-major 3x3 with column j = lattice vector j: r_k = sum_j h[k*3+j] s_j.

#include <Kokkos_Core.hpp>

#include "klasp/mat3.hpp"

namespace klasp::crystdof {

// def = I + E, E symmetric from Voigt eps[6]=[xx,yy,zz,xy,xz,yz]. Row-major 3x3.
KOKKOS_INLINE_FUNCTION
void strain_to_def(const double eps[6], double def[9]) {
    def[0] = 1.0 + eps[0]; def[1] = eps[3];       def[2] = eps[4];
    def[3] = eps[3];       def[4] = 1.0 + eps[1]; def[5] = eps[5];
    def[6] = eps[4];       def[7] = eps[5];       def[8] = 1.0 + eps[2];
}

// 3x3 primitives from mat3.hpp (row-major [9] overloads), re-exported as crystdof::.
using klasp::mat3_mul;
using klasp::mat3_vec;
using klasp::mat3T_vec;
using klasp::mat3_transpose;
using klasp::mat3_det;
using klasp::mat3_inv;

// Bare symmetric 3x3 (row-major) from Voigt [xx,yy,zz,xy,xz,yz], not I+E.
KOKKOS_INLINE_FUNCTION
void voigt_to_sym(const double v[6], double m[9]) {
    m[0] = v[0]; m[1] = v[3]; m[2] = v[4];
    m[3] = v[3]; m[4] = v[1]; m[5] = v[5];
    m[6] = v[4]; m[7] = v[5]; m[8] = v[2];
}

// Conversions between the full-tensor Voigt gradient (off-diagonals hold xy+yx)
// and the rank-2 tensor. Use these, not voigt_to_sym, before rotating a gradient:
// voigt_to_sym gives 2D - diag(D), which is only correct under signed permutations.

// Full-tensor Voigt -> the rank-2 tensor it encodes (off-diagonals halved).
KOKKOS_INLINE_FUNCTION
void voigt_full_to_tensor(const double v[6], double m[9]) {
    m[0] = v[0];        m[1] = 0.5 * v[3]; m[2] = 0.5 * v[4];
    m[3] = 0.5 * v[3];  m[4] = v[1];       m[5] = 0.5 * v[5];
    m[6] = 0.5 * v[4];  m[7] = 0.5 * v[5]; m[8] = v[2];
}

// Component v of the full-tensor Voigt encoding of a symmetric tensor m.
KOKKOS_INLINE_FUNCTION
double tensor_to_voigt_full(const double m[9], int v) {
    const int vi[6] = {0, 1, 2, 0, 0, 1}, vj[6] = {0, 1, 2, 1, 2, 2};
    const double c = m[vi[v] * 3 + vj[v]];
    return v < 3 ? c : 2.0 * c;
}

// Preconditioning length for the fractional DOFs: a unit step in lambda*s moves
// an atom by ~1 Bohr (lambda = cbrt|det h0|, the cube-root cell volume).
KOKKOS_INLINE_FUNCTION
double precond_length(const double h0[9]) {
    return Kokkos::cbrt(Kokkos::fabs(mat3_det(h0)));
}

// Current cell h = (I+E) h0 from the strain DOF block (eps already de-scaled).
KOKKOS_INLINE_FUNCTION
void current_cell(const double h0[9], const double eps[6], double h[9]) {
    double def[9];
    strain_to_def(eps, def);
    mat3_mul(def, h0, h);
}

// Cartesian position of one atom from its preconditioned fractional DOF u (3):
// s = u/lambda, r = h s.
KOKKOS_INLINE_FUNCTION
void cart_from_dof(const double h[9], double lambda, const double u[3], double r[3]) {
    const double s[3] = {u[0] / lambda, u[1] / lambda, u[2] / lambda};
    mat3_vec(h, s, r);
}

// Atomic-DOF gradient from the Cartesian force F (= -dE/dr): g = -(1/lambda) h^T F.
KOKKOS_INLINE_FUNCTION
void atom_grad(const double h[9], double lambda, const double F[3], double g[3]) {
    double hTF[3];
    mat3T_vec(h, F, hTF);
    const double inv = 1.0 / lambda;
    for (int c = 0; c < 3; ++c) g[c] = -inv * hTF[c];
}

// Convert PET's current-frame virial W = sum_i (dE/dr_i) (x) r_i into the gradient
// of the cumulative strain DOF: Wc = W_tensor * def^-T, symmetric part kept.
// W and Wc are full-tensor Voigt [xx,yy,zz, xy+yx, xz+zx, yz+zy]. Returns W
// unchanged if def is singular.
KOKKOS_INLINE_FUNCTION
void strain_correct_virial(const double W[6], const double def[9], double Wc[6]) {
    // Full-tensor Voigt to tensor: halve the off-diagonals.
    const double Wt[9] = {W[0],      W[3] * 0.5, W[4] * 0.5,
                          W[3] * 0.5, W[1],      W[5] * 0.5,
                          W[4] * 0.5, W[5] * 0.5, W[2]};
    double definv[9];
    if (!mat3_inv(def, definv)) {
        for (int k = 0; k < 6; ++k) Wc[k] = W[k];
        return;
    }
    double definvT[9], Wp[9];
    mat3_transpose(definv, definvT);
    mat3_mul(Wt, definvT, Wp);  // Wp = Wt * def^-T (not symmetric in general)
    // Back to full-tensor Voigt: sum the off-diagonal pairs.
    Wc[0] = Wp[0]; Wc[1] = Wp[4]; Wc[2] = Wp[8];
    Wc[3] = Wp[1] + Wp[3];
    Wc[4] = Wp[2] + Wp[6];
    Wc[5] = Wp[5] + Wp[7];
}

// Strain-DOF gradient from the PET virial W (Voigt, Hartree): DOFs are eps*cf, so
// dE/d(eps*cf) = W/cf. Also reports max|W|/V for convergence. Writes g6[0..5].
KOKKOS_INLINE_FUNCTION
double strain_grad(const double W[6], double cell_factor, double volume, double g6[6]) {
    const double invcf = 1.0 / cell_factor;
    const double invV = volume > 0 ? 1.0 / volume : 1.0;
    double ms = 0;
    for (int k = 0; k < 6; ++k) {
        g6[k] = W[k] * invcf;
        const double sk = Kokkos::fabs(W[k]) * invV;
        ms = sk > ms ? sk : ms;
    }
    return ms;
}

// Reference-cell Cartesian parameterization: r = def p, g = -def^T F.

// Cartesian position from a reference position p: r = def p (def = I+E).
KOKKOS_INLINE_FUNCTION
void cart_from_ref(const double def[9], const double p[3], double r[3]) {
    mat3_vec(def, p, r);
}

// Atomic gradient (reference-Cartesian DOF): g = -def^T F.
KOKKOS_INLINE_FUNCTION
void atom_grad_cart(const double def[9], const double F[3], double g[3]) {
    double dTF[3];
    mat3T_vec(def, F, dTF);
    for (int c = 0; c < 3; ++c) g[c] = -dTF[c];
}

// Periodic image ranges for a cell whose rows are lattice vectors (PET frame):
// n[k] = ceil(cutoff / interplanar spacing perpendicular to lattice vector k).
KOKKOS_INLINE_FUNCTION
void image_ranges_rows(const double cell[9], double cutoff, int n[3]) {
    n[0] = n[1] = n[2] = 0;
    const double a[3] = {cell[0], cell[1], cell[2]};
    const double b[3] = {cell[3], cell[4], cell[5]};
    const double c[3] = {cell[6], cell[7], cell[8]};
    const double bc[3] = {b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]};
    const double ca[3] = {c[1]*a[2]-c[2]*a[1], c[2]*a[0]-c[0]*a[2], c[0]*a[1]-c[1]*a[0]};
    const double ab[3] = {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    const double vol = Kokkos::fabs(a[0]*bc[0] + a[1]*bc[1] + a[2]*bc[2]);
    if (vol < 1e-12) return;
    const double nbc = Kokkos::sqrt(bc[0]*bc[0]+bc[1]*bc[1]+bc[2]*bc[2]);
    const double nca = Kokkos::sqrt(ca[0]*ca[0]+ca[1]*ca[1]+ca[2]*ca[2]);
    const double nab = Kokkos::sqrt(ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2]);
    n[0] = (int)Kokkos::ceil(cutoff / (vol / nbc));
    n[1] = (int)Kokkos::ceil(cutoff / (vol / nca));
    n[2] = (int)Kokkos::ceil(cutoff / (vol / nab));
}

}  // namespace klasp::crystdof
