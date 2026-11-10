#pragma once
// Symmetry-adapted cell strain degrees of freedom. The optimiser's cell
// variables are amplitudes q along a basis of allowed Voigt strain directions:
//
//     eps_v = sum_s q_s * basis[s][v]        (basis_expand)
//     q_s   = sum_v basis[s][v] * eps_v      (basis_project, rows orthonormal)
//     dE/dq_s = sum_v basis[s][v] * dE/deps_v
//
// The basis couples components where the lattice requires it (e.g. eps_xx =
// eps_yy for tetragonal), so the metric cannot leave its lattice type.
//
// Voigt order matches batch_shared.cpp: 0,1,2 = xx,yy,zz; 3,4,5 = yz,xz,xy,
// with the engineering factor of 2 on the shear components.

#include <Kokkos_Core.hpp>
#include "klasp/sgdata.hpp"
#include "klasp/types.hpp"

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

// The strain basis for one structure, device-callable; the host entry points
// below delegate to it.
//
// `cell` is 3x3 row-major with the lattice vectors as columns, read only for
// rhombohedral (its basis depends on the 3-fold direction). nullptr there gives
// the isotropic DOF alone. `lattice` is a LatticeType value (klasp/sgdata.hpp).
//
// Writes at most `max_dof` rows and returns how many it wrote. Rows are ordered
// lengths before angles, so a length-only basis is a prefix of the full one.
KOKKOS_INLINE_FUNCTION
int build_strain_basis(int lattice, const RealType* cell, RealType* basis, int max_dof) {
    for (int i = 0; i < 36; ++i) basis[i] = 0;
    if (max_dof <= 0) return 0;
    auto row = [&](int r) { return basis + r * 6; };
    const RealType r2 = RealType(1) / Kokkos::sqrt(RealType(2));
    const RealType r3 = RealType(1) / Kokkos::sqrt(RealType(3));
    auto capped = [&](int n) { return n < max_dof ? n : max_dof; };

    if (lattice == CUBIC) {
        row(0)[0] = row(0)[1] = row(0)[2] = r3;
        return 1;
    }
    if (lattice == RHOMBOHEDRAL) {
        // For a 3-fold about unit n, the symmetric tensors commuting with the
        // rotation are spanned by P = n n^T and Q = I - n n^T: the (a, alpha)
        // freedom of a rhombohedral cell.
        RealType n[3] = {0, 0, 0};                   // n along a + b + c
        RealType len = 0;
        if (cell) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) n[r] += cell[r*3 + c];
            len = Kokkos::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        }
        if (!(len > RealType(1e-12))) {              // no cell, or degenerate
            row(0)[0] = row(0)[1] = row(0)[2] = r3;
            return 1;
        }
        for (int r = 0; r < 3; ++r) n[r] /= len;

        // Voigt with the engineering factor of 2 on the shears, unit-normalised.
        auto emit = [&](int row_idx, const RealType t[3][3]) {
            RealType v[6] = {t[0][0], t[1][1], t[2][2],
                             RealType(2)*t[1][2], RealType(2)*t[0][2], RealType(2)*t[0][1]};
            RealType nrm = 0;
            for (int i = 0; i < 6; ++i) nrm += v[i]*v[i];
            nrm = Kokkos::sqrt(nrm);
            if (nrm > RealType(1e-12))
                for (int i = 0; i < 6; ++i) basis[row_idx*6 + i] = v[i] / nrm;
        };
        const RealType iso[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
        emit(0, iso);                                // isotropic first
        if (max_dof <= 1) return 1;
        // Uniaxial along n as P - Q/2 = 1.5 n n^T - I/2, traceless and so
        // orthogonal to the isotropic row.
        RealType uni[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                uni[i][j] = RealType(1.5)*n[i]*n[j]
                          - RealType(0.5)*(i == j ? RealType(1) : RealType(0));
        emit(1, uni);
        return 2;
    }
    if (lattice == TETRAGONAL || lattice == HEXAGONAL) {
        row(0)[0] = row(0)[1] = r2;                  // eps_xx = eps_yy
        if (max_dof <= 1) return 1;
        row(1)[2] = 1;
        return 2;
    }
    if (lattice == ORTHORHOMBIC) {
        row(0)[0] = 1; if (max_dof <= 1) return 1;
        row(1)[1] = 1; if (max_dof <= 2) return 2;
        row(2)[2] = 1;
        return 3;
    }
    if (lattice == MONOCLINIC) {
        row(0)[0] = 1; row(1)[1] = 1; row(2)[2] = 1;
        if (max_dof <= 3) return capped(3);
        row(3)[4] = 1;                               // unique axis b -> xz shear only
        return 4;
    }
    // Triclinic and anything unrecognised: the full six.
    row(0)[0] = 1; row(1)[1] = 1; row(2)[2] = 1;
    if (max_dof <= 3) return capped(3);
    row(3)[3] = 1; if (max_dof <= 4) return 4;
    row(4)[4] = 1; if (max_dof <= 5) return 5;
    row(5)[5] = 1;
    return 6;
}

// ---------------------------------------------------------------------------
// Host-side basis construction (src/strain_basis.cpp).
// ---------------------------------------------------------------------------

// Writes n_dof unit-normalised rows into `basis` (6x6 row-major) and returns
// n_dof. `lattice_type` is a LatticeType enum value (klasp/sgdata.hpp).
// fix_angles drops the shear DOFs. Rhombohedral gets the isotropic DOF only;
// prefer set_strain_basis_cell.
int set_strain_basis(int lattice_type, RealType* basis, bool fix_angles = false);

// As above but exact for rhombohedral. `cell` is 3x3 row-major with the lattice
// vectors as columns. The basis stays valid as the cell deforms along it, so it
// is built once per structure.
int set_strain_basis_cell(int lattice_type, const RealType* cell, RealType* basis,
                          bool fix_angles = false);

}  // namespace klasp
