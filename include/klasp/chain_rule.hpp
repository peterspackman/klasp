#pragma once

#include "klasp/types.hpp"
#include "klasp/molecule.hpp"

namespace klasp {

// Chain rule from atom forces (n_atoms, 3), forces = -dE/dr, to DOF gradients:
// d_gt = dE/d(frac translation), d_gr = dE/d(angle-axis), both (n_indep, 3) and
// preallocated on device. d_cell is the (3,3) cell matrix on device.
void compute_dof_gradients(const MoleculeData& mol,
                           const View2D<RealType>& forces,
                           const View2D<RealType>& d_cell,
                           View2D<RealType>& d_gt,
                           View2D<RealType>& d_gr,
                           RealType* grad_trans,
                           RealType* grad_rot);

// Allocating overload for tests and non-hot paths.
void compute_dof_gradients(const MoleculeData& mol,
                           const View2D<RealType>& forces,
                           RealType* grad_trans,
                           RealType* grad_rot);

// Adds the multipole torque term. dE_dmp: (n_atoms, 35) dE/d(lab-frame Cartesian multipoles).
void compute_dof_gradients(const MoleculeData& mol,
                           const View2D<RealType>& forces,
                           const View2D<RealType>& dE_dmp,
                           const View2D<RealType>& d_cell,
                           View2D<RealType>& d_gt,
                           View2D<RealType>& d_gr,
                           RealType* grad_trans,
                           RealType* grad_rot);

}  // namespace klasp
