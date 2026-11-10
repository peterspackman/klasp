#pragma once
// Batch energy/gradient kernel strategies.

#include "klasp/batch.hpp"
#include "klasp/trust_region.hpp"

namespace klasp {

// Asymmetric-unit kernel for any Z': each independent molecule against all
// others, forces on identity copies expanded by symmetry. Computes energy and
// DOF gradients (d_gt, d_gr, d_g_strain); takes no optimizer step.
template <int MaxRank, typename Space = ExecSpace>
void batch_energy_gradient_asymunit(const ModelTablesT<Space>& tables,
                                     BatchGeometryT<Space>& geom,
                                     RigidDofStateT<Space>& dofs,
                                     const BatchOptimizeSettings& settings, int iteration,
                                     KernelTimings* timings = nullptr);

// Rigid-body Jacobians of the asymmetric unit (identity copies only), for
// contracting forces and multipole adjoints to DOF gradients.
template <int MaxRank, typename Space = ExecSpace>
void batch_build_jacobians(const ModelTablesT<Space>& tables,
                           const BatchGeometryT<Space>& geom,
                           RigidDofStateT<Space>& dofs);

template <int MaxRank, typename Space = ExecSpace>
BatchOptimizeResult batch_optimize_asymunit(BatchWorkspaceT<Space>& bws,
                                             const BatchOptimizeSettings& settings);

// Rigid press (stage 0): pairwise volume-minimizing soft repulsion to resolve
// collisions before the Ewald optimization.
template <typename Space = ExecSpace>
void batch_energy_gradient_rigid_press(const ModelTablesT<Space>& tables,
                                       BatchGeometryT<Space>& geom,
                                       RigidDofStateT<Space>& dofs,
                                       BatchLBFGSStateT<Space>& lbfgs,
                                       const BatchOptimizeSettings& settings, int iteration);

template <typename Space = ExecSpace>
BatchOptimizeResult batch_optimize_rigid_press(BatchWorkspaceT<Space>& bws,
                                               const BatchOptimizeSettings& settings,
                                               int lattice_type = 0);

// Seed the scaled Hessian (molecular + strain blocks) for the slots selected by
// bws.tr.hess_mask, empty meaning all.
template <int MaxRank, typename Space = ExecSpace>
void batch_seed_hessian_blocks(BatchWorkspaceT<Space>& bws,
                               const BatchOptimizeSettings& settings,
                               const tr::StepViews<Space>& sv, int nb,
                               bool warm_mol);

// Fill trs.strain_hess with the analytic strain second derivative (batch, 6, 6)
// in Voigt order, and trs.fc with the per-atom force constants; (re)allocates
// both if their shape is wrong.
template <typename Space = ExecSpace>
void batch_strain_hessian(const ModelTablesT<Space>& tables,
                          const BatchGeometryT<Space>& geom,
                          TrustRegionScratchT<Space>& trs,
                          const BatchOptimizeSettings& settings);

// Recompute cell_params (a, b, c, alpha, beta, gamma) from the cell matrix.
// batch_update_cell_from_strain does not update cell_params, so call this
// before reading them after a step.
template <typename Space = ExecSpace>
void cell_matrix_to_params_device(const CellStateT<Space>& cells, int count) {
    auto cell_params = cells.cell_params;
    auto cell        = cells.cell;
    constexpr RealType RAD2DEG = RealType(180.0 / M_PI);
    Kokkos::parallel_for("cell_to_params",
        Kokkos::RangePolicy<Space>(0, count),
        KOKKOS_LAMBDA(const int b) {
            RealType ax = cell(b,0,0), ay = cell(b,1,0), az = cell(b,2,0);
            RealType bx = cell(b,0,1), by = cell(b,1,1), bz = cell(b,2,1);
            RealType cx = cell(b,0,2), cy = cell(b,1,2), cz = cell(b,2,2);
            RealType a  = Kokkos::sqrt(ax*ax + ay*ay + az*az);
            RealType bv = Kokkos::sqrt(bx*bx + by*by + bz*bz);
            RealType c  = Kokkos::sqrt(cx*cx + cy*cy + cz*cz);
            auto clamp1 = [](RealType x) {
                return x < RealType(-1) ? RealType(-1)
                    : (x > RealType(1) ? RealType(1) : x);
            };
            cell_params(b,0) = a;
            cell_params(b,1) = bv;
            cell_params(b,2) = c;
            cell_params(b,3) = Kokkos::acos(clamp1((bx*cx+by*cy+bz*cz)/(bv*c))) * RAD2DEG;
            cell_params(b,4) = Kokkos::acos(clamp1((ax*cx+ay*cy+az*cz)/(a*c)))  * RAD2DEG;
            cell_params(b,5) = Kokkos::acos(clamp1((ax*bx+ay*by+az*bz)/(a*bv))) * RAD2DEG;
        });
}

}  // namespace klasp
