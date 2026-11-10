// The damped L-BFGS step the rigid press (stage 0) takes.

#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/lbfgs.hpp"
#include <fmt/base.h>

namespace klasp {

// ---------------------------------------------------------------------------
// Standalone L-BFGS step: pack gradients, compute direction, update DOFs.
//
// State convention:
//   lbfgs_x  — current position x_k (just evaluated gradient here)
//   lbfgs_xn — previous position x_{k-1} (used to build s_k = x_k - x_{k-1})
//   lbfgs_g  — current gradient g_k = grad(x_k)
//   lbfgs_gn — previous gradient g_{k-1}
// ---------------------------------------------------------------------------
template <typename Space>
void batch_lbfgs_step(const ModelTablesT<Space>& tables,
                       const BatchGeometryT<Space>& geom,
                       RigidDofStateT<Space>& dofs,
                       BatchLBFGSStateT<Space>& lbfgs,
                       const BatchOptimizeSettings& settings,
                       int iteration) {
    const int n_batch = geom.n_batch;
    const int n_indep = tables.n_indep;
    const int n_cell_dof = dofs.n_cell_dof;
    const int n_mol_dof = n_indep * 6;
    const int n_dof = n_mol_dof + n_cell_dof;
    const RealType max_step = settings.max_step;
    const RealType rot_scale = settings.rotation_scale;
    const RealType cell_scale = settings.cell_scale;
    const RealType max_strain_step = settings.max_strain_step;

    auto b_dgt    = dofs.d_gt;
    auto b_dgr    = dofs.d_gr;
    auto b_dgs    = dofs.d_g_strain;
    auto b_trans  = dofs.translations;
    auto b_rots   = dofs.rot_vectors;
    auto b_strain = dofs.strain;
    auto b_basis  = dofs.cell_dof_basis;

    auto lbfgs_x   = lbfgs.x;
    auto lbfgs_g   = lbfgs.g;
    auto lbfgs_xn  = lbfgs.x_new;
    auto lbfgs_gn  = lbfgs.g_new;
    auto lbfgs_d   = lbfgs.direction;
    auto lbfgs_s   = lbfgs.s_hist;
    auto lbfgs_y   = lbfgs.y_hist;
    auto lbfgs_rho = lbfgs.rho;
    auto lbfgs_alpha = lbfgs.alpha_scratch;
    auto lbfgs_ns  = lbfgs.num_stored;
    auto lbfgs_cur = lbfgs.cursor;
    const int lbfgs_mem = lbfgs.memory;

    Kokkos::parallel_for("batch_lbfgs_step", n_batch,
        KOKKOS_LAMBDA(const int b) {
            for (int k = 0; k < n_indep; ++k)
                for (int d = 0; d < 3; ++d) {
                    lbfgs_g(b, k*3+d) = b_dgt(b,k,d);
                    lbfgs_g(b, n_indep*3+k*3+d) = b_dgr(b,k,d);
                }

            // Pack cell strain gradients: dE/dq_s = sum_v basis[s][v] dE/deps_v.
            const int n_cd = n_cell_dof;
            if (n_cd > 0) {
                RealType geps[6], gq[6];
                for (int v = 0; v < 6; ++v) geps[v] = b_dgs(b, v);
                basis_project(&b_basis(b,0,0), geps, n_cd, gq);
                for (int s = 0; s < n_cd; ++s) lbfgs_g(b, n_mol_dof + s) = gq[s];
            }

            if (iteration == 0) {
                // Initialize x from the current DOFs. No history → steepest
                // descent direction with the per-subblock scaling heuristic.
                for (int k = 0; k < n_indep; ++k)
                    for (int d = 0; d < 3; ++d) {
                        lbfgs_x(b, k*3+d) = b_trans(b,k,d);
                        lbfgs_x(b, n_indep*3+k*3+d) = b_rots(b,k,d);
                    }
                for (int s = 0; s < n_cd; ++s)
                    lbfgs_x(b, n_mol_dof + s) = 0;

                for (int i = 0; i < n_dof; ++i)
                    lbfgs_d(b, i) = -lbfgs_g(b, i);
            } else {
                // Previous iter stored x_{k-1} in xn and g_{k-1} in gn. Build
                // the curvature pair from the real step taken.
                RealType s_vec[54], y_vec[54];
                for (int i = 0; i < n_dof; ++i) {
                    s_vec[i] = lbfgs_x(b,i) - lbfgs_xn(b,i);
                    y_vec[i] = lbfgs_g(b,i) - lbfgs_gn(b,i);
                }
                int ns = lbfgs_ns(b);
                int cur = lbfgs_cur(b);
                ns = lbfgs_push(&lbfgs_s(b,0,0), &lbfgs_y(b,0,0), &lbfgs_rho(b,0),
                                s_vec, y_vec, ns, &cur, lbfgs_mem, n_dof);
                lbfgs_ns(b) = ns;
                lbfgs_cur(b) = cur;

                lbfgs_direction(&lbfgs_s(b,0,0), &lbfgs_y(b,0,0), &lbfgs_rho(b,0),
                                &lbfgs_alpha(b,0), &lbfgs_g(b,0), &lbfgs_d(b,0),
                                ns, lbfgs_mem, n_dof);

                // Guard: if the Hessian approximation gives an ascent
                // direction, fall back to SD and clear history.
                RealType dg = vec_dot(&lbfgs_d(b,0), &lbfgs_g(b,0), n_dof);
                if (dg > 0) {
                    for (int i = 0; i < n_dof; ++i) lbfgs_d(b,i) = -lbfgs_g(b,i);
                    lbfgs_ns(b) = 0;
                    lbfgs_cur(b) = 0;
                }
            }

            // Scale the rotation and cell subblocks.
            for (int i = n_indep*3; i < n_mol_dof; ++i)
                lbfgs_d(b, i) *= rot_scale;
            for (int i = n_mol_dof; i < n_dof; ++i)
                lbfgs_d(b, i) *= cell_scale;

            // Step-size clamp, no line search: a fixed iteration count keeps batch-wide
            // work predictable.
            RealType dnorm = Kokkos::sqrt(vec_norm_sq(&lbfgs_d(b,0), n_dof));
            RealType alpha = (dnorm > 1e-15) ? Kokkos::fmin(1.0, max_step / dnorm) : 0.0;

            // Stash x_k into xn before overwriting x with the new trial.
            for (int i = 0; i < n_dof; ++i) {
                lbfgs_xn(b,i) = lbfgs_x(b,i);
                lbfgs_x (b,i) = lbfgs_x(b,i) + alpha * lbfgs_d(b,i);
            }

            // Save gradient for the next iter's y_vec.
            for (int i = 0; i < n_dof; ++i)
                lbfgs_gn(b,i) = lbfgs_g(b,i);

            for (int k = 0; k < n_indep; ++k)
                for (int d = 0; d < 3; ++d) {
                    b_trans(b,k,d) = lbfgs_x(b, k*3+d);
                    b_rots(b,k,d)  = lbfgs_x(b, n_indep*3+k*3+d);
                }
            // Unpack the cell amplitudes, then expand them to Voigt strain, subject to a
            // per-amplitude step cap and the absolute strain bounds (whole move scaled,
            // not clamped per component, to stay in the symmetry subspace). Both limits
            // are written back into lbfgs_x so the next s_vec is the real step.
            if (n_cd > 0) {
                RealType q_old[6], dq[6];
                const RealType mss = max_strain_step;
                for (int s = 0; s < n_cd; ++s) {
                    q_old[s] = lbfgs_xn(b, n_mol_dof + s);
                    RealType dv = lbfgs_x(b, n_mol_dof + s) - q_old[s];
                    if (dv >  mss) dv =  mss;
                    if (dv < -mss) dv = -mss;
                    dq[s] = dv;
                }
                const RealType t = strain_step_fraction(&b_basis(b,0,0), q_old, dq, n_cd);
                RealType q_new[6];
                for (int s = 0; s < n_cd; ++s) {
                    q_new[s] = q_old[s] + t * dq[s];
                    lbfgs_x(b, n_mol_dof + s) = q_new[s];
                }
                RealType eps[6];
                basis_expand(&b_basis(b,0,0), q_new, n_cd, eps);
                for (int v = 0; v < 6; ++v) b_strain(b, v) = eps[v];
            }
        });
    Space().fence();
}

template void batch_lbfgs_step<ExecSpace>(const ModelTables&, const BatchGeometry&,
                                          RigidDofState&, BatchLBFGSState&,
                                            const BatchOptimizeSettings&, int);

}  // namespace klasp
