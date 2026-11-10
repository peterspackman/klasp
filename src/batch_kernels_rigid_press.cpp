// Soft-sphere repulsion + pressure batch kernel for collision resolution.
// Uses the asymmetric-unit pair iteration pattern (mol_0 vs all images).
// Energy = p*V + sum A*(sigma/r)^12 in Hartree (r in Bohr, V in Bohr^3).
// Forces = -dE/dr in Hartree/Bohr.
// Cell strain DOFs (Voigt) optimized alongside molecular translations/rotations.

#include "klasp/batch_kernels.hpp"
#include "klasp/pair_iteration.hpp"
#include "klasp/rigid_press.hpp"
#include "klasp/rotation.hpp"
#include "klasp/lbfgs.hpp"
#include "klasp/constants.hpp"
#include "klasp/sgdata.hpp"

namespace klasp {

template <typename Space>
void batch_energy_gradient_rigid_press(const ModelTablesT<Space>& tables,
                                       BatchGeometryT<Space>& geom,
                                       RigidDofStateT<Space>& dofs,
                                       BatchLBFGSStateT<Space>& lbfgs,
                                       const BatchOptimizeSettings& settings, int iteration) {
    const int n_batch = geom.n_batch;
    const int n_indep = tables.n_indep;

    auto b_cell    = geom.cell;
    auto b_ci      = geom.cell_inv;
    auto b_E       = geom.energies;
    auto b_dgt     = dofs.d_gt;
    auto b_dgr     = dofs.d_gr;
    auto b_dgs     = dofs.d_g_strain;
    auto sym_rot   = tables.sym_rotations;
    auto sym_idx   = tables.sym_indep_index;

    using team_t = typename Kokkos::TeamPolicy<Space>::member_type;

    Kokkos::parallel_for("rp_zero",
        Kokkos::TeamPolicy<Space>(n_batch, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_t& team) {
            const int b = team.league_rank();
            // No per-atom forces to clear: the press contracts directly to
            // rigid-body gradients.
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_indep),
                [&](int k) { for (int d = 0; d < 3; ++d) { b_dgt(b,k,d) = 0; b_dgr(b,k,d) = 0; } });
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                b_E(b) = 0;
                for (int s = 0; s < 6; ++s) b_dgs(b, s) = 0;
            });
        });
    Space().fence();

    // Pair iteration: WCA-12 over the asymmetric-unit pattern, using the same
    // scaffolding as the production gradient kernel. MaxRank 0 and no dipoles.
    WCA12PairFn<Space> wca{
        tables.vdw_radii,
        RP_EPS,
        RP_ALPHA,
        RP_MAX_CUTOFF * RP_MAX_CUTOFF
    };
    batch_asymunit_pair_iteration<0, Space>(tables, geom, dofs,
                                            dofs.d_gt, dofs.d_gr,
                                            /*has_dipoles=*/false,
                                            "pair_iter", wca);

    // p*V is added by batch_post_pairs, not here.

    // The press's model: a short-range pair potential cut at RP_MAX_CUTOFF and
    // nothing else.
    batch_post_pairs<0, Space>(pair_only_force_field(RP_MAX_CUTOFF),
                               tables, geom, dofs, settings, nullptr);

    if (!settings.energy_only)
        batch_lbfgs_step(tables, geom, dofs, lbfgs, settings, iteration);
}

// -----------------------------------------------------------------------
// Full optimize loop
// -----------------------------------------------------------------------
template <typename Space>
BatchOptimizeResult batch_optimize_rigid_press(BatchWorkspaceT<Space>& bws,
                                               const BatchOptimizeSettings& settings,
                                               int lattice_type) {
    LatticeType lattice = static_cast<LatticeType>(lattice_type);
    const ForceField rp_model = pair_only_force_field(RP_MAX_CUTOFF);
    const int n_batch = bws.geom.n_batch;
    const int n_indep = bws.tables.n_indep;

    batch_set_strain_dofs(bws.geom, bws.dofs, static_cast<int>(lattice));
    Kokkos::deep_copy(bws.dofs.cell_ref, bws.geom.cell);
    Kokkos::deep_copy(bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(bws.dofs.d_g_strain, RealType(0));

    const int n_mol_dof = n_indep * 6;
    const int n_dof = n_mol_dof + bws.dofs.n_cell_dof;
    auto lbfgs = allocate_batch_lbfgs_state<Space>(n_batch, n_dof, settings.lbfgs_memory);

    for (int iter = 0; iter < settings.n_iterations; ++iter) {
        batch_expand_asymmetric_unit<0, Space>(bws.tables, bws.geom, bws.dofs);
        batch_energy_gradient_rigid_press<Space>(bws.tables, bws.geom, bws.dofs, lbfgs, settings, iter);

        // Cell, volume and shift range from the new strain, as in the Ewald kernels.
        batch_update_cell_from_strain<Space>(rp_model, bws.geom, bws.dofs);
    }

    // Extract updated cell_params (Bohr, degrees) on the device for later stages.
    cell_matrix_to_params_device<Space>(bws.geom, n_batch);

    bws.dofs.n_cell_dof = 0;

    batch_setup_cells(bws.tables, bws.geom);

    BatchOptimizeResult result;
    result.final_energies = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws.geom.energies);
    result.final_translations = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws.dofs.translations);
    result.final_rotations = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws.dofs.rot_vectors);
    result.n_iterations = settings.n_iterations;
    result.n_batch = n_batch;
    return result;
}

template void batch_energy_gradient_rigid_press<ExecSpace>(
    const ModelTables&, BatchGeometry&, RigidDofState&, BatchLBFGSState&,
                                                            const BatchOptimizeSettings&, int);
template BatchOptimizeResult batch_optimize_rigid_press<ExecSpace>(BatchWorkspace&,
                                                                    const BatchOptimizeSettings&,
                                                                    int);

}  // namespace klasp
