#pragma once
// Streaming stage-2 trust-region optimizer: a slot that converges or hits the
// iteration cap is harvested and refilled from a pool of stage-1-relaxed
// structures. Each slot owns its trust radius and Hessian, so slots are independent.

#include "klasp/batch.hpp"
#include <vector>

namespace klasp {

// Pending pool of stage-1-relaxed structures that streaming stage 2 pulls from.
// Holds full cell/Ewald state so a refill is a pure device gather.
template <typename Space = ExecSpace>
struct StreamPoolT {
    CellStateT<Space> cells;                // (P, ...)
    // Not a RigidDofStateT: a parked structure needs no Jacobians or gradients.
    SView3D<RealType, Space> translations;  // (P, n_indep, 3)
    SView3D<RealType, Space> rot_vectors;   // (P, n_indep, 3)
    // Stage-1 molecular Hessian (scaled coords), warm-starts stage 2.
    SView3D<RealType, Space> hess_mol;      // (P, nmd, nmd), nmd = n_indep*6
    // Stage-1 convergence, reported with the stage-2 row.
    SView1D<RealType, Space> s1_grad;       // (P)
    SView1D<int, Space>      s1_iters;      // (P)
    SView1D<int, Space>      seeds;         // (P)

    int P = 0;          // capacity
    int filled = 0;     // entries written by stage 1 (host-tracked)
    int consumed = 0;   // entries already pulled into slots (host-tracked)
    int n_indep = 0;
    int nmd = 0;        // n_indep * 6

    void reset() { filled = 0; consumed = 0; }
    int available() const { return filled - consumed; }
};

using StreamPool = StreamPoolT<ExecSpace>;

template <typename Space = ExecSpace>
StreamPoolT<Space> allocate_stream_pool(int P, int n_indep);

// Append the first `count` slots of bws to the pool tail; advances pool.filled.
// seeds_host length must be >= count. with_hessian is false when bws.tr.tr_hessian_mol
// does not belong to these structures; such entries get an identity Hessian block.
template <typename Space = ExecSpace>
void stream_pool_append(StreamPoolT<Space>& pool, const BatchWorkspaceT<Space>& bws,
                        int count, const std::vector<int>& seeds_host,
                        const BatchDiagnostics& s1_diag = {},
                        bool with_hessian = true);

// Rebuild the strain basis of slots idx(0..n) from their cell_ref, after a
// refill. Its own kernel, kept out of the fill functors' register budget. A
// slot whose basis cannot be built gets no rows (its cell is held) and is
// counted in failed(0). Does nothing for an imposed basis (cell_dof_n_rot 0).
template <typename Space>
void refill_strain_basis(const RigidDofStateT<Space>& dofs, SView1D<int, Space> idx,
                         int n, SView1D<int, Space> failed) {
    const int ncd = dofs.n_cell_dof, n_rot = dofs.cell_dof_n_rot, dim = dofs.cell_dof_dim;
    if (n <= 0 || ncd <= 0 || n_rot <= 0) return;
    const bool diag_only = dofs.cell_dof_diag_only;
    auto cell_ref = dofs.cell_ref;
    auto basis = dofs.cell_dof_basis;
    auto rot = dofs.cell_dof_rot;
    Kokkos::parallel_for("st_strain_basis", Kokkos::RangePolicy<Space>(0, n),
        KOKKOS_LAMBDA(const int j) {
            const int b = idx(j);
            RealType lc[9], lb[36];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) lc[r*3 + c] = cell_ref(b, r, c);
            const int got = build_symmetric_strain_basis(lc, rot.data(), n_rot, dim,
                                                         diag_only, lb);
            if (got != ncd) {
                for (int i = 0; i < 36; ++i) lb[i] = 0;
                Kokkos::atomic_inc(&failed(0));
            }
            for (int s = 0; s < 6; ++s)
                for (int v = 0; v < 6; ++v) basis(b, s, v) = lb[s*6 + v];
        });
}

// Warns when refill_strain_basis held any slot's cell.
void warn_strain_refill_failures(int n_failed);

// batch_set_strain_dofs for a streamed stage: the slots' cells are stale until
// a refill, so the count is checked over the pool's pending entries instead,
// and the n_slots slots get empty rows that their refill replaces. Throws,
// naming the entry, as batch_set_strain_dofs does.
template <typename Space>
int stream_set_strain_dofs(const SymmetryOpsT<Space>& ops, const StreamPoolT<Space>& pool,
                           RigidDofStateT<Space>& dofs, int n_slots, bool fix_angles);

extern template int stream_set_strain_dofs<ExecSpace>(const SymmetryOps&, const StreamPool&,
                                                      RigidDofState&, int, bool);

// Harvested results, one row per converged/capped pool entry (n <= pool.P).
// The cell is the optimizer's final matrix in an arbitrary frame.
struct StreamResult {
    int n = 0;
    Kokkos::View<RealType*,   Kokkos::HostSpace> energies;       // (n)
    Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> translations; // (n, n_indep, 3)
    Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> rotations;    // (n, n_indep, 3)
    Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> cell;         // (n, 3, 3)
    std::vector<int> seeds;                                       // (n)
    BatchDiagnostics diagnostics;   // empty unless settings.store_diagnostics
    BatchDiagnostics s1_diagnostics;  // stage-1 convergence, same indexing
    double wall_time_s = 0;
    KernelTimings timings;
    int total_iters = 0;            // sum of per-slot iterations run (for perf)
};

// Run streaming trust-region optimization over `pool` until it is exhausted.
// bws.geom.n_batch must equal the full slot capacity. MaxRank is 0 for stage 1.
// out_pool == nullptr: harvest to host, returned in the StreamResult.
// out_pool != nullptr: harvest device-to-device onto out_pool's tail (with the
// Hessian block); the StreamResult carries counts only. out_pool must differ from pool.
template <int MaxRank, typename Space = ExecSpace>
StreamResult batch_optimize_stream_tr(BatchWorkspaceT<Space>& bws,
                                      StreamPoolT<Space>& pool,
                                      const BatchOptimizeSettings& settings,
                                      StreamPoolT<Space>* out_pool = nullptr);

extern template StreamResult batch_optimize_stream_tr<0, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
extern template StreamResult batch_optimize_stream_tr<1, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
extern template StreamResult batch_optimize_stream_tr<2, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
extern template StreamResult batch_optimize_stream_tr<3, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);
extern template StreamResult batch_optimize_stream_tr<4, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, StreamPool*);

}  // namespace klasp
