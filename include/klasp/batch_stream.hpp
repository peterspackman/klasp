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
