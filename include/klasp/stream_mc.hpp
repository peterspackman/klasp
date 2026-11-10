#pragma once
// The Metropolis stepper for stream::schedule.
//
// Per slot: perturb the rigid-body and strain DOFs, accept against
// exp(-dE/RT), finish after a fixed number of moves, and emit the lowest-energy
// structure visited. Moves stay inside the support of the sampling prior
// (klasp/sampling_prior.hpp): translations wrap on the unit cube, rotations
// compose with the current orientation (Haar-preserving), and a strain step
// leaving the angle window or volume budget is refused before evaluation.
//
// Stage 2 only; there is no downstream pool.

#include "klasp/batch.hpp"
#include "klasp/batch_stream.hpp"
#include "klasp/sampling_prior.hpp"

namespace klasp {

// Chain settings; the energy model stays in BatchOptimizeSettings.
struct McSettings {
    // Boltzmann temperature in kJ/mol (already multiplied by R). <= 0 is a
    // greedy chain: downhill only.
    RealType kT = RealType(0.0083144626 * 300.0);
    // Moves per structure; the stepper finishes on this.
    int n_moves = 400;
    // Move sizes, per degree of freedom drawn uniformly in [-step, +step].
    RealType step_trans  = RealType(0.03);   // fractional COM
    RealType step_rot    = RealType(5.0 * M_PI / 180.0);  // radians
    RealType step_strain = RealType(0.01);   // dimensionless Voigt strain
    // Shortest intermolecular contact a proposal may have, Angstrom; 0 disables.
    // Required because the force field is unbounded below at short range. 1.2 A
    // still lets hot chains into the singularity; much above 1.5 A would exclude
    // carboxylic-acid O-H...O contacts.
    RealType min_contact = RealType(1.5);
    // The support the moves are confined to, from make_sampling_prior with the
    // generator's volume budget.
    SamplingPrior prior;
};

// Downhill moves are always taken, uphill with probability exp(-dE / kT).
// `u` is uniform on [0, 1). kT <= 0 means downhill only.
KOKKOS_INLINE_FUNCTION
bool mc_accepts(RealType dE, RealType kT, RealType u) {
    if (dE <= RealType(0)) return true;
    if (kT <= RealType(0)) return false;
    return u < Kokkos::exp(-dE / kT);
}

// A structure's chain seed, from its seed alone, so a chain does not depend on
// rank count or slot placement. Every slot's RNG state is overwritten with this
// at fill time. splitmix64 finalizer, to decorrelate consecutive seeds.
KOKKOS_INLINE_FUNCTION
uint64_t mc_slot_seed(int structure_seed) {
    uint64_t z = static_cast<uint64_t>(static_cast<uint32_t>(structure_seed))
               + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z == 0 ? 0x1234567ULL : z;
}

inline constexpr double MC_GAS_CONSTANT_KJ = 0.0083144626;   // kJ / (mol K)

// Run the Metropolis chain over `pool`, harvesting each structure's best-seen
// state to host buffers. bws.geom.n_batch must be the full slot capacity, as
// with the trust-region stream: every slot is used and the engine masks the
// empty ones out of the energy pipeline.
template <int MaxRank, typename Space = ExecSpace>
StreamResult batch_optimize_stream_mc(BatchWorkspaceT<Space>& bws,
                                      StreamPoolT<Space>& pool,
                                      const BatchOptimizeSettings& settings,
                                      const McSettings& mc);

extern template StreamResult batch_optimize_stream_mc<0, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
extern template StreamResult batch_optimize_stream_mc<1, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
extern template StreamResult batch_optimize_stream_mc<2, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
extern template StreamResult batch_optimize_stream_mc<3, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);
extern template StreamResult batch_optimize_stream_mc<4, ExecSpace>(
    BatchWorkspace&, StreamPool&, const BatchOptimizeSettings&, const McSettings&);

}  // namespace klasp
