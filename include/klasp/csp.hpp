#pragma once

#include "klasp/generator.hpp"
#include "klasp/sampling_prior.hpp"   // GeneratorRequest, DofBatch, MAX_INDEP
#include "klasp/types.hpp"
#include "klasp/sgdata.hpp"
#include <vector>
#include <string>

namespace klasp {

enum class CollisionMethod { DIRECT, RIGID_PRESS, FFT_PACK };

template <typename Space> struct BatchWorkspaceT;
using BatchWorkspace = BatchWorkspaceT<ExecSpace>;

// Device-resident structure generation into BatchWorkspace Views. Returns the
// number of valid structures. For co-crystals the body arrays are concatenated per
// component and, when n_indep > 1, mol_n_sites/mol_site_offsets describe the layout.
// Seed per thread idx is (seed_start + idx)*nranks + rank + 1, disjoint across ranks.
int batch_generate_structures(
    int space_group, int seed_start, int n_seeds,
    const View2D<RealType>& body_positions_ang,  // (total_sites, 3) device, Angstrom
    const View1D<RealType>& cov_radii,           // (total_sites) device, Angstrom
    const View1D<RealType>& vdw_radii,           // (total_sites) device, Angstrom
    int n_sites, double tvp, double collision_buffer,
    View2D<RealType> cell_params,      // (batch, 6)
    View3D<RealType> translations,     // (batch, n_indep, 3)
    View3D<RealType> rot_vectors,      // (batch, n_indep, 3)
    View1D<int> valid_flags,           // (batch) 1=valid, 0=invalid
    CollisionMethod collision_method = CollisionMethod::DIRECT,
    int n_indep = 1,
    const View1D<int>& mol_n_sites = View1D<int>(),
    const View1D<int>& mol_site_offsets = View1D<int>(),
    // Per-molecule bounding radius (Angstrom), max of |atom - centroid| + vdw;
    // a coarse pair filter for the multi-component path. Empty disables it.
    const View1D<RealType>& mol_bounding_radius = View1D<RealType>(),
    int rank = 0, int nranks = 1,
    // nudge_budget: per-seed hull-MTV COM-nudge budget (0 disables).
    // contract: shrink the cell to the clash boundary after clearing.
    int nudge_budget = 16, bool contract = true,
    // If set, receives the sampling prior this call drew from.
    SamplingPrior* prior_out = nullptr);

// Generate n_try seeds and compact the valid ones into the output Views on device.
// Returns the number written (<= max_output).
int generate_and_compact_on_device(
    int space_group, int seed_start, int n_try,
    const View2D<RealType>& body_positions_ang,
    const View1D<RealType>& cov_radii,
    const View1D<RealType>& vdw_radii,
    int n_sites, double tvp, double collision_buffer,
    CollisionMethod collision_method,
    View2D<RealType> cell_params_out,
    View3D<RealType> translations_out,
    View3D<RealType> rot_vectors_out,
    View1D<int> seeds_out,
    int max_output,
    int n_indep = 1,
    const View1D<int>& mol_n_sites = View1D<int>(),
    const View1D<int>& mol_site_offsets = View1D<int>(),
    const View1D<RealType>& mol_bounding_radius = View1D<RealType>(),
    int rank = 0, int nranks = 1,
    int nudge_budget = 16, bool contract = true,
    int* raw_valid_out = nullptr,    // if set, the uncapped valid count (>= return)
    SamplingPrior* prior_out = nullptr);   // if set, the prior this call drew from

// The quasi-random generator as a StructureGenerator: basis and clash-resolution
// settings held once, seeds supplied per call.
struct QuasiRandomGenerator {
    View2D<RealType> body_positions_ang;   // (total_sites, 3) device, Angstrom
    View1D<RealType> cov_radii;            // (total_sites) device, Angstrom
    View1D<RealType> vdw_radii;            // (total_sites) device, Angstrom
    int n_sites = 0;
    double tvp = 1.2;
    double collision_buffer = 0.5;
    CollisionMethod collision_method = CollisionMethod::DIRECT;

    // Co-crystal layout; empty Views mean Z'=1 (see batch_generate_structures).
    View1D<int> mol_n_sites;
    View1D<int> mol_site_offsets;
    View1D<RealType> mol_bounding_radius;

    int nudge_budget = 16;
    bool contract = true;

    // raw_valid, if set, receives the uncapped valid count; prior_out, if set,
    // the sampling prior drawn from.
    int generate(const GeneratorRequest& req, DofBatch& out,
                 int* raw_valid = nullptr,
                 SamplingPrior* prior_out = nullptr) const;
};
static_assert(StructureGenerator<QuasiRandomGenerator>);

}  // namespace klasp
