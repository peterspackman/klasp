#pragma once
// FFT-based crystal structure generator: the crystallographic fast translation
// function (Crowther & Blow) with a hard-sphere overlap kernel. At fixed cell and
// orientation, the overlap with symmetry image s = (W_s, w_s) is
//     O_s(x) = (1/V) SUM_h f(h) conj(f_s(h)) exp(2 pi i h.w_s) exp(2 pi i h.M_s x)
// with M_s = W_s - I an integer matrix, so one inverse FFT per operation gives
// the whole translation landscape on the grid.

#include "klasp/generator.hpp"   // GeneratorRequest, DofBatch
#include "klasp/types.hpp"
#include "klasp/sgdata.hpp"

#include <vector>

namespace klasp {

struct FFTGeneratorConfig {
    double grid_spacing = 0.5;      // Angstrom -- FFT grid resolution
    // Positions extracted per (cell, orientation) trial; also the identity block
    // size each trial reserves (see fft_identity).
    int k_per_trial = 8;
    double collision_tol = 0.1;     // overlap tolerance (A^3) for the FFT pre-filter
    double collision_buffer = 0.5;  // buffer on covalent radii for the direct check (Angstrom)
    double tvp = 1.2;               // target volume parameter
    double max_aspect = 5.0;        // max cell aspect ratio
};

// Structure identity for a generator that emits several structures per draw:
// trial t owns identities [t*k, t*k + k). Identities must be unique (they are the
// database key) and k must match on --resume.
inline int fft_identity(int trial_seed, int slot, int k_per_trial) {
    return trial_seed * k_per_trial + slot;
}
inline int fft_trial_of(int identity, int k_per_trial) {
    return identity / k_per_trial;
}

// Generate structures using FFT overlap detection. Each quasi-random trial takes
// cell and orientation from the shared SamplingPrior and finds up to k_per_trial
// non-colliding positions, verified by a direct collision check.
// Returns the number of valid structures written (<= max_output).
int fft_generate_batch(
    int space_group,
    int seed_start,            // first quasi-random seed
    int n_seeds,               // number of (cell, orientation) trials
    const FFTGeneratorConfig& config,
    const View2D<RealType>& body_positions_ang,  // (n_sites, 3) device, Angstrom, COM-centered
    const View1D<RealType>& cov_radii,           // (n_sites) device, Angstrom
    const View1D<RealType>& vdw_radii,           // (n_sites) device, Angstrom
    int n_sites,
    View2D<RealType> cell_params_out,      // (batch, 6) Bohr + degrees
    View3D<RealType> translations_out,     // (batch, 1, 3) fractional
    View3D<RealType> rot_vectors_out,      // (batch, 1, 3) angle-axis
    View1D<int> seeds_out,                 // (batch) seed per structure
    int max_output,
    int rank = 0, int nranks = 1);

// Symmetry-overlap map in Angstrom^3 on an (nx, ny, nz) row-major grid of
// fractional centres x: overlap with every non-identity symmetry image and its
// lattice translates. `radius_buffer` is added to every covalent radius.
void fft_overlap_map(const SpaceGroupInfo& sg, int id_op,
                     const double cell[3][3], const double R_mol[3][3],
                     const View2D<RealType>& body_positions_ang,
                     const View1D<RealType>& cov_radii,
                     int n_sites, double radius_buffer,
                     int nx, int ny, int nz,
                     std::vector<double>& out);

// The FFT position finder as a StructureGenerator. One seed yields up to
// k_per_trial rows, so `n_try` bounds the yield rather than sizing it.
// Z'=1 only: out.n_indep must be 1.
struct FftGenerator {
    FFTGeneratorConfig config;
    View2D<RealType> body_positions_ang;   // (n_sites, 3) device, Angstrom
    View1D<RealType> cov_radii;            // (n_sites) device, Angstrom
    View1D<RealType> vdw_radii;            // (n_sites) device, Angstrom -- the volume budget
    int n_sites = 0;

    int generate(const GeneratorRequest& req, DofBatch& out) const;
};
static_assert(StructureGenerator<FftGenerator>);

}  // namespace klasp
