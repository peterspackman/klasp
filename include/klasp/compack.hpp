#pragma once
// COMPACK-style packing comparison: overlay coordination clusters (a central
// molecule and its nearest neighbours) and match neighbours by per-molecule RMSD.
// Every molecule is the same rigid body, so atom correspondence is the identity
// and the per-molecule RMSD has the closed form
//     RMSD^2 = |dc|^2 + (2/m) * (tr G - tr(M_a^T M_b Q G))
// with G the body-frame gyration tensor and Q over the body's own symmetry
// operations (improper ones included, so planar bodies match across mirror images).

#include "klasp/compack_kernel.hpp"
#include "klasp/dedup.hpp"
#include "klasp/io.hpp"
#include "klasp/types.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace klasp {

// One rigid molecule placed in a cell.
struct PlacedMolecule {
    int component = 0;      // which basis component; always 0 for one molecule type
    double centroid[3] = {0, 0, 0};
    // Body -> cell orientation, row-major 3x3. Orthogonal; det may be -1.
    double orient[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

// Invariant description of one basis component. `m` and `gyration` cover only
// the counted atoms (heavy-only or all).
struct RigidBody {
    int m = 0;                        // atoms counted
    double gyration[9] = {0};         // G = sum_k p_k p_k^T, body frame, row-major
    double trace_g = 0.0;
    // The body's own symmetry operations, row-major 3x3, identity first,
    // improper operations included.
    std::vector<double> self_ops;     // [n_ops * 9]
    // (Q G)^T for each self op; filled by detect_self_symmetry.
    std::vector<double> qgt;          // [n_ops * 9]
    // Body-frame positions (counted centroid at the origin) and elements.
    std::vector<double> positions;    // [m * 3]
    std::vector<int> numbers;         // [m]
    // Offset from the caller's frame origin (COM of all sites) to the counted
    // centroid; place_molecule applies it.
    double origin_offset[3] = {0, 0, 0};

    int n_self_ops() const { return static_cast<int>(self_ops.size() / 9); }
};

// A central molecule and the molecules packed around it, nearest first.
struct PackingCluster {
    PlacedMolecule central;
    std::vector<PlacedMolecule> neighbours;
    // neighbours[0, untied) must all be matched; the rest are tied with the Nth
    // and fill places up to `scored`. untied < 0: all must be matched.
    int untied = -1;
    int scored = 0;
};

// Nothing here converts units: tolerance, radius, body positions and cell
// parameters must share one unit, and RMSD is returned in it. Stored klasp cells
// are in Bohr; COMPACK tolerances are in Angstrom.
struct CompackOptions {
    // Reference cluster size: the central molecule's N nearest neighbours.
    // COMPACK's default is 15.
    int n_neighbours = 15;
    // Per-molecule RMSD below which a neighbour counts as matched.
    double match_tolerance = 1.0;
    int max_iterations = 8;
    // H positions from X-ray work are unreliable.
    bool heavy_only = true;
    // Also try an improper overlay. Valid only for centrosymmetric crystals.
    bool allow_inversion = false;
};

struct PackingMatch {
    int reference = -1;   // index into the reference cluster's neighbours
    // Index into the candidate pool: the counterpart when `matched`, else the
    // nearest candidate.
    int candidate = -1;
    double rmsd = 0.0;    // per-molecule RMSD in the overlaid frame
    bool matched = false;
};

struct CompackResult {
    double rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // maps A onto B, row-major
    double translation[3] = {0, 0, 0};
    std::vector<PackingMatch> matches;
    int n_matched = 0;
    int n_neighbours = 0;
    bool same_packing = false;   // every reference neighbour found a counterpart
    // RMSD over the central molecule plus matched neighbours, in input units.
    // Only meaningful at a full match.
    double rmsd = 0.0;
    bool inverted = false;       // the overlay needed an improper transform
    bool valid = false;          // false when there was nothing to compare
};

// --- independently testable pieces ------------------------------------------

// `positions` is [n*3] in any frame; the centroid is removed. Self-symmetry is
// left as the identity; call detect_self_symmetry to fill it in.
RigidBody make_rigid_body(const double* positions, const int* numbers, int n,
                          bool heavy_only);

// False when the heavy atoms are collinear but the molecule is not (e.g.
// ethylene): heavy atoms alone cannot distinguish rotations about that line.
bool heavy_atoms_suffice(const double* positions, const int* numbers, int n);

// PDD row blocks for dedup: one per molecule of the asymmetric unit, with the
// atom permutations its symmetry operations make, found at `tol` Angstrom.
std::vector<DedupConfig::PddBlock> pdd_symmetry_blocks(const BasisConfig& basis,
                                                       double tol = 0.05);

// Find the body's own symmetry operations among its counted atoms and store them
// in `body.self_ops`. An operation is accepted when it maps every atom onto one of
// the same element within `tol` Angstrom.
void detect_self_symmetry(RigidBody& body, double tol = 0.05);

// Closed-form RMSD between two placements of the same rigid body, minimised over
// the body's own symmetry operations. Exact, not an approximation.
double molecule_rmsd(const RigidBody& body, const PlacedMolecule& a,
                     const PlacedMolecule& b);

// Optimal transform mapping `from` onto `onto` (both [n*3] row-major), by SVD.
// When `allow_improper` is false the result is a proper rotation; otherwise a
// reflection is accepted if it fits better. Returns the RMSD.
double kabsch(const double* from, const double* onto, int n,
              double rotation[9], double translation[3], bool allow_improper);

// Host-only view: pointers address the RigidBody's own vectors. Use DeviceBody
// for a kernel launch.
compack::BodyView body_view(const RigidBody& body);

// The body's tables in ExecSpace memory, with a view safe to capture in a kernel.
class DeviceBody {
public:
    explicit DeviceBody(const RigidBody& body);

    // Never aliases the RigidBody's buffers, on any backend.
    compack::BodyView view() const;

private:
    compack::BodyView scalars_;    // m, n_ops, gyration, trace_g; pointers unset
    Kokkos::View<double*, ExecSpace> ops_;
    Kokkos::View<double*, ExecSpace> qgt_;
};

// Compare two clusters. `body` describes the component both are built from.
// Cluster `b` should carry a larger neighbour pool than opts.n_neighbours (see
// candidate_pool_size) so a counterpart just outside B's Nth-closest still matches.
CompackResult compack_compare(const PackingCluster& a, const PackingCluster& b,
                              const RigidBody& body, const CompackOptions& opts);

// How many neighbours the candidate cluster should offer when the reference has n.
inline int candidate_pool_size(int n) { return n + (n / 2 > 8 ? n / 2 : 8); }

// Reference neighbour count and pair options for the kernel: the N nearest plus
// any shell tied with the Nth, scored out of N.
int kernel_reference_count(const PackingCluster& a, const CompackOptions& opts);
compack::PairOptions kernel_pair_options(const PackingCluster& a, const CompackOptions& opts);

// --- clusters from a stored structure ---------------------------------------

// One stored row's rigid-body geometry.
struct StructureGeometry {
    double cell_params[6] = {0, 0, 0, 0, 0, 0};
    int space_group = 1;
    int n_mols = 1;
    const double* translations = nullptr;  // [n_mols*3] fractional COM per molecule
    const double* rotations = nullptr;     // [n_mols*3] angle-axis per molecule
};

struct ClusterSpec {
    int n_neighbours = 15;   // how many neighbours to keep, nearest first
    double radius = 14.0;    // centroid-to-centroid search radius
};

// Place one molecule, shifting the stored COM by the body's counted-atom offset.
PlacedMolecule place_molecule(const RigidBody& body, const double cart_com[3],
                              const double orient[9], int component);

// Coordination cluster around independent molecule `central` of a stored
// structure, including periodic images. `bodies` is indexed by component.
// Returns an empty cluster when the cell is degenerate or nothing is in range.
PackingCluster build_cluster(const StructureGeometry& g,
                             const std::vector<RigidBody>& bodies,
                             int central, const ClusterSpec& spec);

// Coordination cluster around `central`, given the molecules of one unit cell.
// `cell` is row-major 3x3 with lattice vectors as columns. Returns an empty
// cluster for a degenerate cell.
PackingCluster build_cluster_from_cell(const double cell[9],
                                       const std::vector<PlacedMolecule>& cell_mols,
                                       int central, const ClusterSpec& spec);

// Compare two stored structures, trying each of A's independent molecules as the
// cluster centre and keeping the best; stops at the first full match.
CompackResult compare_structures(const StructureGeometry& a, const StructureGeometry& b,
                                 const std::vector<RigidBody>& bodies,
                                 const CompackOptions& opts,
                                 const ClusterSpec& spec = {});

// --- all-pairs sweep ---------------------------------------------------------

// Compare every pair among `n` structures, one team per pair. `pools` carries the
// larger candidate set. Fills `n_matched` ([n*n], symmetric, diagonal =
// n_neighbours) and `same` ([n*n], 1 when every reference neighbour matched).
struct SweepClusters {
    int n = 0;                                // structures
    int n_ref = 0;                            // reference neighbours per structure
    int n_pool = 0;                           // candidate neighbours per structure
    std::vector<PlacedMolecule> centrals;     // [n]
    std::vector<PlacedMolecule> ref;          // [n * n_ref]; n_ref is the stride
    std::vector<int> ref_count;               // [n] references each holds, tied shell included
    std::vector<int> untied;                  // [n] start of its tied shell, -1 if none
    std::vector<PlacedMolecule> pool;         // [n * n_pool]
};

// `progress`, when set, is called on the host with (pairs done, pairs total);
// the launch is then split into chunks.
void compack_sweep(const SweepClusters& c, const RigidBody& body,
                   const CompackOptions& opts,
                   std::vector<int>& n_matched, std::vector<unsigned char>& same,
                   const std::function<void(long long, long long)>& progress = {});

}  // namespace klasp
