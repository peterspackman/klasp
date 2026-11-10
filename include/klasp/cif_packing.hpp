#pragma once
// CIF to packing comparison: match each CIF molecule's atoms to a reference rigid
// body (sorted interatomic-distance descriptors, minimum-cost assignment, then
// overlay-and-reassign), and Kabsch-fit its placement, accepting improper fits.
// Lengths are Angstrom.

#include "klasp/cif_reader.hpp"
#include "klasp/compack.hpp"

#include <string>
#include <vector>

namespace klasp {

struct BodyFit {
    bool ok = false;
    double rmsd = 0.0;              // Angstrom, over the counted atoms
    // Body atom k is molecule atom permutation[k], over the counted atoms of each.
    std::vector<int> permutation;
    PlacedMolecule placed;
};

// Fit `n` atoms onto `body`. `heavy_only` must match how the body was built.
BodyFit fit_to_body(const RigidBody& body, const double* positions, const int* numbers,
                    int n, bool heavy_only, int max_iterations = 8);

// Reference rigid body from one CIF molecule, with its self-symmetry detected.
// `self_symmetry_tol` is passed to detect_self_symmetry.
RigidBody body_from_molecule(const CifMolecule& m, bool heavy_only,
                             double self_symmetry_tol = 0.05);

// One unit cell of rigid placements.
struct CifPacking {
    std::string name;                     // data block
    std::string source;                   // file
    double cell[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // lattice vectors are the columns
    int space_group = 0;
    std::string space_group_name;
    std::vector<PlacedMolecule> molecules;   // every molecule of one unit cell

    // One molecule per distinct environment, as indices into `molecules`. Set by
    // find_representatives; until then every molecule is its own.
    std::vector<int> representatives;

    double worst_fit = 0.0;    // largest per-molecule fit RMSD, Angstrom
    bool valid = false;
    std::string problem;       // why not, when invalid
};

// Place every molecule of `s` against `body`. Fails, with `problem` set, when a
// molecule cannot be fitted or its residual exceeds `max_fit_rmsd`.
CifPacking place_cif_structure(const CifStructure& s, const RigidBody& body,
                               bool heavy_only, double max_fit_rmsd = 1.0);

// Keep one molecule index per distinct environment in `p.representatives`,
// by comparing coordination clusters. Needed because a CIF (e.g. P1) may not
// record the symmetry relating its molecules.
void find_representatives(CifPacking& p, const RigidBody& body,
                          const CompackOptions& opts, const ClusterSpec& spec = {});

// Compare two CIF packings, centring on each of A's representatives.
CompackResult compare_packings(const CifPacking& a, const CifPacking& b,
                               const RigidBody& body, const CompackOptions& opts,
                               const ClusterSpec& spec = {});

// All pairs among `packings` (each valid and through find_representatives).
// Fills `n_matched` and `same` ([n*n], symmetric). Returns how many structures
// found fewer neighbours than asked within spec.radius.
int compare_all_packings(const std::vector<CifPacking>& packings, const RigidBody& body,
                         const CompackOptions& opts, const ClusterSpec& spec,
                         std::vector<int>& n_matched, std::vector<unsigned char>& same,
                         const std::function<void(long long, long long)>& progress = {});

}  // namespace klasp
