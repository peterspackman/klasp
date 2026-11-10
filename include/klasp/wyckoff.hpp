#pragma once
// Special positions, derived rather than tabulated. The site symmetry of a point
// x is {(S,w) : S x + w == x (mod 1)}; multiplicity, free directions and coset
// representatives follow from that set and the gemmi space group.

#include "klasp/sgdata.hpp"

#include <array>
#include <string>
#include <vector>

namespace klasp {

struct WyckoffSite {
    // A representative position, and the subspace through it that shares the
    // same site symmetry. free_basis holds n_free fractional direction vectors;
    // a position on the site is rep + sum(t_i * free_basis[i]).
    double rep[3] = {0, 0, 0};
    int n_free = 0;                  // 0 = a point, 1 = a line, 2 = a plane
    double free_basis[3][3] = {};

    int site_order = 1;              // |site symmetry group|
    int multiplicity = 1;            // n_symops / site_order = copies in the cell

    std::vector<int> site_ops;       // indices into SpaceGroupInfo::symops
    std::vector<int> coset_reps;     // one op per distinct copy; size == multiplicity
};

// The operations of `sg` that fix `frac` modulo a lattice translation.
std::vector<int> site_symmetry_ops(const SpaceGroupInfo& sg, const double frac[3],
                                   double tol = 1e-6);

// Coset representatives for a site symmetry group: one operation per distinct
// image of the site, so each copy is produced exactly once.
std::vector<int> coset_representatives(const SpaceGroupInfo& sg, const double frac[3],
                                       double tol = 1e-6);

// Every distinct special position of a space group, most symmetric first,
// excluding the general position. Found by scanning a 1/12 grid, which contains
// every special position since site symmetry operation orders divide 6.
std::vector<WyckoffSite> special_positions(const SpaceGroupInfo& sg, double tol = 1e-6);

// A cell of the right shape for a lattice type, with arbitrary free ratios. A
// symmetry operation's Cartesian form depends only on the lattice system.
void canonical_cell(LatticeType lattice, double cell[3][3]);

// The site's symmetry operations as Cartesian orthogonal matrices (row-major
// 3x3, one after another), which is the form the molecule's own operations come
// in from detect_self_symmetry. The fractional rotation R becomes A R A^-1.
std::vector<double> site_ops_cartesian(const SpaceGroupInfo& sg,
                                       const WyckoffSite& site,
                                       const double cell[3][3]);

// Orientations R in which a molecule with symmetry operations `mol_ops` is
// invariant under every operation of `site_ops_cart` (R^-1 W R in M). Empty when
// the molecule cannot sit on the site. All matrices are row-major 3x3 laid end
// to end. `degrees_of_freedom`: 0 pinned, 1 free about an axis, 2 unconstrained.
std::vector<double> orientations_on_site(const std::vector<double>& site_ops_cart,
                                         const std::vector<double>& mol_ops,
                                         int* degrees_of_freedom = nullptr,
                                         double tol = 1e-4);

// --- reporting --------------------------------------------------------------

struct WyckoffReport {
    struct Entry {
        int site_order = 0;
        int multiplicity = 0;
        int n_free = 0;              // translational freedom left on the site
        int n_orientations = 0;      // 0 means the molecule does not fit
        int orientation_dof = 0;     // rotational freedom left, 0 to 2
        double rep[3] = {0, 0, 0};
    };
    int space_group = 0;
    int n_symops = 0;
    int mol_n_ops = 0;               // the molecule's own point group order
    std::vector<Entry> entries;      // every special position, most symmetric first
};

// Which of a space group's special positions the molecule can occupy, and what
// Z each would give. `mol_ops` is the molecule's own symmetry operations as
// row-major Cartesian 3x3 matrices -- what detect_self_symmetry produces.
WyckoffReport wyckoff_report(int space_group, const std::vector<double>& mol_ops);

struct WyckoffOptions {
    std::string basis_path;
    std::string space_groups = "1-230";
    bool all_sites = false;          // also list the sites the molecule cannot use
    // Stricter than compack's tolerance: a false symmetry here places a molecule
    // on a site it does not fit. See detect_self_symmetry.
    double self_symmetry_tolerance = 0.05;
};

int cmd_wyckoff(const WyckoffOptions& o);

}  // namespace klasp
