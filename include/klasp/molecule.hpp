#pragma once
#include <string>
#include <stdexcept>
#include <vector>

#include "klasp/symmetry_ops.hpp"
#include "klasp/types.hpp"

namespace klasp {

// Rigid molecule data, stored alongside Workspace, with body-frame data per
// independent molecule. The symmetry half is the host copy of the table
// ModelTablesT carries on the device.
struct MoleculeData : SymmetryOpsT<ExecSpace> {
    SymmetryOpsT<ExecSpace>& symmetry() { return *this; }
    const SymmetryOpsT<ExecSpace>& symmetry() const { return *this; }

    View2D<RealType> translations;      // (max_indep, 3) fractional COMs
    View2D<RealType> rot_vectors;       // (max_indep, 3) angle-axis
    RealType cell_params[6];            // [a, b, c, alpha, beta, gamma]

    // Body-frame atom data, indexed as body_positions(indep_idx * n_sites + site, d).
    View2D<RealType> body_positions;    // (max_indep * max_sites, 3)
    View1D<RealType> body_charges;      // (max_indep * max_sites)
    View1D<int>      body_types;        // (max_indep * max_sites)
    View2D<RealType> body_multipoles;   // (max_indep * max_sites, 35) Cartesian, body frame
    int max_multipole_rank;             // 0 = charge-only, 4 = full

    // Symmetry operations, n_indep and n_symops come from SymmetryOpsT
    // (klasp/symmetry_ops.hpp).

    // The stride of the per-molecule arrays: the largest component's site count.
    // mol_site_count(s) is how many are real.
    int n_sites_per_mol;
};

MoleculeData allocate_molecule_data(int max_indep, int max_sites, int max_symops,
                                    int max_multipole_rank = 0);

// Apply the symmetry operations to generate all Z molecules. Fills
// ws.positions, ws.charges, ws.types, ws.cell_matrix, ws.n_atoms, and for
// MaxRank > 0 rotates body_multipoles into ws.multipoles (MaxRank 4 = full).
template <int MaxRank = 0>
void expand_asymmetric_unit(const MoleculeData& mol, Workspace& ws);

extern template void expand_asymmetric_unit<0>(const MoleculeData& mol, Workspace& ws);
extern template void expand_asymmetric_unit<4>(const MoleculeData& mol, Workspace& ws);

// The symmetry operators as Cartesian rotations, S_cart = cell S_frac cell^-1,
// flattened (n_symops, 9) like sym_rotations. Use these for Cartesian data
// (body coordinates, multipoles); the fractional ones are rotations only in
// cells where they are signed permutations.
View2D<RealType> build_cartesian_symops(const MoleculeData& mol);

// As above, with the cell supplied explicitly rather than from mol.cell_params.
View2D<RealType> build_cartesian_symops(const MoleculeData& mol,
                                        const RealType cell[3][3]);


// Every independent molecule must have the same number of symmetry copies: the
// asymmetric-unit pair enumeration needs it for the energy and gradient to agree.
inline void validate_symmetry_multiplicity(const MoleculeData& mol) {
    if (mol.n_indep <= 1) return;
    auto h_si = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    mol.sym_indep_index);
    std::vector<int> mult(mol.n_indep, 0);
    for (int op = 0; op < mol.n_symops; ++op) {
        const int k = h_si(op);
        if (k >= 0 && k < mol.n_indep) ++mult[k];
    }
    for (int k = 1; k < mol.n_indep; ++k) {
        if (mult[k] == mult[0]) continue;
        std::string counts;
        for (int i = 0; i < mol.n_indep; ++i)
            counts += (i ? ", " : "") + std::to_string(mult[i]);
        throw std::runtime_error(
            "symmetry: the independent molecules have unequal numbers of copies ("
            + counts + "). The pair enumeration assumes they are equal, and with "
            "unequal counts the gradient stops matching the energy it belongs to. "
            "A molecule on a special position needs its own asymmetric unit, or "
            "the structure needs expanding to P1.");
    }
}

}  // namespace klasp
