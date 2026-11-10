#pragma once

#include <vector>

#include "klasp/types.hpp"

namespace klasp {

// Forward-declared to keep sgdata.hpp out of every device translation unit.
struct SpaceGroupInfo;

// Symmetry operators of one crystal in the layout the expansion kernels index.
// MoleculeData and ModelTablesT derive from it (host and device copies).
// Layout contract:
//   - Rotations are fractional, row-major `(n_symops, 9)`: (r, c) at r*3 + c.
//     Cartesian operators: see build_cartesian_symops in klasp/molecule.hpp.
//   - Translations are fractional in [0, 1), `(n_symops, 3)`.
//   - `n_symops` counts molecule copies. For Z' > 1, entry `oi * n_indep + m`
//     is crystal operator `oi` applied to independent molecule `m`;
//     `sym_indep_index` returns `m`.
//   - The identity comes first (identity_first_order); operator zero is each
//     molecule's reference placement.
//   - `mol_site_count(s)` is the number of real (non-padding) sites of copy s.
template <typename Space = ExecSpace>
struct SymmetryOpsT {
    SView2D<RealType, Space> sym_rotations;    // (n_symops, 9) fractional, row-major
    SView2D<RealType, Space> sym_translations; // (n_symops, 3) fractional
    SView1D<int, Space>      sym_indep_index;  // (n_symops) -> which Z' molecule
    SView1D<int, Space>      mol_site_count;   // (n_symops) real sites of that copy

    int n_indep = 0;    // independent molecules (Z')
    int n_symops = 0;   // molecule copies = crystal operators * n_indep
};

using SymmetryOps     = SymmetryOpsT<ExecSpace>;
using HostSymmetryOps = SymmetryOpsT<HostExecSpace>;

// The identity, then the rest in gemmi's order. The generator and the
// MoleculeData builder must both use this, since they share operator indices.
std::vector<int> identity_first_order(const SpaceGroupInfo& sg);

// Allocate the four tables for up to max_symops copies; counts are left at zero.
template <typename Space>
void allocate_symmetry_ops(SymmetryOpsT<Space>& ops, int max_symops) {
    ops.sym_rotations    = SView2D<RealType, Space>("sym_rotations",    max_symops, 9);
    ops.sym_translations = SView2D<RealType, Space>("sym_translations", max_symops, 3);
    ops.sym_indep_index  = SView1D<int, Space>("sym_indep_index",       max_symops);
    ops.mol_site_count   = SView1D<int, Space>("mol_site_count",        max_symops);
}

// Fill `ops` from a space group: `n_crystal_ops` operators identity-first,
// interleaved molecule-minor over `n_indep` molecules. `site_counts` may be
// shorter than n_indep; `default_sites` fills the rest. Views must be allocated
// for n_crystal_ops * n_indep entries. Sets n_indep and n_symops.
void fill_symmetry_ops(SymmetryOps& ops, const SpaceGroupInfo& sg,
                       int n_crystal_ops, int n_indep,
                       const std::vector<int>& site_counts, int default_sites);

// Crystal operators alone (n_indep == 1, site counts zero), allocated and uploaded.
SymmetryOps crystal_symmetry_ops(const SpaceGroupInfo& sg);

// Mirror `src` into `Space`, counts included.
template <typename Space, typename SrcSpace>
void copy_symmetry_ops(SymmetryOpsT<Space>& dst, const SymmetryOpsT<SrcSpace>& src) {
    using Mem = typename Space::memory_space;
    dst.sym_rotations    = Kokkos::create_mirror_view_and_copy(Mem(), src.sym_rotations);
    dst.sym_translations = Kokkos::create_mirror_view_and_copy(Mem(), src.sym_translations);
    dst.sym_indep_index  = Kokkos::create_mirror_view_and_copy(Mem(), src.sym_indep_index);
    dst.mol_site_count   = Kokkos::create_mirror_view_and_copy(Mem(), src.mol_site_count);
    dst.n_indep  = src.n_indep;
    dst.n_symops = src.n_symops;
}

}  // namespace klasp
