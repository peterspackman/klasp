#pragma once
// Descriptors for explicit-atom geometries that are not rigid-molecule packings.
// Every atom in the cell is a row (n_indep = 1, n_sites_per_mol = n_atoms).

#include "klasp/dedup.hpp"   // AMD_K
#include "klasp/descriptors.hpp"
#include "klasp/geometry.hpp"
#include "klasp/powder.hpp"

#include <array>
#include <vector>

namespace klasp {

struct FlexDescriptors {
    std::vector<std::vector<double>> amd;     // [n][k]
    std::vector<std::vector<float>> powder;   // [n][POWDER_N_BINS]
    std::vector<std::vector<double>> pdd;     // [n][n_asym*k], empty unless asked
    int n_asym = 0;
};

// Descriptors for a batch of explicit-atom structures. All structures must have
// the same atom count and element order.
//
// The neighbour search radius is derived from the atom density so the sphere
// holds comfortably more than k atoms; a positive cutoff_bohr overrides it.
//
// With fold_rows > 0 the PDD has one row per asymmetric-unit atom, the mean over
// its copies, with atom i taken as a copy of atom i % fold_rows (operator, then
// molecule, then site order). A cell whose atom count is not a multiple gets no PDD.
FlexDescriptors compute_flex_descriptors(const std::vector<CrystalGeometry>& geoms,
                                         bool want_pdd = false,
                                         double cutoff_bohr = 0.0,
                                         int k = AMD_K,
                                         int fold_rows = 0);

}  // namespace klasp
