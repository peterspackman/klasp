#pragma once
// Cell volume ceiling for sampling, shared by every generator so `--tvp` means
// the same thing on each path. Molecular volume is the vdW-inflated bounding box
// (not a sum of atomic spheres, which undercounts).

#include "klasp/sgdata.hpp"   // LatticeType

#include <algorithm>
#include <cmath>

namespace klasp {

struct VolumeBudget {
    double max_volume = 0;       // A^3: the ceiling a drawn cell must fit under
    double sum_mol_vol = 0;      // A^3: before n_symops, tvp and the floor
    double max_vdw = 0;          // A: largest vdW radius over all sites
    double max_mol_radius = 0;   // A: largest component bounding sphere
    int total_real_sites = 0;
    int max_sites_per_mol = 0;
};

// `pos` is (n_sites, 3) row-major Angstrom in the input orientation; `vdw` is
// (n_sites) Angstrom. `mol_n_sites` and `mol_site_offsets` locate the independent
// molecules in those arrays.
inline VolumeBudget volume_budget(const double* pos, const double* vdw,
                                  const int* mol_n_sites, const int* mol_site_offsets,
                                  int n_indep, int n_symops, LatticeType lattice,
                                  double tvp) {
    VolumeBudget b;
    for (int m = 0; m < n_indep; ++m) {
        const int ns_m = mol_n_sites[m];
        const int off = mol_site_offsets[m];
        b.total_real_sites += ns_m;
        b.max_sites_per_mol = std::max(b.max_sites_per_mol, ns_m);
        if (ns_m <= 0) continue;

        double cen[3] = {0, 0, 0};
        for (int i = 0; i < ns_m; ++i)
            for (int d = 0; d < 3; ++d) cen[d] += pos[(off + i) * 3 + d];
        for (int d = 0; d < 3; ++d) cen[d] /= ns_m;

        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int i = 0; i < ns_m; ++i) {
            const double v = vdw[off + i];
            b.max_vdw = std::max(b.max_vdw, v);
            double r2 = 0;
            for (int d = 0; d < 3; ++d) {
                const double p = pos[(off + i) * 3 + d];
                const double q = p - cen[d];
                r2 += q * q;
                lo[d] = std::min(lo[d], p - v);
                hi[d] = std::max(hi[d], p + v);
            }
            // Bounding sphere, valid in any orientation.
            b.max_mol_radius = std::max(b.max_mol_radius, std::sqrt(r2) + v);
        }
        b.sum_mol_vol += (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
    }

    b.max_volume = b.sum_mol_vol * n_symops * tvp;

    // Rotation-invariant floor of ~(2R)^3 per molecule, since the input-orientation
    // box understates a rotated molecule. Applied only to lattices whose final
    // density comes from the later contract pass.
    if (lattice == TRICLINIC || lattice == MONOCLINIC || lattice == ORTHORHOMBIC) {
        const double two_r = 2.0 * b.max_mol_radius;
        b.max_volume = std::max(b.max_volume, two_r * two_r * two_r * tvp);
    }
    return b;
}

}  // namespace klasp
