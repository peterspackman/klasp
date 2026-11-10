#pragma once
// Niggli reduction of a lattice basis, so cutoff-sphere image counts stay
// bounded when an optimizer drifts toward a nearly coplanar basis.

#include "klasp/types.hpp"

namespace klasp {

// Result of reducing a basis. `transform` is the unimodular integer matrix P
// with reduced = original * P (lattice vectors as columns), so anything written
// in the old basis follows: a fractional coordinate maps as f' = P^-1 f, which
// leaves the Cartesian position unchanged.
struct NiggliResult {
    RealType cell[3][3];      // reduced basis, lattice vectors as columns
    int transform[3][3];      // P, with reduced = original * P
    int iterations = 0;       // steps taken
    bool converged = false;   // false if the step cap was hit; cell is then the input
};

// Reduce `cell` (lattice vectors as columns, Bohr). Volume and lattice are
// preserved exactly. eps is the relative tolerance on the metric comparisons;
// 0 scales it with the cell size (Grosse-Kunstleve et al.).
NiggliResult niggli_reduce(const RealType cell[3][3], RealType eps = 0);

// Convenience form over cell parameters (a, b, c in Bohr; angles in degrees).
// Returns the reduced parameters, with the transform in `transform`.
bool niggli_reduce_params(const RealType params_in[6], RealType params_out[6],
                          int transform[3][3]);

// Map a fractional coordinate through a reduction transform: f' = P^-1 f.
// P is unimodular so the inverse is exact in integers.
void niggli_map_fractional(const int transform[3][3], const double frac_in[3],
                           double frac_out[3]);

}  // namespace klasp
