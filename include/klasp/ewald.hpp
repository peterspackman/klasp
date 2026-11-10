#pragma once

#include "klasp/types.hpp"
#include <cmath>

namespace klasp {

// Default relative truncation for both Ewald sums.
inline constexpr RealType DEFAULT_EWALD_ACCURACY = RealType(1e-6);

// s = sqrt(-ln(accuracy)). The real-space sum runs to s/kappa and the reciprocal
// sum to gmax = 2*kappa*s. Every site that truncates either sum must use this, so
// box sizing, the pair kernel and the strain gradient agree.
inline RealType ewald_split_width(RealType accuracy) {
    return std::sqrt(-std::log(accuracy));
}

// Compute Ewald kappa, rcut, k-vectors and prefactors into the workspace.
void compute_ewald_params(const RealType cell[3][3], int n_particles,
                          RealType accuracy, Workspace& ws);

}  // namespace klasp
