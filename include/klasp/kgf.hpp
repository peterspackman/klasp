#pragma once

#include <Kokkos_Core.hpp>
#include "klasp/types.hpp"

namespace klasp {

// KGF (Kronecker Generalized Fibonacci) quasi-random sequence: integer seed
// and dimension -> point in [0,1)^D. Martin Roberts, "The Unreasonable
// Effectiveness of Quasirandom Sequences" (2018).

// phi(d): the positive root of x^(d+1) = x^d + 1, by fixed-point iteration.
KOKKOS_INLINE_FUNCTION
RealType kgf_phi(int d) {
    RealType x = 2.0;
    for (int i = 0; i < 30; ++i)
        x = Kokkos::pow(1.0 + x, 1.0 / (d + 1.0));
    return x;
}

// alpha[i] = (1/phi)^(i+1) mod 1.
KOKKOS_INLINE_FUNCTION
RealType kgf_alpha(RealType phi, int i) {
    RealType a = 1.0;
    for (int k = 0; k <= i; ++k)
        a /= phi;
    return a - Kokkos::floor(a);
}

// point[i] = (0.5 + alpha[i] * (N + 1)) mod 1
KOKKOS_INLINE_FUNCTION
RealType kgf_component(int seed, int dim_idx, int total_dims) {
    RealType phi = kgf_phi(total_dims);
    RealType alpha = kgf_alpha(phi, dim_idx);
    RealType val = 0.5 + alpha * (seed + 1.0);
    return val - Kokkos::floor(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void kgf_point(int seed, int D, T* point) {
    RealType phi = kgf_phi(D);
    for (int i = 0; i < D; ++i) {
        RealType alpha = kgf_alpha(phi, i);
        RealType val = 0.5 + alpha * (seed + 1.0);
        point[i] = static_cast<T>(val - Kokkos::floor(val));
    }
}

}  // namespace klasp
