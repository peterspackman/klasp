#pragma once

#include "klasp/types.hpp"
#include <Kokkos_Core.hpp>

namespace klasp {

// Map flat upper-triangle index ij in [0, n*(n+1)/2) to (i, j) with j >= i.
KOKKOS_INLINE_FUNCTION
void flat_to_upper_triangle(int ij, int n, int& i, int& j) {
    // Row i starts at i*n - i*(i-1)/2 and has n-i elements; invert with
    // i = n - 1 - floor((sqrt(1 + 8*(N - 1 - ij)) - 1) / 2), N = n*(n+1)/2.
    int N = n * (n + 1) / 2;
    RealType d = Kokkos::sqrt(1.0 + 8.0 * static_cast<RealType>(N - 1 - ij));
    i = n - 1 - static_cast<int>((d - 1.0) * 0.5);
    // Guard against floating-point rounding.
    if (i < 0) i = 0;
    int row_start = i * n - i * (i - 1) / 2;
    j = i + (ij - row_start);
}

// True if the (i, j, shift s) pair is a self-pair or an intramolecular pair at
// zero shift. shifts: (n_shifts, 3) integer lattice shifts; mol_id: (n_atoms,)
// molecule membership, -1 = no exclusion.
template <typename ShiftView, typename MolView>
KOKKOS_INLINE_FUNCTION
bool should_skip_pair(int i, int j,
                      ShiftView const& shifts, int s,
                      MolView const& mol_id) {
    const int sa = shifts(s, 0), sb = shifts(s, 1), sc = shifts(s, 2);

    if (i == j) {
        if (sa == 0 && sb == 0 && sc == 0) return true;
        if (sa < 0) return true;
        if (sa == 0 && sb < 0) return true;
        if (sa == 0 && sb == 0 && sc < 0) return true;
    }

    if (mol_id(i) >= 0 && mol_id(i) == mol_id(j)) {
        if (sa == 0 && sb == 0 && sc == 0) return true;
    }

    return false;
}

// Batch variant: mol_id has shape (batch, n_atoms), indexed by batch b.
template <typename ShiftView, typename MolView>
KOKKOS_INLINE_FUNCTION
bool should_skip_pair(int i, int j,
                      ShiftView const& shifts, int s,
                      MolView const& mol_id, int b) {
    const int sa = shifts(s, 0), sb = shifts(s, 1), sc = shifts(s, 2);

    if (i == j) {
        if (sa == 0 && sb == 0 && sc == 0) return true;
        if (sa < 0) return true;
        if (sa == 0 && sb < 0) return true;
        if (sa == 0 && sb == 0 && sc < 0) return true;
    }

    if (mol_id(b, i) >= 0 && mol_id(b, i) == mol_id(b, j)) {
        if (sa == 0 && sb == 0 && sc == 0) return true;
    }

    return false;
}

// Variant taking integer shift components directly (no shift view).
template <typename MolView>
KOKKOS_INLINE_FUNCTION
bool should_skip_pair(int i, int j,
                      int sa, int sb, int sc,
                      MolView const& mol_id, int b) {
    if (i == j) {
        if (sa == 0 && sb == 0 && sc == 0) return true;
        if (sa < 0) return true;
        if (sa == 0 && sb < 0) return true;
        if (sa == 0 && sb == 0 && sc < 0) return true;
    }

    if (mol_id(b, i) >= 0 && mol_id(b, i) == mol_id(b, j)) {
        if (sa == 0 && sb == 0 && sc == 0) return true;
    }

    return false;
}

}  // namespace klasp
