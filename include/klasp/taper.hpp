#pragma once

#include <Kokkos_Core.hpp>
#include "klasp/types.hpp"

namespace klasp {

// Cubic cutoff spline: 1 for r <= r_on, smooth to 0 at r_off, 0 for r > r_off.
// C1 continuous (matching occ DMACRYS SPLI order=3).
// r_on == 0 && r_off == 0 means no taper (always returns 1).
KOKKOS_INLINE_FUNCTION
RealType cutoff_spline(RealType r, RealType r_on, RealType r_off) {
    if (r_on <= 0.0 && r_off <= 0.0) return 1.0;
    if (r <= r_on) return 1.0;
    if (r >= r_off) return 0.0;
    RealType vdr = 1.0 / (r_on - r_off);
    RealType dr = r - r_off;
    return 3.0 * vdr * vdr * dr * dr - 2.0 * vdr * vdr * vdr * dr * dr * dr;
}

KOKKOS_INLINE_FUNCTION
RealType cutoff_spline_deriv(RealType r, RealType r_on, RealType r_off) {
    if (r_on <= 0.0 && r_off <= 0.0) return 0.0;
    if (r <= r_on || r >= r_off) return 0.0;
    RealType vdr = 1.0 / (r_on - r_off);
    RealType dr = r - r_off;
    return 6.0 * vdr * vdr * dr - 6.0 * vdr * vdr * vdr * dr * dr;
}

// Second derivative of the same cubic, for the analytic Hessian.
KOKKOS_INLINE_FUNCTION
RealType cutoff_spline_deriv2(RealType r, RealType r_on, RealType r_off) {
    if (r_on <= 0.0 && r_off <= 0.0) return 0.0;
    if (r <= r_on || r >= r_off) return 0.0;
    RealType vdr = 1.0 / (r_on - r_off);
    RealType dr = r - r_off;
    return 6.0 * vdr * vdr - 12.0 * vdr * vdr * vdr * dr;
}

// Taper window used by the pair terms. With no taper configured (taper_off <= 0),
// ramp the last `taper_width` down to rcut so the objective has no hard cut.
KOKKOS_INLINE_FUNCTION
void effective_taper(RealType taper_on, RealType taper_off, RealType rcut,
                     RealType taper_width, RealType& t_on, RealType& t_off) {
    if (taper_off > RealType(0)) { t_on = taper_on; t_off = taper_off; return; }
    t_off = rcut;
    t_on  = Kokkos::fmax(RealType(0.5) * rcut, rcut - taper_width);
}

// Window that closes the Ewald reciprocal sphere smoothly so k-vectors crossing
// gmax do not step the energy. The cutoff_spline cubic in k^2, templated so the
// Dual6 gradient path differentiates the same expression. The caller handles
// k2 <= k2_on (-> 1) and k2 >= k2_off (-> skip).
template <typename T>
KOKKOS_INLINE_FUNCTION
T ewald_k_window(const T& k2, RealType k2_on, RealType k2_off) {
    const RealType inv = RealType(1) / (k2_on - k2_off);
    const T d  = k2 - T(k2_off);
    const T d2 = d * d;
    return d2 * (RealType(3) * inv * inv) - (d2 * d) * (RealType(2) * inv * inv * inv);
}

// Fraction of gmax^2 at which the window starts to come down.
constexpr double EWALD_K_WINDOW_FRAC = 0.8;

}  // namespace klasp
