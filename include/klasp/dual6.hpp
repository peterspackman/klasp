#pragma once
// Dual6: forward-mode AD carrying value + 6 strain derivatives.

#include <Kokkos_Core.hpp>
#include <cmath>
#include "klasp/types.hpp"

namespace klasp {

struct Dual6 {
    RealType v;      // value
    RealType g[6];   // gradient w.r.t. 6 Voigt strain components

    KOKKOS_INLINE_FUNCTION Dual6() : v(0) { for (int i = 0; i < 6; ++i) g[i] = 0; }
    KOKKOS_INLINE_FUNCTION explicit Dual6(RealType val) : v(val) { for (int i = 0; i < 6; ++i) g[i] = 0; }
};

KOKKOS_INLINE_FUNCTION Dual6 d6_const(RealType c) { return Dual6(c); }

KOKKOS_INLINE_FUNCTION Dual6 operator+(const Dual6& a, const Dual6& b) {
    Dual6 r; r.v = a.v + b.v;
    for (int i = 0; i < 6; ++i) r.g[i] = a.g[i] + b.g[i];
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 operator-(const Dual6& a, const Dual6& b) {
    Dual6 r; r.v = a.v - b.v;
    for (int i = 0; i < 6; ++i) r.g[i] = a.g[i] - b.g[i];
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 operator-(const Dual6& a) {
    Dual6 r; r.v = -a.v;
    for (int i = 0; i < 6; ++i) r.g[i] = -a.g[i];
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 operator*(const Dual6& a, const Dual6& b) {
    Dual6 r; r.v = a.v * b.v;
    for (int i = 0; i < 6; ++i) r.g[i] = a.g[i] * b.v + b.g[i] * a.v;
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 operator*(const Dual6& a, RealType s) {
    Dual6 r; r.v = a.v * s;
    for (int i = 0; i < 6; ++i) r.g[i] = a.g[i] * s;
    return r;
}
KOKKOS_INLINE_FUNCTION Dual6 operator*(RealType s, const Dual6& a) { return a * s; }

KOKKOS_INLINE_FUNCTION Dual6 operator/(const Dual6& a, const Dual6& b) {
    Dual6 r;
    RealType inv = RealType(1) / b.v;
    r.v = a.v * inv;
    for (int i = 0; i < 6; ++i) r.g[i] = (a.g[i] * b.v - b.g[i] * a.v) * (inv * inv);
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 operator/(const Dual6& a, RealType s) {
    return a * (RealType(1) / s);
}

KOKKOS_INLINE_FUNCTION Dual6& operator+=(Dual6& a, const Dual6& b) { a = a + b; return a; }
KOKKOS_INLINE_FUNCTION Dual6& operator-=(Dual6& a, const Dual6& b) { a = a - b; return a; }

KOKKOS_INLINE_FUNCTION Dual6 d6_sqrt(const Dual6& a) {
    Dual6 r;
    r.v = Kokkos::sqrt(a.v);
    RealType fp = RealType(0.5) / r.v;
    for (int i = 0; i < 6; ++i) r.g[i] = fp * a.g[i];
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 d6_exp(const Dual6& a) {
    Dual6 r;
    r.v = Kokkos::exp(a.v);
    for (int i = 0; i < 6; ++i) r.g[i] = r.v * a.g[i];
    return r;
}

KOKKOS_INLINE_FUNCTION void d6_sincos(const Dual6& a, Dual6& s, Dual6& c) {
    RealType sv, cv;
    sv = Kokkos::sin(a.v);
    cv = Kokkos::cos(a.v);
    s.v = sv; c.v = cv;
    for (int i = 0; i < 6; ++i) {
        s.g[i] = cv * a.g[i];
        c.g[i] = -sv * a.g[i];
    }
}

KOKKOS_INLINE_FUNCTION Dual6 d6_erfc(const Dual6& a) {
    Dual6 r;
    r.v = Kokkos::erfc(a.v);
    RealType fp = RealType(-2) / Kokkos::sqrt(RealType(M_PI)) * Kokkos::exp(-a.v * a.v);
    for (int i = 0; i < 6; ++i) r.g[i] = fp * a.g[i];
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 d6_erf(const Dual6& a) {
    Dual6 r;
    r.v = Kokkos::erf(a.v);
    RealType fp = RealType(2) / Kokkos::sqrt(RealType(M_PI)) * Kokkos::exp(-a.v * a.v);
    for (int i = 0; i < 6; ++i) r.g[i] = fp * a.g[i];
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 d6_inv(const Dual6& a) {
    Dual6 r;
    RealType inv = RealType(1) / a.v;
    RealType inv2 = inv * inv;
    r.v = inv;
    for (int i = 0; i < 6; ++i) r.g[i] = -a.g[i] * inv2;
    return r;
}

KOKKOS_INLINE_FUNCTION Dual6 d6_abs(const Dual6& a) {
    if (a.v >= 0) return a;
    return -a;
}

// Cubic cutoff spline for Dual6
KOKKOS_INLINE_FUNCTION Dual6 d6_cutoff_spline(const Dual6& r, RealType r_on, RealType r_off) {
    if (r.v <= r_on) return Dual6(RealType(1));
    if (r.v >= r_off) return Dual6(RealType(0));
    RealType vdr = RealType(1) / (r_on - r_off);
    RealType dr = r.v - r_off;
    RealType f_val = 3.0 * vdr * vdr * dr * dr - 2.0 * vdr * vdr * vdr * dr * dr * dr;
    RealType f_deriv = 6.0 * vdr * vdr * dr - 6.0 * vdr * vdr * vdr * dr * dr;
    Dual6 result;
    result.v = f_val;
    for (int i = 0; i < 6; ++i) result.g[i] = f_deriv * r.g[i];
    return result;
}

} // namespace klasp

// Reduction identity so parallel_reduce can accumulate Dual6 sums directly.
namespace Kokkos {
template <>
struct reduction_identity<klasp::Dual6> {
    KOKKOS_FORCEINLINE_FUNCTION static klasp::Dual6 sum() {
        return klasp::Dual6();
    }
};
}  // namespace Kokkos
