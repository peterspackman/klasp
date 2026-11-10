#pragma once

// Tabulated radial functions with value, dV/dr and d2V/dr2, as a C2 cubic
// spline (batch_strain_hessian needs a continuous second derivative).
//
// Grid and value transform are template parameters so the default
// (Linear/Direct) has no log() or exp() in the lookup. Coefficients are stored
// four per interval, interval-major, in the normalised local variable
// s = (t - t_i)/dt in [0,1).
//
// RadialSpline is a POD holding a bare pointer to its coefficients; the owner
// must outlive every copy.

#include "klasp/types.hpp"

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

namespace klasp {

// Spacing of the knots: the table is uniform in t = map(r).
enum class RadialGrid {
    Linear,   // t = r
    Log,      // t = ln(r): geometric knots, costs a log() per evaluation
};

// What is splined.
enum class RadialScale {
    Direct,   // f itself; required for a function that changes sign
    Log,      // ln(f), for strictly positive f; costs an exp() per evaluation
};

// Continuation below r_min. Both match the table's value and slope at r_min.
enum class RadialBelow {
    Linear,       // v0 + d0 (r - r_min)
    Exponential,  // v0 exp(k (r - r_min)), k = d0/v0; falls back to Linear if v0 <= 0
};

// Continuation above r_max.
enum class RadialAbove {
    Zero,     // hard zero; check RadialTable::value_at_max() for a discontinuity
    Linear,   // v1 + d1 (r - r_max).
};

struct RadialEnds {
    RadialBelow below = RadialBelow::Linear;
    RadialAbove above = RadialAbove::Zero;
};

// Everything the evaluator needs that is not a compile-time choice.
struct RadialSplineData {
    const RealType* c = nullptr;  // 4 coefficients per interval, interval-major,
                                  // in the normalised local variable [0,1)
    int n_intervals = 0;

    RealType t0 = 0;       // mapped coordinate of knot 0
    RealType inv_dt = 0;   // 1 / mapped spacing
    RealType inv_dt2 = 0;
    RealType r_min = 0;
    RealType r_max = 0;

    // Value and slope at each end, in r, after the value transform.
    RealType v_lo = 0, d_lo = 0, k_lo = 0;   // k_lo = d_lo / v_lo
    RealType v_hi = 0, d_hi = 0;

    RadialBelow below = RadialBelow::Linear;
    RadialAbove above = RadialAbove::Zero;
};

// Device-side spline: trivially copyable, capture by value. `c` points at
// coefficients owned elsewhere (RadialTable or a View), which must outlive it.
template <RadialGrid Grid = RadialGrid::Linear,
          RadialScale Scale = RadialScale::Direct>
struct RadialSpline : RadialSplineData {
    static constexpr RadialGrid grid = Grid;
    static constexpr RadialScale scale = Scale;

    KOKKOS_INLINE_FUNCTION bool valid() const { return c != nullptr && n_intervals > 0; }
    KOKKOS_INLINE_FUNCTION bool in_range(RealType r) const {
        return r > r_min && r < r_max;
    }

    // Value, first and second derivative with respect to r.
    KOKKOS_INLINE_FUNCTION
    void eval(RealType r, RealType& v, RealType& dv, RealType& d2v) const {
        if (!in_range(r)) { edges(r, v, dv, d2v); return; }
        eval_in_range(r, RealType(1) / r, v, dv, d2v);
    }

    // Same, reusing a reciprocal the caller already has.
    KOKKOS_INLINE_FUNCTION
    void eval(RealType r, RealType inv_r, RealType& v, RealType& dv, RealType& d2v) const {
        if (!in_range(r)) { edges(r, v, dv, d2v); return; }
        eval_in_range(r, inv_r, v, dv, d2v);
    }

    // Value and first derivative only.
    KOKKOS_INLINE_FUNCTION
    void eval(RealType r, RealType& v, RealType& dv) const {
        RealType d2;
        eval(r, v, dv, d2);
    }

    KOKKOS_INLINE_FUNCTION
    RealType value(RealType r) const {
        if (!in_range(r)) {
            RealType v, dv, d2v;
            edges(r, v, dv, d2v);
            return v;
        }
        const RealType s = local(r);
        const int i = interval(r);
        const RealType* q = c + 4 * i;
        const RealType u = q[0] + s * (q[1] + s * (q[2] + s * q[3]));
        if constexpr (Scale == RadialScale::Log) return Kokkos::exp(u);
        else return u;
    }

    // No range check: for a caller that has already gated on the cutoff.
    KOKKOS_INLINE_FUNCTION
    void eval_in_range(RealType r, RealType inv_r,
                       RealType& v, RealType& dv, RealType& d2v) const {
        const RealType x = mapped(r) ;
        int i = static_cast<int>(x);
        i = (i < 0) ? 0 : (i > n_intervals - 1 ? n_intervals - 1 : i);
        const RealType s = x - static_cast<RealType>(i);

        const RealType* q = c + 4 * i;
        const RealType q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];

        // u and its derivatives with respect to t.
        const RealType u    = q0 + s * (q1 + s * (q2 + s * q3));
        const RealType u_t  = (q1 + s * (RealType(2) * q2 + RealType(3) * s * q3)) * inv_dt;
        const RealType u_tt = (RealType(2) * q2 + RealType(6) * s * q3) * inv_dt2;

        // Chain rule out of the mapped coordinate: t' = 1/r, t'' = -1/r^2.
        RealType u_r, u_rr;
        if constexpr (Grid == RadialGrid::Log) {
            u_r  = u_t * inv_r;
            u_rr = (u_tt - u_t) * inv_r * inv_r;
        } else {
            (void)inv_r;
            u_r  = u_t;
            u_rr = u_tt;
        }

        // ... and out of the value transform.
        if constexpr (Scale == RadialScale::Log) {
            v   = Kokkos::exp(u);
            dv  = v * u_r;
            d2v = v * (u_rr + u_r * u_r);
        } else {
            v = u; dv = u_r; d2v = u_rr;
        }
    }

private:
    // floor(x) is the interval; the fraction is the normalised local variable.
    KOKKOS_INLINE_FUNCTION RealType mapped(RealType r) const {
        if constexpr (Grid == RadialGrid::Log) return (Kokkos::log(r) - t0) * inv_dt;
        else                                   return (r - t0) * inv_dt;
    }
    KOKKOS_INLINE_FUNCTION int interval(RealType r) const {
        int i = static_cast<int>(mapped(r));
        return (i < 0) ? 0 : (i > n_intervals - 1 ? n_intervals - 1 : i);
    }
    KOKKOS_INLINE_FUNCTION RealType local(RealType r) const {
        const RealType x = mapped(r);
        int i = static_cast<int>(x);
        i = (i < 0) ? 0 : (i > n_intervals - 1 ? n_intervals - 1 : i);
        return x - static_cast<RealType>(i);
    }

    KOKKOS_INLINE_FUNCTION
    void edges(RealType r, RealType& v, RealType& dv, RealType& d2v) const {
        if (!valid()) { v = 0; dv = 0; d2v = 0; return; }
        if (r <= r_min) {
            if (below == RadialBelow::Exponential && v_lo > RealType(0)) {
                const RealType e = v_lo * Kokkos::exp(k_lo * (r - r_min));
                v = e; dv = k_lo * e; d2v = k_lo * k_lo * e;
            } else {
                v = v_lo + d_lo * (r - r_min); dv = d_lo; d2v = 0;
            }
        } else {
            if (above == RadialAbove::Zero) { v = 0; dv = 0; d2v = 0; }
            else { v = v_hi + d_hi * (r - r_max); dv = d_hi; d2v = 0; }
        }
    }
};

// Host-side builder and owner.

// The knots a (r_min, r_max, n, grid) implies, where samples must be taken.
std::vector<double> radial_knots(double r_min, double r_max, int n, RadialGrid grid);

// Builds the coefficients and the header; use RadialTable.
RadialSplineData build_radial_spline(double r_min, double r_max,
                                     const std::vector<double>& samples,
                                     RadialGrid grid, RadialScale scale,
                                     RadialEnds ends,
                                     std::vector<RealType>& coeff_out);

template <RadialGrid Grid = RadialGrid::Linear,
          RadialScale Scale = RadialScale::Direct>
class RadialTable {
public:
    using Spline = RadialSpline<Grid, Scale>;

    RadialTable() = default;

    // Samples must be the function at radial_knots(r_min, r_max, n, Grid); at
    // least two. Not-a-knot end conditions, so the curvature at r_min is not
    // forced to zero.
    RadialTable(double r_min, double r_max, const std::vector<double>& samples,
                RadialEnds ends = {})
        : m_n_knots(static_cast<int>(samples.size())) {
        static_cast<RadialSplineData&>(m_spline) =
            build_radial_spline(r_min, r_max, samples, Grid, Scale, ends, m_coeff);
        m_spline.c = m_coeff.data();
    }

    // Same, sampling `f` at the knots.
    template <typename F>
    RadialTable(double r_min, double r_max, int n_knots, F&& f, RadialEnds ends = {})
        : RadialTable(r_min, r_max,
                      sample(r_min, r_max, n_knots, static_cast<F&&>(f)), ends) {}

    // Points into this object's own storage: it must outlive the spline.
    Spline view() const { return m_spline; }

    int n_knots() const { return m_n_knots; }
    const std::vector<RealType>& coefficients() const { return m_coeff; }

    // The tabulated value at r_max, to check whether RadialAbove::Zero truncates
    // or creates a discontinuity.
    RealType value_at_max() const { return m_spline.v_hi; }

private:
    template <typename F>
    static std::vector<double> sample(double r_min, double r_max, int n, F&& f) {
        const std::vector<double> r = radial_knots(r_min, r_max, n, Grid);
        std::vector<double> y(r.size());
        for (size_t i = 0; i < r.size(); ++i) y[i] = f(r[i]);
        return y;
    }

    std::vector<RealType> m_coeff;
    Spline m_spline;
    int m_n_knots = 0;
};

// A table in device memory. `spline` points into `coeffs`; keep the struct
// alive while any kernel can see the spline.
template <RadialGrid Grid, RadialScale Scale>
struct DeviceRadialTable {
    View1D<RealType> coeffs;
    RadialSpline<Grid, Scale> spline;
};

template <RadialGrid Grid, RadialScale Scale>
DeviceRadialTable<Grid, Scale> upload(const RadialTable<Grid, Scale>& table,
                                      const std::string& label = "radial_table") {
    const std::vector<RealType>& host = table.coefficients();
    DeviceRadialTable<Grid, Scale> out;
    out.coeffs = View1D<RealType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, label), host.size());
    auto mirror = Kokkos::create_mirror_view(out.coeffs);
    for (size_t i = 0; i < host.size(); ++i) mirror(i) = host[i];
    Kokkos::deep_copy(out.coeffs, mirror);
    out.spline = table.view();
    out.spline.c = out.coeffs.data();
    return out;
}

}  // namespace klasp
