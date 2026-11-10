#include "klasp/interp.hpp"

#include <cmath>
#include <stdexcept>

namespace klasp {

namespace {

// Uniform-grid cubic spline through y, in the second-derivative (moment) form:
//
//   M_{i-1} + 4 M_i + M_{i+1} = 6 (y_{i-1} - 2 y_i + y_{i+1}) / h^2,   1 <= i <= n-2
//
// with not-a-knot end conditions, which on a uniform grid give M_1 = b_1/6 and
// M_{n-2} = b_{n-2}/6 in closed form; only the interior moments need a solve.
// Solved in double regardless of RealType.
std::vector<double> spline_moments(const std::vector<double>& y, double h) {
    const int n = static_cast<int>(y.size());
    std::vector<double> M(static_cast<size_t>(n), 0.0);
    if (n <= 2) return M;                    // a single interval is linear

    const double inv_h2 = 1.0 / (h * h);
    std::vector<double> b(static_cast<size_t>(n), 0.0);
    for (int i = 1; i <= n - 2; ++i)
        b[static_cast<size_t>(i)] =
            6.0 * (y[static_cast<size_t>(i - 1)] - 2.0 * y[static_cast<size_t>(i)]
                   + y[static_cast<size_t>(i + 1)]) * inv_h2;

    if (n == 3) {
        // Three points determine one quadratic: constant second derivative.
        const double m = b[1] / 6.0;
        M[0] = M[1] = M[2] = m;
        return M;
    }

    M[1] = b[1] / 6.0;
    M[static_cast<size_t>(n - 2)] = b[static_cast<size_t>(n - 2)] / 6.0;

    // Interior unknowns are 2 .. n-3; M_1 and M_{n-2} move to the right-hand
    // side. Thomas algorithm on the tridiagonal (1, 4, 1) system.
    const int lo = 2, hi = n - 3;
    const int m = hi - lo + 1;
    if (m > 0) {
        std::vector<double> cp(static_cast<size_t>(m)), dp(static_cast<size_t>(m));
        for (int k = 0; k < m; ++k) {
            double rhs = b[static_cast<size_t>(lo + k)];
            if (k == 0)     rhs -= M[1];
            if (k == m - 1) rhs -= M[static_cast<size_t>(n - 2)];
            const double denom = (k == 0) ? 4.0 : 4.0 - cp[static_cast<size_t>(k - 1)];
            cp[static_cast<size_t>(k)] = 1.0 / denom;
            dp[static_cast<size_t>(k)] =
                ((k == 0) ? rhs : rhs - dp[static_cast<size_t>(k - 1)]) / denom;
        }
        M[static_cast<size_t>(hi)] = dp[static_cast<size_t>(m - 1)];
        for (int k = m - 2; k >= 0; --k)
            M[static_cast<size_t>(lo + k)] =
                dp[static_cast<size_t>(k)]
                - cp[static_cast<size_t>(k)] * M[static_cast<size_t>(lo + k + 1)];
    }

    M[0] = 2.0 * M[1] - M[2];
    M[static_cast<size_t>(n - 1)] =
        2.0 * M[static_cast<size_t>(n - 2)] - M[static_cast<size_t>(n - 3)];
    return M;
}

}  // namespace

std::vector<double> radial_knots(double r_min, double r_max, int n, RadialGrid grid) {
    if (n < 2) throw std::runtime_error("RadialTable: need at least 2 knots");
    if (!(r_max > r_min)) throw std::runtime_error("RadialTable: r_max must exceed r_min");
    if (grid == RadialGrid::Log && r_min <= 0.0)
        throw std::runtime_error("RadialTable: a log grid needs r_min > 0");

    std::vector<double> r(static_cast<size_t>(n));
    const double t0 = (grid == RadialGrid::Log) ? std::log(r_min) : r_min;
    const double t1 = (grid == RadialGrid::Log) ? std::log(r_max) : r_max;
    const double h = (t1 - t0) / (n - 1);
    for (int i = 0; i < n; ++i) {
        const double t = t0 + i * h;
        r[static_cast<size_t>(i)] = (grid == RadialGrid::Log) ? std::exp(t) : t;
    }
    // Pin the ends exactly; log/exp round-tripping is not the identity.
    r.front() = r_min;
    r.back()  = r_max;
    return r;
}

RadialSplineData build_radial_spline(double r_min, double r_max,
                                     const std::vector<double>& samples,
                                     RadialGrid grid, RadialScale scale,
                                     RadialEnds ends,
                                     std::vector<RealType>& coeff_out) {
    const int n = static_cast<int>(samples.size());
    if (n < 2) throw std::runtime_error("RadialTable: need at least 2 samples");
    if (!(r_max > r_min)) throw std::runtime_error("RadialTable: r_max must exceed r_min");
    if (grid == RadialGrid::Log && r_min <= 0.0)
        throw std::runtime_error("RadialTable: a log grid needs r_min > 0");

    // Apply the value transform; an untransformable sample is a build-time error.
    std::vector<double> y(samples);
    if (scale == RadialScale::Log) {
        for (int i = 0; i < n; ++i) {
            if (!(samples[static_cast<size_t>(i)] > 0.0))
                throw std::runtime_error(
                    "RadialTable: RadialScale::Log needs strictly positive samples");
            y[static_cast<size_t>(i)] = std::log(samples[static_cast<size_t>(i)]);
        }
    }

    const double t0 = (grid == RadialGrid::Log) ? std::log(r_min) : r_min;
    const double t1 = (grid == RadialGrid::Log) ? std::log(r_max) : r_max;
    const double h = (t1 - t0) / (n - 1);

    const std::vector<double> M = spline_moments(y, h);

    // Per interval, in the normalised local variable s = (t - t_i)/h on [0,1):
    //
    //   S = y_i + (b h) s + (M_i h^2 / 2) s^2 + ((M_{i+1} - M_i) h^2 / 6) s^3
    const int n_int = n - 1;
    coeff_out.resize(static_cast<size_t>(4 * n_int));
    for (int i = 0; i < n_int; ++i) {
        const double yi = y[static_cast<size_t>(i)];
        const double yn = y[static_cast<size_t>(i + 1)];
        const double Mi = M[static_cast<size_t>(i)];
        const double Mn = M[static_cast<size_t>(i + 1)];
        const double b = (yn - yi) / h - h * (2.0 * Mi + Mn) / 6.0;
        coeff_out[static_cast<size_t>(4 * i + 0)] = static_cast<RealType>(yi);
        coeff_out[static_cast<size_t>(4 * i + 1)] = static_cast<RealType>(b * h);
        coeff_out[static_cast<size_t>(4 * i + 2)] = static_cast<RealType>(Mi * h * h / 2.0);
        coeff_out[static_cast<size_t>(4 * i + 3)] = static_cast<RealType>((Mn - Mi) * h * h / 6.0);
    }

    RadialSplineData d;
    d.c = coeff_out.data();
    d.n_intervals = n_int;
    d.t0 = static_cast<RealType>(t0);
    d.inv_dt = static_cast<RealType>(1.0 / h);
    d.inv_dt2 = static_cast<RealType>(1.0 / (h * h));
    d.r_min = static_cast<RealType>(r_min);
    d.r_max = static_cast<RealType>(r_max);
    d.below = ends.below;
    d.above = ends.above;

    // Edge value and slope for the continuations, taken from the spline at s = 0
    // and s = 1 through the same chain rules eval() uses, so the join is C1.
    auto edge = [&](int interval, double s, double r, RealType& v, RealType& dd) {
        const double q0 = coeff_out[static_cast<size_t>(4 * interval + 0)];
        const double q1 = coeff_out[static_cast<size_t>(4 * interval + 1)];
        const double q2 = coeff_out[static_cast<size_t>(4 * interval + 2)];
        const double q3 = coeff_out[static_cast<size_t>(4 * interval + 3)];
        const double u   = q0 + s * (q1 + s * (q2 + s * q3));
        const double u_t = (q1 + s * (2.0 * q2 + 3.0 * s * q3)) / h;
        const double u_r = (grid == RadialGrid::Log) ? u_t / r : u_t;
        if (scale == RadialScale::Log) {
            const double f = std::exp(u);
            v  = static_cast<RealType>(f);
            dd = static_cast<RealType>(f * u_r);
        } else {
            v  = static_cast<RealType>(u);
            dd = static_cast<RealType>(u_r);
        }
    };
    edge(0, 0.0, r_min, d.v_lo, d.d_lo);
    edge(n_int - 1, 1.0, r_max, d.v_hi, d.d_hi);
    d.k_lo = (d.v_lo > RealType(0)) ? d.d_lo / d.v_lo : RealType(0);

    return d;
}

}  // namespace klasp
