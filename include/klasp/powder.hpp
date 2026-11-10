#pragma once

#include "klasp/types.hpp"
#include "klasp/batch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <cstddef>
#include <vector>

namespace klasp {

// Fixed pattern length for serialized descriptors and the CDTW scratch path:
// about 0.04 deg per bin over 2-theta in [5, 50] deg.
static constexpr int POWDER_N_BINS = 1024;

// Largest Sakoe-Chiba band the CDTW keeps its scratch on the stack for.
static constexpr int POWDER_CDTW_MAX_BAND = 32;

// X-ray powder diffraction descriptor computed per batch from the expanded atom
// positions and cell. Cromer-Mann form factors (gemmi IT92), direct hkl
// enumeration, Gaussian peak profiles binned onto a 2-theta grid.

enum class PowderNormalization {
    NONE,   // raw binned intensities
    L2,     // L2 norm = 1 (simple cosine geometry)
    AREA,   // trapezoidal integral = 1 (matches mol-cspy Simpson normalization)
};

struct PowderSettings {
    double wavelength      = 1.5406;  // X-ray wavelength (Å), Cu Kα1
    double two_theta_min   = 5.0;     // degrees
    double two_theta_max   = 50.0;    // degrees
    int    n_bins          = POWDER_N_BINS;
    double peak_fwhm       = 0.1;     // Gaussian FWHM (degrees)
    // Cap on the per-axis |h|,|k|,|l| range. The range used is derived per cell
    // from d_min = lambda / (2 sin(theta_max)); this only bounds a pathological
    // cell. 32 covers a ~58 A axis at the Cu K-alpha / 50 deg defaults.
    int    max_hkl         = 32;
    bool   lorentz_polarization = true;
    PowderNormalization normalization = PowderNormalization::AREA;
};

// Per-atom Cromer-Mann form-factor coefficients, sized (n_atoms, 9) with
// columns [a0..a3, b0..b3, c] matching pack.cpp's xray_ff layout. n_atoms
// must match the expanded unit cell (geom.n_atoms).
struct PowderFormFactors {
    View2D<RealType> cm;
    int n_atoms = 0;
};

// Build per-atom Cromer-Mann coefficients from element symbols (length ==
// geom.n_atoms; padded ghost atoms can use "X" — treated as zero scatterers).
PowderFormFactors build_powder_form_factors(
    const std::vector<std::string>& elements);

struct PowderResult {
    std::vector<std::vector<double>> patterns;  // [batch][n_bins]
    std::vector<double> two_theta;              // [n_bins] bin centers (degrees)
    int n_bins = 0;
};

// --------------------------------------------------------------------------
// Storage encoding. Sparse (index, value) pairs with a magic word, lossless;
// the encoder falls back to dense if that would be larger. Anything without the
// magic word is read as a dense float array.
// --------------------------------------------------------------------------

std::vector<char> encode_powder(const float* pattern, int n_bins);

// Fills `out` (n_bins floats) from either encoding. False if the blob is a
// truncated or foreign value; `out` may then be partly written.
bool decode_powder(const void* blob, std::size_t n_bytes, float* out, int n_bins);

// Compute powder patterns for all structures in a batch workspace.
// Requires positions and cell to be expanded (call batch_expand_asymmetric_unit
// first). Patterns are returned on the host.
PowderResult compute_powder_pattern(const BatchGeometry& geom,
                                    const PowderFormFactors& ff,
                                    const PowderSettings& settings = PowderSettings{});

// --------------------------------------------------------------------------
// Similarity / distance metrics. Definitions match mol-cspy's so its dedup
// thresholds carry over. All work on contiguous float buffers sized n.
// --------------------------------------------------------------------------

// Cosine similarity in [-1, 1]. For non-negative patterns (e.g. intensities)
// bounded in [0, 1]. 1 = identical direction.
inline float powder_cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += double(a[i]) * double(b[i]);
        na  += double(a[i]) * double(a[i]);
        nb  += double(b[i]) * double(b[i]);
    }
    if (na <= 0 || nb <= 0) return 0.0f;
    return static_cast<float>(dot / std::sqrt(na * nb));
}

// Group count for the cosine bound below: 32 groups of 32 bins.
static constexpr int POWDER_N_GROUPS = 32;

// Per-group L2 norms of a pattern, plus its total norm, precomputed once per
// pattern for powder_cosine_can_reach.
struct PowderNorms {
    float group[POWDER_N_GROUPS];
    float total;
};

inline PowderNorms powder_group_norms(const float* p, int n = POWDER_N_BINS) {
    PowderNorms out{};
    const int gs = n / POWDER_N_GROUPS;
    double total = 0;
    for (int g = 0; g < POWDER_N_GROUPS; ++g) {
        double s2 = 0;
        for (int j = 0; j < gs; ++j) {
            const double v = p[g * gs + j];
            s2 += v * v;
        }
        out.group[g] = static_cast<float>(std::sqrt(s2));
        total += s2;
    }
    out.total = static_cast<float>(std::sqrt(total));
    return out;
}

// Can powder_cosine_similarity(a, b) possibly reach `thresh`?
//
// Cauchy-Schwarz applied per group bounds the dot product from above:
//   <a,b> = sum_g <a_g,b_g> <= sum_g |a_g| |b_g|
// so when (sum_g |a_g||b_g|) < thresh * |a| * |b| the pair cannot match. It
// never rejects a pair the cosine would accept.
inline bool powder_cosine_can_reach(const PowderNorms& a, const PowderNorms& b,
                                    double thresh) {
    if (a.total <= 0 || b.total <= 0) return false;
    double ub = 0;
    for (int g = 0; g < POWDER_N_GROUPS; ++g)
        ub += double(a.group[g]) * double(b.group[g]);
    return ub >= thresh * double(a.total) * double(b.total);
}

// L1 distance: the r=0 case of constrained DTW.
inline float powder_l1_distance(const float* a, const float* b, int n) {
    double d = 0;
    for (int i = 0; i < n; ++i) d += std::fabs(double(a[i]) - double(b[i]));
    return static_cast<float>(d);
}

namespace detail {

// Banded DTW: D(i,j) = |a_i - b_j| + min(D(i-1,j-1), D(i-1,j), D(i,j-1)) over
// |i - j| <= r. Row i is stored by diagonal offset k = j - i + r at slot k + 1,
// so slots 0 and 2r + 2 stay infinite. Each row is a vectorisable pass over the
// two cells above, then a serial pass for the cell to the left.
//
// Returns +inf once a whole row exceeds `abandon`: costs are non-negative and
// every warping path crosses every row.
inline double cdtw_band(const float* a, const float* b, int n, int r, float abandon,
                        double* prev, double* curr, double* cost) {
    constexpr double INF = std::numeric_limits<double>::infinity();
    const int w = 2 * r + 3;
    std::fill(prev, prev + w, INF);
    std::fill(curr, curr + w, INF);
    prev[r + 1] = 0;
    for (int i = 0; i < n; ++i) {
        const int lo = std::max(0, r - i);
        const int hi = std::min(2 * r, n - 1 - i + r);
        const float* bk = b + (i - r);
        const double ai = a[i];
        for (int k = lo; k <= hi; ++k) {
            cost[k] = std::fabs(ai - double(bk[k]));
            curr[k + 1] = cost[k] + std::min(prev[k + 1], prev[k + 2]);
        }
        double left = curr[lo];
        double row_min = INF;
        for (int k = lo; k <= hi; ++k) {
            left = std::min(curr[k + 1], cost[k] + left);
            curr[k + 1] = left;
            row_min = std::min(row_min, left);
        }
        if (static_cast<float>(row_min) > abandon) return INF;
        std::swap(prev, curr);
    }
    return prev[r + 1];
}

inline float cdtw(const float* a, const float* b, int n, int r, float abandon) {
    r = std::min(r, n - 1);
    const int w = 2 * r + 3;
    constexpr int kStack = 2 * POWDER_CDTW_MAX_BAND + 3;
    if (w <= kStack) {
        double prev[kStack], curr[kStack], cost[kStack];
        return static_cast<float>(cdtw_band(a, b, n, r, abandon, prev, curr, cost));
    }
    std::vector<double> scratch(3 * static_cast<size_t>(w));
    return static_cast<float>(cdtw_band(a, b, n, r, abandon, scratch.data(),
                                        scratch.data() + w, scratch.data() + 2 * w));
}

}  // namespace detail

// Running min and max of a pattern over a window of +/- r bins.
struct PowderEnvelope {
    float lo[POWDER_N_BINS];
    float hi[POWDER_N_BINS];
};

inline PowderEnvelope powder_envelope(const float* p, int r, int n = POWDER_N_BINS) {
    PowderEnvelope env;
    for (int i = 0; i < n; ++i) {
        float lo = p[i], hi = p[i];
        for (int j = std::max(0, i - r); j <= std::min(n - 1, i + r); ++j) {
            lo = std::min(lo, p[j]);
            hi = std::max(hi, p[j]);
        }
        env.lo[i] = lo;
        env.hi[i] = hi;
    }
    return env;
}

// Can the CDTW distance between x and the pattern `env` was built from (same
// band) possibly be within `threshold`? A lower bound (LB_Keogh): every warping
// path matches each x_i to some bin within the band, which costs at least x_i's
// distance outside [lo_i, hi_i]. Never rejects a pair the CDTW would accept: the
// bound is shrunk to cover the two sums rounding differently, then rounded to
// float as the distance is.
inline bool powder_cdtw_can_reach(const float* x, const PowderEnvelope& env, int n,
                                  float threshold) {
    auto over = [threshold](double lb) {
        return static_cast<float>(lb * (1.0 - 1e-9)) > threshold;
    };
    double lb = 0;
    for (int i = 0; i < n; ++i) {
        const double v = x[i];
        if (v > env.hi[i]) lb += v - env.hi[i];
        else if (v < env.lo[i]) lb += env.lo[i] - v;
        if ((i & 63) == 63 && over(lb)) return false;
    }
    return !over(lb);
}

// Minimum L1-accumulated distance over all warpings within a Sakoe-Chiba band
// |i - j| <= r. r = 0 is the plain L1 distance.
inline float powder_cdtw_sakoe_chiba(const float* a, const float* b, int n, int r) {
    if (r <= 0 || n <= 1) return powder_l1_distance(a, b, n);
    return detail::cdtw(a, b, n, r, std::numeric_limits<float>::infinity());
}

// Is the CDTW distance under a band of width band_range - 1 within `threshold`?
// The distance is non-increasing in the band width, so the widest band decides.
// Near-identical patterns are settled by the L1 pass; the rest stop at the
// first row that is already over the threshold.
inline bool powder_cdtw_within(const float* a, const float* b, int n,
                               int band_range, float threshold) {
    if (powder_l1_distance(a, b, n) <= threshold) return true;
    if (band_range <= 1 || n <= 1) return false;
    return detail::cdtw(a, b, n, band_range - 1, threshold) <= threshold;
}

// Cosine distance (= 1 - cos_similarity) for non-negative patterns.
inline double powder_cosine_distance(const std::vector<double>& a,
                                     const std::vector<double>& b) {
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na <= 0 || nb <= 0) return 1.0;
    return 1.0 - dot / std::sqrt(na * nb);
}

}  // namespace klasp
