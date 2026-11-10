#pragma once

#include "klasp/types.hpp"
#include "klasp/batch.hpp"
#include <vector>

namespace klasp {

// AMD (Average Minimum Distance) descriptors, k nearest neighbours per atom.

struct AMDResult {
    std::vector<std::vector<double>> vectors;  // [batch][k] AMD vectors (Å)
    int k = 0;

    // PDD (Pointwise Distance Distribution), filled only when want_pdd is set:
    // per-asym-atom sorted k-NN distances, n_asym = n_indep * n_sites rows with
    // uniform weight, so AMD is the mean over rows.
    int n_asym = 0;
    std::vector<std::vector<double>> pdd;  // [batch][n_asym*k] row-major (Å)
};

// Compute AMD descriptors for a batch. Positions and cell must already be
// expanded. want_pdd also copies out the PDD rows. k must be 100, 200 or 400;
// anything else throws.
AMDResult compute_amd(const ModelTables& tables, const BatchGeometry& geom,
                      int k = 100, bool want_pdd = false);

inline double amd_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0;
    int n = static_cast<int>(std::min(a.size(), b.size()));
    for (int i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// L-infinity ground distance between two PDD rows (Widdowson & Kurlin).
inline double pdd_row_distance(const double* a, const double* b, int k) {
    double m = 0;
    for (int j = 0; j < k; ++j) {
        double d = std::abs(a[j] - b[j]);
        if (d > m) m = d;
    }
    return m;
}

// Earth Mover's Distance between two uniform-weight PDDs with the same number
// of rows: an exact Hungarian assignment over L-infinity row distances, scaled
// by 1/n_rows. P and Q are row-major [n_rows*k].
double pdd_emd(const double* P, const double* Q, int n_rows, int k);

// Mean L-infinity distance between corresponding rows, assuming the same
// asym-atom order in both. An upper bound on pdd_emd.
inline double pdd_distance_aligned(const double* P, const double* Q,
                                   int n_rows, int k) {
    if (n_rows <= 0) return 0.0;
    double s = 0;
    for (int i = 0; i < n_rows; ++i)
        s += pdd_row_distance(P + static_cast<size_t>(i)*k,
                              Q + static_cast<size_t>(i)*k, k);
    return s / n_rows;
}

// PDD EMD with assignment restricted to rows of the same `group` (element or
// symmetry-equivalence class per asym atom, identical for P and Q).
double pdd_emd_grouped(const double* P, const double* Q, int n_rows, int k,
                       const int* group);

}  // namespace klasp
