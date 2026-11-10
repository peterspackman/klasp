#pragma once
// Minimum-cost assignment (dense O(n^3) shortest augmenting path); rectangular
// cost matrices are supported.

#include <algorithm>
#include <limits>
#include <vector>

namespace klasp {

// cost is row-major [n_rows * n_cols] and must be finite: use a large sentinel,
// not infinity, for forbidden pairs. Fills `assignment` (size n_rows) with each
// row's column, or -1 for a leftover row when n_rows > n_cols. Returns the total cost.
inline double min_cost_assignment(const double* cost, int n_rows, int n_cols,
                                  std::vector<int>* assignment = nullptr) {
    if (assignment) assignment->assign(n_rows > 0 ? n_rows : 0, -1);
    if (n_rows <= 0 || n_cols <= 0) return 0.0;

    if (n_rows > n_cols) {
        std::vector<double> t(static_cast<std::size_t>(n_cols) * n_rows);
        for (int i = 0; i < n_rows; ++i)
            for (int j = 0; j < n_cols; ++j)
                t[static_cast<std::size_t>(j) * n_rows + i] = cost[static_cast<std::size_t>(i) * n_cols + j];
        std::vector<int> col_to_row;
        const double total = min_cost_assignment(t.data(), n_cols, n_rows, &col_to_row);
        if (assignment)
            for (int j = 0; j < n_cols; ++j)
                if (col_to_row[j] >= 0) (*assignment)[col_to_row[j]] = j;
        return total;
    }

    const double INF = std::numeric_limits<double>::max();
    const int n = n_rows, m = n_cols;
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, INF);
        std::vector<char> used(m + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = INF;
            int j1 = -1;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = cost[static_cast<std::size_t>(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }

    double total = 0.0;
    for (int j = 1; j <= m; ++j) {
        if (p[j] == 0) continue;
        total += cost[static_cast<std::size_t>(p[j] - 1) * m + (j - 1)];
        if (assignment) (*assignment)[p[j] - 1] = j - 1;
    }
    return total;
}

inline double min_cost_assignment(const std::vector<double>& cost, int n) {
    return min_cost_assignment(cost.data(), n, n, nullptr);
}

}  // namespace klasp
