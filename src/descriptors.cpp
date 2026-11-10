#include "klasp/descriptors.hpp"

#include "klasp/assignment.hpp"
#include "klasp/constants.hpp"
#include "klasp/dedup.hpp"  // AMD_K

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace klasp {

double pdd_emd(const double* P, const double* Q, int n_rows, int k) {
    if (n_rows <= 0) return 0.0;
    std::vector<double> cost(static_cast<size_t>(n_rows) * n_rows);
    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_rows; ++j)
            cost[static_cast<size_t>(i) * n_rows + j] =
                pdd_row_distance(P + static_cast<size_t>(i) * k,
                                 Q + static_cast<size_t>(j) * k, k);
    return min_cost_assignment(cost, n_rows) / n_rows;
}

double pdd_emd_grouped(const double* P, const double* Q, int n_rows, int k,
                       const int* group) {
    if (n_rows <= 0) return 0.0;
    // Partition row indices by group value (small number of distinct elements).
    std::vector<int> labels;            // distinct group values, in first-seen order
    std::vector<std::vector<int>> rows; // row indices per distinct group
    for (int i = 0; i < n_rows; ++i) {
        int g = -1;
        for (int t = 0; t < (int)labels.size(); ++t)
            if (labels[t] == group[i]) { g = t; break; }
        if (g < 0) { g = (int)labels.size(); labels.push_back(group[i]); rows.emplace_back(); }
        rows[g].push_back(i);
    }
    double total = 0.0;
    std::vector<double> cost;
    for (const auto& idx : rows) {
        const int m = (int)idx.size();
        if (m == 1) {  // forced match — no assignment
            total += pdd_row_distance(P + static_cast<size_t>(idx[0]) * k,
                                      Q + static_cast<size_t>(idx[0]) * k, k);
            continue;
        }
        cost.assign(static_cast<size_t>(m) * m, 0.0);
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b)
                cost[static_cast<size_t>(a) * m + b] =
                    pdd_row_distance(P + static_cast<size_t>(idx[a]) * k,
                                     Q + static_cast<size_t>(idx[b]) * k, k);
        total += min_cost_assignment(cost, m);
    }
    return total / n_rows;
}

// Average Minimum Distance (Widdowson & Kurlin 2022), matching
// occ::descriptors::PointwiseDistanceDistribution: for each asymmetric-unit atom,
// the k smallest distances to any other atom in the infinite crystal, excluding
// the self-at-origin pair.
namespace {

// Sorted k nearest neighbour distances (Bohr) of each real asymmetric-unit atom.
template <int K>
void fill_knn(const BatchGeometry& geom, View1D<int> real_asym, int n_asym,
              bool has_types, int ghost_type, View3D<RealType> d_knn) {
    const int n_batch = geom.n_batch;
    const int n_atoms = geom.n_atoms;
    auto positions = geom.positions;
    auto b_cell    = geom.cell;
    auto b_mshift  = geom.max_shift;
    auto b_types   = geom.types;
    Kokkos::parallel_for("amd_knn",
        Kokkos::RangePolicy<ExecSpace>(0, n_batch * n_asym),
        KOKKOS_LAMBDA(const int flat) {
            const int b = flat / n_asym;
            const int row = flat % n_asym;
            const int i = real_asym(row);

            const int sa = b_mshift(b, 0);
            const int sb = b_mshift(b, 1);
            const int sc = b_mshift(b, 2);
            const int n_sb = 2*sb + 1;
            const int n_sc = 2*sc + 1;
            const int n_shifts = (2*sa + 1) * n_sb * n_sc;

            const RealType pix = positions(b, i, 0);
            const RealType piy = positions(b, i, 1);
            const RealType piz = positions(b, i, 2);

            // Squared distances, sorted ascending; sqrt is taken on write-back.
            RealType topk[K];
            for (int m = 0; m < K; ++m) topk[m] = RealType(1e30);

            for (int j = 0; j < n_atoms; ++j) {
                if (has_types && b_types(b, j) == ghost_type) continue;  // padding, not an atom
                const RealType dx = positions(b, j, 0) - pix;
                const RealType dy = positions(b, j, 1) - piy;
                const RealType dz = positions(b, j, 2) - piz;

                for (int s = 0; s < n_shifts; ++s) {
                    const int ia = s / (n_sb * n_sc) - sa;
                    const int rem = s % (n_sb * n_sc);
                    const int ib = rem / n_sc - sb;
                    const int ic = rem % n_sc - sc;

                    const RealType sx = RealType(ia)*b_cell(b,0,0) + RealType(ib)*b_cell(b,0,1) + RealType(ic)*b_cell(b,0,2);
                    const RealType sy = RealType(ia)*b_cell(b,1,0) + RealType(ib)*b_cell(b,1,1) + RealType(ic)*b_cell(b,1,2);
                    const RealType sz = RealType(ia)*b_cell(b,2,0) + RealType(ib)*b_cell(b,2,1) + RealType(ic)*b_cell(b,2,2);

                    const RealType rx = dx + sx;
                    const RealType ry = dy + sy;
                    const RealType rz = dz + sz;
                    const RealType r2 = rx*rx + ry*ry + rz*rz;
                    if (r2 < RealType(1e-20)) continue;
                    if (r2 >= topk[K-1]) continue;

                    topk[K-1] = r2;
                    for (int m = K-2; m >= 0; --m) {
                        if (topk[m+1] < topk[m]) {
                            RealType tmp = topk[m];
                            topk[m] = topk[m+1];
                            topk[m+1] = tmp;
                        } else break;
                    }
                }
            }

            for (int m = 0; m < K; ++m)
                d_knn(b, row, m) = Kokkos::sqrt(topk[m]);
        });
    ExecSpace().fence();

}

}  // namespace

AMDResult compute_amd(const ModelTables& tables, const BatchGeometry& geom,
                      int k, bool want_pdd) {
    if (k != 100 && k != 200 && k != 400)
        throw std::invalid_argument(
            "compute_amd: k must be 100, 200 or 400, got " + std::to_string(k));
    const int n_batch = geom.n_batch;
    const int n_indep = tables.n_indep;
    const int n_sites = tables.n_sites_per_mol;

    // Padded co-crystal slots carry ghost_type and are excluded as neighbours and
    // as rows. ghost_type < 0 means every slot is a real atom.
    const int ghost_type = tables.ghost_type;
    const bool has_types = ghost_type >= 0 && geom.types.extent(0) > 0;
    std::vector<int> real_asym_host;
    if (has_types) {
        auto h_types = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), Kokkos::subview(geom.types, 0, Kokkos::ALL));
        for (int i = 0; i < n_indep * n_sites; ++i)
            if (h_types(i) != ghost_type) real_asym_host.push_back(i);
    } else {
        for (int i = 0; i < n_indep * n_sites; ++i) real_asym_host.push_back(i);
    }
    const int n_asym = static_cast<int>(real_asym_host.size());
    View1D<int> real_asym("amd_real_asym", std::max(1, n_asym));
    {
        auto h = Kokkos::create_mirror_view(real_asym);
        for (int i = 0; i < n_asym; ++i) h(i) = real_asym_host[i];
        Kokkos::deep_copy(real_asym, h);
    }

    // Per-(batch, asym-atom) sorted top-k neighbour distances (Bohr).
    View3D<RealType> d_knn("amd_knn", n_batch, n_asym, k);

    switch (k) {
        case 100: fill_knn<100>(geom, real_asym, n_asym, has_types, ghost_type, d_knn); break;
        case 200: fill_knn<200>(geom, real_asym, n_asym, has_types, ghost_type, d_knn); break;
        default: fill_knn<400>(geom, real_asym, n_asym, has_types, ghost_type, d_knn); break;
    }

    // Column mean (Bohr to Angstrom), a fixed-order sum per output cell.
    View2D<RealType> d_amd("d_amd", n_batch, k);
    const RealType inv_n = BOHR_TO_ANGSTROM / RealType(n_asym);
    Kokkos::parallel_for("amd_mean",
        Kokkos::RangePolicy<ExecSpace>(0, n_batch * k),
        KOKKOS_LAMBDA(const int flat) {
            const int b = flat / k;
            const int j = flat % k;
            RealType sum = 0;
            for (int i = 0; i < n_asym; ++i)
                sum += d_knn(b, i, j);
            d_amd(b, j) = sum * inv_n;
        });
    ExecSpace().fence();

    auto h_amd = Kokkos::create_mirror_view(d_amd);
    Kokkos::deep_copy(ExecSpace(), h_amd, d_amd);

    AMDResult result;
    result.k = k;
    result.vectors.resize(n_batch);
    for (int b = 0; b < n_batch; ++b) {
        result.vectors[b].resize(k);
        for (int j = 0; j < k; ++j)
            result.vectors[b][j] = h_amd(b, j);
    }

    // PDD rows, converted to Angstrom so AMD is their mean.
    if (want_pdd) {
        result.n_asym = n_asym;
        auto h_knn = Kokkos::create_mirror_view(d_knn);
        Kokkos::deep_copy(ExecSpace(), h_knn, d_knn);
        ExecSpace().fence();
        result.pdd.resize(n_batch);
        for (int b = 0; b < n_batch; ++b) {
            result.pdd[b].resize(static_cast<size_t>(n_asym) * k);
            for (int i = 0; i < n_asym; ++i)
                for (int j = 0; j < k; ++j)
                    result.pdd[b][static_cast<size_t>(i)*k + j] =
                        h_knn(b, i, j) * BOHR_TO_ANGSTROM;
        }
    }
    return result;
}

}  // namespace klasp
