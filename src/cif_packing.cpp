#include "klasp/cif_packing.hpp"

#include "klasp/assignment.hpp"
#include "klasp/types.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <vector>

namespace klasp {

namespace {

// Must be finite: the assignment does arithmetic on its costs.
constexpr double FORBIDDEN = 1e6;

struct Counted {
    std::vector<double> positions;   // [m*3]
    std::vector<int> numbers;        // [m]
    std::vector<int> original;       // [m] index back into the molecule
    double centroid[3] = {0, 0, 0};
    int m = 0;
};

Counted count_atoms(const double* positions, const int* numbers, int n, bool heavy_only) {
    Counted c;
    for (int i = 0; i < n; ++i) {
        if (heavy_only && numbers[i] <= 1) continue;
        c.numbers.push_back(numbers[i]);
        c.original.push_back(i);
        for (int d = 0; d < 3; ++d) c.positions.push_back(positions[i * 3 + d]);
    }
    c.m = static_cast<int>(c.numbers.size());
    if (c.m == 0) return c;
    for (int k = 0; k < c.m; ++k)
        for (int d = 0; d < 3; ++d) c.centroid[d] += c.positions[k * 3 + d];
    for (int d = 0; d < 3; ++d) c.centroid[d] /= c.m;
    return c;
}

// Sorted distances from each atom to every other, [m * (m-1)]. Invariant under
// rigid motion and reflection, so mirror-image molecules still match.
std::vector<double> distance_descriptors(const double* positions, int m) {
    std::vector<double> d(static_cast<std::size_t>(m) * (m > 0 ? m - 1 : 0));
    if (m < 2) return d;
    for (int i = 0; i < m; ++i) {
        double* row = &d[static_cast<std::size_t>(i) * (m - 1)];
        int w = 0;
        for (int j = 0; j < m; ++j) {
            if (j == i) continue;
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double e = positions[i * 3 + k] - positions[j * 3 + k];
                s += e * e;
            }
            row[w++] = std::sqrt(s);
        }
        std::sort(row, row + (m - 1));
    }
    return d;
}

bool same_element_multiset(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    std::vector<int> x = a, y = b;
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    return x == y;
}

}  // namespace

BodyFit fit_to_body(const RigidBody& body, const double* positions, const int* numbers,
                    int n, bool heavy_only, int max_iterations) {
    BodyFit fit;
    if (body.m <= 0) return fit;

    const Counted mol = count_atoms(positions, numbers, n, heavy_only);
    const int m = body.m;
    if (mol.m != m) return fit;
    if (!same_element_multiset(mol.numbers, body.numbers)) return fit;

    std::vector<double> cost(static_cast<std::size_t>(m) * m);
    std::vector<double> aligned(static_cast<std::size_t>(m) * 3);
    std::vector<int> next;
    double rotation[9], translation[3];

    // Refine a seed correspondence: overlay, reassign by distance, repeat until stable.
    // Returns the RMSD; the converged permutation is left in `perm`.
    auto refine = [&](std::vector<int>& perm) -> double {
        for (int iter = 0; iter < max_iterations; ++iter) {
            for (int k = 0; k < m; ++k)
                for (int d = 0; d < 3; ++d)
                    aligned[k * 3 + d] = mol.positions[perm[k] * 3 + d];
            // Improper fits accepted: a mirror-image molecule is the same packing unit.
            kabsch(body.positions.data(), aligned.data(), m, rotation, translation,
                   /*allow_improper=*/true);

            for (int i = 0; i < m; ++i) {
                double moved[3];
                for (int d = 0; d < 3; ++d) {
                    moved[d] = translation[d];
                    for (int k = 0; k < 3; ++k)
                        moved[d] += rotation[d * 3 + k] * body.positions[i * 3 + k];
                }
                for (int j = 0; j < m; ++j) {
                    if (body.numbers[i] != mol.numbers[j]) {
                        cost[static_cast<std::size_t>(i) * m + j] = FORBIDDEN;
                        continue;
                    }
                    double s = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        const double e = moved[d] - mol.positions[j * 3 + d];
                        s += e * e;
                    }
                    cost[static_cast<std::size_t>(i) * m + j] = s;
                }
            }
            min_cost_assignment(cost.data(), m, m, &next);
            bool changed = false;
            for (int k = 0; k < m; ++k) {
                if (next[k] < 0) return -1.0;
                if (next[k] != perm[k]) changed = true;
            }
            perm = next;
            if (!changed) break;
        }
        for (int k = 0; k < m; ++k)
            for (int d = 0; d < 3; ++d)
                aligned[k * 3 + d] = mol.positions[perm[k] * 3 + d];
        return kabsch(body.positions.data(), aligned.data(), m, rotation, translation,
                      /*allow_improper=*/true);
    };

    std::vector<int> best_perm;
    double best_rmsd = -1.0;
    double best_rot[9], best_trans[3];
    auto keep_if_better = [&](std::vector<int>& perm) {
        const double r = refine(perm);
        if (r < 0.0) return;
        if (best_rmsd >= 0.0 && r >= best_rmsd) return;
        best_rmsd = r;
        best_perm = perm;
        for (int i = 0; i < 9; ++i) best_rot[i] = rotation[i];
        for (int d = 0; d < 3; ++d) best_trans[d] = translation[d];
    };

    // Seed 1: invariant descriptors, matched within each element only.
    {
        const std::vector<double> db = distance_descriptors(body.positions.data(), m);
        const std::vector<double> dm = distance_descriptors(mol.positions.data(), m);
        const int stride = (m > 1) ? m - 1 : 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
                if (body.numbers[i] != mol.numbers[j]) {
                    cost[static_cast<std::size_t>(i) * m + j] = FORBIDDEN;
                    continue;
                }
                double s = 0.0;
                for (int k = 0; k < stride; ++k) {
                    const double e = db[static_cast<std::size_t>(i) * stride + k]
                                   - dm[static_cast<std::size_t>(j) * stride + k];
                    s += e * e;
                }
                cost[static_cast<std::size_t>(i) * m + j] = std::sqrt(s);
            }
        std::vector<int> perm;
        min_cost_assignment(cost.data(), m, m, &perm);
        bool complete = true;
        for (int k = 0; k < m; ++k) if (perm[k] < 0) complete = false;
        if (!complete) return fit;    // no perfect matching: not the same molecule
        keep_if_better(perm);
    }

    // Seed 2: anchor triples, for symmetric molecules where descriptors are ambiguous.
    // Each body triangle matched by elements and edge lengths fixes an orientation.
    // Skipped once a seed fits to numerical noise.
    if (best_rmsd > 1e-8 && m >= 3) {
        auto dist = [](const double* p, int i, int j) {
            double s = 0.0;
            for (int d = 0; d < 3; ++d) { const double e = p[i*3+d] - p[j*3+d]; s += e * e; }
            return std::sqrt(s);
        };
        // Well-conditioned triangle: atom 0, the atom farthest from it, then the atom
        // farthest from that line.
        const double* bp = body.positions.data();
        int a1 = 1;
        for (int k = 1; k < m; ++k) if (dist(bp, 0, k) > dist(bp, 0, a1)) a1 = k;
        int a2 = -1;
        double best_area = 0.0;
        for (int k = 1; k < m; ++k) {
            if (k == a1) continue;
            double u[3], v[3];
            for (int d = 0; d < 3; ++d) { u[d] = bp[a1*3+d] - bp[d]; v[d] = bp[k*3+d] - bp[d]; }
            const double cx = u[1]*v[2] - u[2]*v[1];
            const double cy = u[2]*v[0] - u[0]*v[2];
            const double cz = u[0]*v[1] - u[1]*v[0];
            const double area = std::sqrt(cx*cx + cy*cy + cz*cz);
            if (area > best_area) { best_area = area; a2 = k; }
        }
        if (a2 >= 0 && best_area > 1e-6) {
            const int anchor[3] = {0, a1, a2};
            const double e01 = dist(bp, anchor[0], anchor[1]);
            const double e02 = dist(bp, anchor[0], anchor[2]);
            const double e12 = dist(bp, anchor[1], anchor[2]);
            // Loose: refinement handles conformer differences.
            const double etol = 0.25;
            const double* mp = mol.positions.data();
            for (int i = 0; i < m && best_rmsd > 1e-8; ++i) {
                if (mol.numbers[i] != body.numbers[anchor[0]]) continue;
                for (int j = 0; j < m && best_rmsd > 1e-8; ++j) {
                    if (j == i || mol.numbers[j] != body.numbers[anchor[1]]) continue;
                    if (std::fabs(dist(mp, i, j) - e01) > etol) continue;
                    for (int k = 0; k < m && best_rmsd > 1e-8; ++k) {
                        if (k == i || k == j) continue;
                        if (mol.numbers[k] != body.numbers[anchor[2]]) continue;
                        if (std::fabs(dist(mp, i, k) - e02) > etol) continue;
                        if (std::fabs(dist(mp, j, k) - e12) > etol) continue;

                        double from[9], onto[9], r3[9], t3[3];
                        for (int d = 0; d < 3; ++d) {
                            from[0*3+d] = bp[anchor[0]*3+d];
                            from[1*3+d] = bp[anchor[1]*3+d];
                            from[2*3+d] = bp[anchor[2]*3+d];
                            onto[0*3+d] = mp[i*3+d];
                            onto[1*3+d] = mp[j*3+d];
                            onto[2*3+d] = mp[k*3+d];
                        }
                        kabsch(from, onto, 3, r3, t3, /*allow_improper=*/true);
                        for (int ii = 0; ii < m; ++ii) {
                            double moved[3];
                            for (int d = 0; d < 3; ++d) {
                                moved[d] = t3[d];
                                for (int c = 0; c < 3; ++c)
                                    moved[d] += r3[d * 3 + c] * bp[ii * 3 + c];
                            }
                            for (int jj = 0; jj < m; ++jj) {
                                if (body.numbers[ii] != mol.numbers[jj]) {
                                    cost[static_cast<std::size_t>(ii) * m + jj] = FORBIDDEN;
                                    continue;
                                }
                                double s = 0.0;
                                for (int d = 0; d < 3; ++d) {
                                    const double e = moved[d] - mp[jj * 3 + d];
                                    s += e * e;
                                }
                                cost[static_cast<std::size_t>(ii) * m + jj] = s;
                            }
                        }
                        std::vector<int> perm;
                        min_cost_assignment(cost.data(), m, m, &perm);
                        bool complete = true;
                        for (int q = 0; q < m; ++q) if (perm[q] < 0) complete = false;
                        if (complete) keep_if_better(perm);
                    }
                }
            }
        }
    }

    if (best_rmsd < 0.0) return fit;
    std::vector<int> perm = best_perm;
    fit.rmsd = best_rmsd;
    for (int i = 0; i < 9; ++i) rotation[i] = best_rot[i];
    for (int d = 0; d < 3; ++d) translation[d] = best_trans[d];

    // Body positions are centred, so the fitted translation is the counted centroid.
    fit.placed.component = 0;
    for (int i = 0; i < 9; ++i) fit.placed.orient[i] = rotation[i];
    for (int d = 0; d < 3; ++d) fit.placed.centroid[d] = translation[d];

    // Permutation over the original atom indices.
    fit.permutation.resize(m);
    for (int k = 0; k < m; ++k) fit.permutation[k] = mol.original[perm[k]];
    fit.ok = true;
    return fit;
}

RigidBody body_from_molecule(const CifMolecule& m, bool heavy_only,
                             double self_symmetry_tol) {
    RigidBody body = make_rigid_body(m.positions.data(), m.numbers.data(), m.size(),
                                     heavy_only);
    detect_self_symmetry(body, self_symmetry_tol);
    return body;
}

CifPacking place_cif_structure(const CifStructure& s, const RigidBody& body,
                               bool heavy_only, double max_fit_rmsd) {
    CifPacking p;
    p.name = s.name;
    p.source = s.source;
    p.space_group = s.space_group;
    p.space_group_name = s.space_group_name;
    for (int i = 0; i < 9; ++i) p.cell[i] = s.cell.matrix[i];

    if (s.molecules.empty()) {
        p.problem = "no molecules";
        return p;
    }

    for (std::size_t i = 0; i < s.molecules.size(); ++i) {
        const CifMolecule& m = s.molecules[i];
        if (m.periodic) {
            p.problem = "contains a periodic component (framework or polymer), not molecules";
            return p;
        }
        const BodyFit fit = fit_to_body(body, m.positions.data(), m.numbers.data(),
                                        m.size(), heavy_only);
        if (!fit.ok) {
            p.problem = "molecule " + std::to_string(i) + " does not match the reference";
            return p;
        }
        if (fit.rmsd > max_fit_rmsd) {
            p.problem = "molecule " + std::to_string(i) + " fits the reference only to "
                      + std::to_string(fit.rmsd) + " A";
            return p;
        }
        p.worst_fit = std::max(p.worst_fit, fit.rmsd);
        // Each molecule is its own environment until find_representatives runs.
        p.representatives.push_back(static_cast<int>(i));
        p.molecules.push_back(fit.placed);
    }
    p.valid = true;
    return p;
}

void find_representatives(CifPacking& p, const RigidBody& body, const CompackOptions& opts,
                          const ClusterSpec& spec) {
    p.representatives.clear();
    if (!p.valid || p.molecules.empty()) return;

    ClusterSpec ref_spec = spec;
    ref_spec.n_neighbours = opts.n_neighbours;
    ClusterSpec pool_spec = spec;
    pool_spec.n_neighbours = candidate_pool_size(opts.n_neighbours);

    std::vector<PackingCluster> kept_pools;
    for (int i = 0; i < static_cast<int>(p.molecules.size()); ++i) {
        const PackingCluster ci = build_cluster_from_cell(p.cell, p.molecules, i, ref_spec);
        if (ci.neighbours.empty()) {          // nothing to judge it by; keep it
            p.representatives.push_back(i);
            kept_pools.push_back(PackingCluster{});
            continue;
        }
        bool duplicate = false;
        for (const PackingCluster& pool : kept_pools) {
            if (pool.neighbours.empty()) continue;
            const CompackResult r = compack_compare(ci, pool, body, opts);
            if (r.valid && r.same_packing) { duplicate = true; break; }
        }
        if (duplicate) continue;
        p.representatives.push_back(i);
        kept_pools.push_back(build_cluster_from_cell(p.cell, p.molecules, i, pool_spec));
    }
    if (p.representatives.empty()) p.representatives.push_back(0);
}

CompackResult compare_packings(const CifPacking& a, const CifPacking& b,
                               const RigidBody& body, const CompackOptions& opts,
                               const ClusterSpec& spec) {
    CompackResult best;
    if (!a.valid || !b.valid) return best;

    ClusterSpec ref_spec = spec;
    ref_spec.n_neighbours = opts.n_neighbours;
    ClusterSpec pool_spec = spec;
    pool_spec.n_neighbours = candidate_pool_size(opts.n_neighbours);

    // B contributes the larger pool, as in compare_structures.
    const PackingCluster cb = build_cluster_from_cell(b.cell, b.molecules, 0, pool_spec);
    if (cb.neighbours.empty()) return best;

    for (int centre : a.representatives) {
        const PackingCluster ca = build_cluster_from_cell(a.cell, a.molecules, centre, ref_spec);
        if (ca.neighbours.empty()) continue;
        const CompackResult r = compack_compare(ca, cb, body, opts);
        if (!r.valid) continue;
        const bool better = !best.valid
                         || r.n_matched > best.n_matched
                         || (r.n_matched == best.n_matched && r.rmsd < best.rmsd);
        if (better) best = r;
        if (best.same_packing) break;
    }
    return best;
}

int compare_all_packings(const std::vector<CifPacking>& packings, const RigidBody& body,
                         const CompackOptions& opts, const ClusterSpec& spec,
                         std::vector<int>& n_matched, std::vector<unsigned char>& same,
                         const std::function<void(long long, long long)>& progress) {
    const int n = static_cast<int>(packings.size());
    SweepClusters sc;
    sc.n = n;
    sc.n_ref = std::min(candidate_pool_size(opts.n_neighbours), compack::MAX_NEIGH);
    sc.n_pool = std::min(candidate_pool_size(opts.n_neighbours), compack::MAX_POOL);
    sc.centrals.resize(n);
    sc.ref.assign(static_cast<std::size_t>(n) * sc.n_ref, PlacedMolecule{});
    sc.pool.assign(static_cast<std::size_t>(n) * sc.n_pool, PlacedMolecule{});
    sc.ref_count.assign(n, 0);
    sc.untied.assign(n, -1);
    ClusterSpec rs = spec, ps = spec;
    rs.n_neighbours = opts.n_neighbours;
    ps.n_neighbours = sc.n_pool;
    int short_cluster = 0;
    std::vector<int> multi;
    for (int i = 0; i < n; ++i) {
        const CifPacking& p = packings[i];
        if (p.representatives.size() > 1) multi.push_back(i);
        const int centre = p.representatives.empty() ? 0 : p.representatives[0];
        const PackingCluster cr = build_cluster_from_cell(p.cell, p.molecules, centre, rs);
        const PackingCluster cp = build_cluster_from_cell(p.cell, p.molecules, centre, ps);
        if (cr.scored < opts.n_neighbours || (int)cp.neighbours.size() < sc.n_pool)
            ++short_cluster;
        sc.centrals[i] = cr.central;
        sc.ref_count[i] = std::min<int>(cr.neighbours.size(), sc.n_ref);
        sc.untied[i] = cr.untied;
        for (int k = 0; k < sc.ref_count[i]; ++k)
            sc.ref[static_cast<std::size_t>(i) * sc.n_ref + k] = cr.neighbours[k];
        for (int k = 0; k < (int)cp.neighbours.size() && k < sc.n_pool; ++k)
            sc.pool[static_cast<std::size_t>(i) * sc.n_pool + k] = cp.neighbours[k];
    }
    compack_sweep(sc, body, opts, n_matched, same, progress);

    const long long n_redo = static_cast<long long>(multi.size()) * n;
    Kokkos::parallel_for(
        "compack_redo_multi", Kokkos::RangePolicy<HostExecSpace>(0, n_redo),
        [&](const long long k) {
            const int i = multi[k / n];
            const int j = static_cast<int>(k % n);
            if (j <= i) return;
            const CompackResult r = compare_packings(packings[i], packings[j], body, opts, spec);
            const unsigned char s = (r.valid && r.same_packing) ? 1 : 0;
            n_matched[static_cast<std::size_t>(i) * n + j] = r.n_matched;
            n_matched[static_cast<std::size_t>(j) * n + i] = r.n_matched;
            same[static_cast<std::size_t>(i) * n + j] = s;
            same[static_cast<std::size_t>(j) * n + i] = s;
        });
    Kokkos::fence();
    return short_cluster;
}

}  // namespace klasp
