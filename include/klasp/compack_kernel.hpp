#pragma once
// Per-pair packing comparison for one Kokkos team or serial use: fixed-size
// buffers, no allocation. For rigid bodies everything closes over centroids and
// orientations:
//
//   per-molecule distance   RMSD^2 = |dc|^2 + (2/m)(tr G - tr(M_a^T M_b Q G))
//   superposition centroid  the mean of the molecule centroids, since sum_k p_k = 0
//   superposition covariance
//       H = sum_i [ m (c_i - cbar)(c_i' - cbar')^T + M_i G (M_i' Q_i)^T ]
//
// where G is the body's gyration tensor and Q ranges over its own symmetry
// operations (see klasp/compack.hpp).

#include <Kokkos_Core.hpp>

namespace klasp {
namespace compack {

// Fixed-size caps: neighbour cluster and pool sizes, and body point-group order.
constexpr int MAX_NEIGH = 24;
constexpr int MAX_POOL = 40;
constexpr int MAX_OPS = 48;

// Must stay finite: the assignment does arithmetic with it.
constexpr double FORBIDDEN = 1e6;

// A basis component as the comparison uses it. `self_ops` points at [n_ops*9]
// doubles that must be reachable from the executing space.
struct BodyView {
    int m = 0;
    int n_ops = 1;
    double gyration[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double trace_g = 0.0;
    const double* self_ops = nullptr;
    // (Q G)^T for every body operation, so tr(P Q G) = <P, (Q G)^T> with P = M_a^T M_b.
    const double* qgt = nullptr;
};

struct Placement {
    double c[3] = {0, 0, 0};
    double M[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    int component = 0;
};

struct PairOptions {
    // Places a result is out of. With n_untied >= 0, references past n_untied are a
    // tied shell at the cluster edge: earlier ones must each match and the shell
    // fills the remaining n_neighbours - n_untied places. With n_untied < 0 every
    // reference must match.
    int n_neighbours = 15;
    int n_untied = -1;
    double match_tolerance = 1.0;
    int max_iterations = 8;
    bool allow_inversion = false;
};

struct PairResult {
    int n_matched = 0;
    int n_neighbours = 0;
    double rmsd = 0.0;
    bool same_packing = false;
    bool inverted = false;
    bool valid = false;
};

KOKKOS_INLINE_FUNCTION void k_mul(const double a[9], const double b[9], double o[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[i * 3 + k] * b[k * 3 + j];
            o[i * 3 + j] = s;
        }
}

// o = a^T b
KOKKOS_INLINE_FUNCTION void k_tmul(const double a[9], const double b[9], double o[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[k * 3 + i] * b[k * 3 + j];
            o[i * 3 + j] = s;
        }
}

// o = a b^T
KOKKOS_INLINE_FUNCTION void k_mult(const double a[9], const double b[9], double o[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[i * 3 + k] * b[j * 3 + k];
            o[i * 3 + j] = s;
        }
}

KOKKOS_INLINE_FUNCTION double k_det(const double m[9]) {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// Frobenius inner product.
KOKKOS_INLINE_FUNCTION double k_frob(const double a[9], const double b[9]) {
    double s = 0.0;
    for (int i = 0; i < 9; ++i) s += a[i] * b[i];
    return s;
}

KOKKOS_INLINE_FUNCTION void k_sym_eigen(const double in[9], double val[3], double vec[9]) {
    double a[9];
    for (int i = 0; i < 9; ++i) a[i] = in[i];
    for (int i = 0; i < 9; ++i) vec[i] = (i % 4 == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 32; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j) off += a[i * 3 + j] * a[i * 3 + j];
        if (off < 1e-30) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                const double apq = a[p * 3 + q];
                if (apq > -1e-300 && apq < 1e-300) continue;
                const double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * apq);
                const double at = theta >= 0 ? theta : -theta;
                const double t = (theta >= 0 ? 1.0 : -1.0) / (at + Kokkos::sqrt(theta * theta + 1.0));
                const double c = 1.0 / Kokkos::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[k * 3 + p], akq = a[k * 3 + q];
                    a[k * 3 + p] = c * akp - s * akq;
                    a[k * 3 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p * 3 + k], aqk = a[q * 3 + k];
                    a[p * 3 + k] = c * apk - s * aqk;
                    a[q * 3 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = vec[k * 3 + p], vkq = vec[k * 3 + q];
                    vec[k * 3 + p] = c * vkp - s * vkq;
                    vec[k * 3 + q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < 3; ++i) val[i] = a[i * 3 + i];
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (val[j] < val[i]) {
                const double tv = val[i]; val[i] = val[j]; val[j] = tv;
                for (int k = 0; k < 3; ++k) {
                    const double t2 = vec[k * 3 + i];
                    vec[k * 3 + i] = vec[k * 3 + j];
                    vec[k * 3 + j] = t2;
                }
            }
}

// Best orthogonal transform from a 3x3 covariance H (Kabsch): proper, or improper
// with `improper`. Handedness is imposed, not chosen by the fit.
KOKKOS_INLINE_FUNCTION
void k_rotation_from_covariance(const double h[9], bool improper, double r[9]) {
    double hth[9];
    k_tmul(h, h, hth);
    double val[3], vec[9];
    k_sym_eigen(hth, val, vec);
    double s[3], v[9];
    for (int i = 0; i < 3; ++i) {
        const double e = val[2 - i];
        s[i] = Kokkos::sqrt(e > 0 ? e : 0.0);
        for (int rr = 0; rr < 3; ++rr) v[rr * 3 + i] = vec[rr * 3 + (2 - i)];
    }
    const double smax = s[0] > 0 ? s[0] : 1.0;
    int rank = 0;
    // Singular values come from eig(H^T H), so a true zero appears as ~sqrt(eps)*smax;
    // the rank cut must stay above that.
    for (int i = 0; i < 3; ++i) if (s[i] > 1e-6 * smax) ++rank;
    double u[9];
    for (int i = 0; i < 3; ++i) {
        if (i < rank) {
            for (int rr = 0; rr < 3; ++rr) {
                double acc = 0.0;
                for (int c = 0; c < 3; ++c) acc += h[rr * 3 + c] * v[c * 3 + i];
                u[rr * 3 + i] = acc / s[i];
            }
        } else {
            for (int rr = 0; rr < 3; ++rr) u[rr * 3 + i] = 0.0;
        }
    }
    // Collinear points: any unit vector perpendicular to the first column will do.
    if (rank == 1) {
        int a = 0;
        for (int i = 1; i < 3; ++i)
            if ((u[i * 3] < 0 ? -u[i * 3] : u[i * 3]) < (u[a * 3] < 0 ? -u[a * 3] : u[a * 3])) a = i;
        double w[3] = {0.0, 0.0, 0.0};
        w[a] = 1.0;
        const double along = u[a * 3];
        for (int r = 0; r < 3; ++r) w[r] -= along * u[r * 3];
        const double nw = Kokkos::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        for (int r = 0; r < 3; ++r) u[r * 3 + 1] = w[r] / nw;
    }
    if (rank == 1 || rank == 2) {
        u[0 * 3 + 2] = u[1 * 3 + 0] * u[2 * 3 + 1] - u[2 * 3 + 0] * u[1 * 3 + 1];
        u[1 * 3 + 2] = u[2 * 3 + 0] * u[0 * 3 + 1] - u[0 * 3 + 0] * u[2 * 3 + 1];
        u[2 * 3 + 2] = u[0 * 3 + 0] * u[1 * 3 + 1] - u[1 * 3 + 0] * u[0 * 3 + 1];
    } else if (rank == 0) {
        for (int i = 0; i < 9; ++i) u[i] = (i % 4 == 0) ? 1.0 : 0.0;
        for (int i = 0; i < 9; ++i) v[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    double d = 1.0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) {
                    const double vk = (k == 2) ? v[i * 3 + k] * d : v[i * 3 + k];
                    acc += vk * u[j * 3 + k];
                }
                r[i * 3 + j] = acc;
            }
        if ((k_det(r) < 0.0) == improper) break;
        d = -1.0;
    }
}

// Lanes: only the cost-matrix fill is split across a team. Every lane repeats
// the assignment and refit on identical data, so results are identical and stay
// in registers.
struct SerialLanes {
    KOKKOS_INLINE_FUNCTION int rank() const { return 0; }
    KOKKOS_INLINE_FUNCTION int size() const { return 1; }
    KOKKOS_INLINE_FUNCTION void barrier() const {}
};

template <class TeamMember>
struct TeamLanes {
    const TeamMember& team;
    KOKKOS_INLINE_FUNCTION int rank() const { return team.team_rank(); }
    KOKKOS_INLINE_FUNCTION int size() const { return team.team_size(); }
    KOKKOS_INLINE_FUNCTION void barrier() const { team.team_barrier(); }
};

// Squared per-molecule distance between two placements of one body, minimised
// over the body's own operations, and the operation that achieved it.
KOKKOS_INLINE_FUNCTION
double k_mol_rmsd2(const BodyView& body, const Placement& a, const Placement& b,
                   int* best_op) {
    double dc2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        const double e = a.c[d] - b.c[d];
        dc2 += e * e;
    }
    if (body.m == 0) { if (best_op) *best_op = 0; return dc2; }
    double pmat[9];
    k_tmul(a.M, b.M, pmat);                       // P = M_a^T M_b
    double best = -1e300;
    int arg = 0;
    for (int o = 0; o < body.n_ops; ++o) {
        const double tr = k_frob(pmat, body.qgt + o * 9);   // tr(P Q G)
        if (tr > best) { best = tr; arg = o; }
    }
    if (best_op) *best_op = arg;
    const double orient = 2.0 * (body.trace_g - best) / body.m;
    const double v = dc2 + (orient > 0.0 ? orient : 0.0);
    return v > 0.0 ? v : 0.0;
}

// Fixed-size minimum-cost assignment, as in klasp/assignment.hpp. Requires rows <= cols.
KOKKOS_INLINE_FUNCTION
void k_assign(const double* cost, int n, int m, int* out) {
    double u[MAX_NEIGH + 1], v[MAX_POOL + 1], minv[MAX_POOL + 1];
    int p[MAX_POOL + 1], way[MAX_POOL + 1];
    bool used[MAX_POOL + 1];
    for (int i = 0; i <= n; ++i) u[i] = 0.0;
    for (int j = 0; j <= m; ++j) { v[j] = 0.0; p[j] = 0; way[j] = 0; }
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        for (int j = 0; j <= m; ++j) { minv[j] = 1e300; used[j] = false; }
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = 1e300;
            int j1 = -1;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
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
    for (int i = 0; i < n; ++i) out[i] = -1;
    for (int j = 1; j <= m; ++j) if (p[j] != 0) out[p[j] - 1] = j - 1;
}

// Compare one pair. `ref` is A's reference cluster (central passed separately),
// `pool` is B's larger candidate set.
template <class Lanes>
KOKKOS_INLINE_FUNCTION
PairResult k_compare_pair_lanes(const Lanes& lanes, const BodyView& body,
                          const Placement& ca, const Placement* ref, int n_ref,
                          const Placement& cb, const Placement* pool, int n_pool,
                          const PairOptions& opts,
                          double* cost /* [n_ref*n_pool] scratch */,
                          // Optional per-reference outputs, each [n_ref]; for an unmatched
                          // reference `cand` is its nearest candidate.
                          int* out_cand = nullptr, double* out_rmsd = nullptr,
                          unsigned char* out_matched = nullptr) {
    PairResult out;
    if (n_ref <= 0 || n_pool <= 0 || body.m == 0) return out;
    if (n_ref > MAX_NEIGH || n_pool > MAX_POOL) return out;
    out.valid = true;
    const int required = (opts.n_untied >= 0) ? opts.n_untied : n_ref;
    const int places = (opts.n_untied >= 0) ? opts.n_neighbours : n_ref;
    out.n_neighbours = places;

    int assign[MAX_NEIGH], match_j[MAX_NEIGH], match_q[MAX_NEIGH];
    int best_n = -1;
    double best_rmsd = 0.0;
    bool best_inverted = false;
    int det_cand[MAX_NEIGH];
    double det_rmsd[MAX_NEIGH];
    unsigned char det_matched[MAX_NEIGH];

    // Seed with every body symmetry operation Q: the centrals overlay exactly for any
    // Q, so only the neighbours can pick the right one. Per Q, three variants of
    // M_b Q M_a^T: refitted proper, refitted improper, and negated.
    //
    // A seed can leave every neighbour just outside the tolerance, so nothing
    // matches and nothing drives the refit. Failing a full match from every seed,
    // each is regrown at 1.5x, then 2x, the tolerance; one that then matches every
    // neighbour is refitted and judged again at the real tolerance.
    const int n_seed_ops = (body.n_ops > 0 ? body.n_ops : 1);
    for (int grow = 0; grow < 3 && best_n != places; ++grow) {
    for (int seed = 0; seed < 3 * n_seed_ops; ++seed) {
        const int variant = seed / n_seed_ops;
        const int q_seed = seed - variant * n_seed_ops;
        if (variant > 0 && !opts.allow_inversion) continue;
        double r[9], t[3];
        double mq_seed[9];
        k_mul(cb.M, body.self_ops + q_seed * 9, mq_seed);   // M_b Q
        k_mult(mq_seed, ca.M, r);                           // M_b Q M_a^T
        if (variant == 1 && k_det(r) > 0.0) continue;       // already the proper seed
        if (variant == 2) for (int i = 0; i < 9; ++i) r[i] = -r[i];
        const bool improper = (variant > 0);
        for (int d = 0; d < 3; ++d) {
            t[d] = cb.c[d];
            for (int k = 0; k < 3; ++k) t[d] -= r[d * 3 + k] * ca.c[k];
        }

        int n_match = 0;
        double fit_rmsd = 0.0;
        bool judged = true;
        for (int pass = 0; pass < (grow > 0 ? 2 : 1); ++pass) {
            const double scale = (pass > 0 || grow == 0) ? 1.0 : (grow == 1 ? 1.5 : 2.0);
            const double tol = scale * opts.match_tolerance;
            int prev_sig = -1;
            lanes.barrier();   // a pass can end on a break, with lanes still reading cost
            for (int iter = 0; iter < opts.max_iterations; ++iter) {
                Placement moved[MAX_NEIGH];
                for (int i = 0; i < n_ref; ++i) {
                    moved[i].component = ref[i].component;
                    for (int d = 0; d < 3; ++d) {
                        moved[i].c[d] = t[d];
                        for (int k = 0; k < 3; ++k) moved[i].c[d] += r[d * 3 + k] * ref[i].c[k];
                    }
                    k_mul(r, ref[i].M, moved[i].M);
                }
                // The only part split across lanes.
                const int n_cell = n_ref * n_pool;
                for (int e = lanes.rank(); e < n_cell; e += lanes.size()) {
                    const int i = e / n_pool, j = e - i * n_pool;
                    if (ref[i].component != pool[j].component) {
                        cost[(size_t)e] = FORBIDDEN;
                        continue;
                    }
                    cost[(size_t)e] = Kokkos::sqrt(k_mol_rmsd2(body, moved[i], pool[j], nullptr));
                }
                lanes.barrier();
                k_assign(cost, n_ref, n_pool, assign);

                n_match = 0;
                int sig = 0;
                for (int i = 0; i < n_ref; ++i) {
                    const int j = assign[i];
                    if (j < 0) continue;
                    if (cost[(size_t)i * n_pool + j] > tol) continue;
                    match_j[n_match] = j;
                    // Body operation this pair used under the current transform, carried into the refit.
                    int q = 0;
                    k_mol_rmsd2(body, moved[i], pool[j], &q);
                    match_q[n_match] = q;
                    assign[n_match] = i;          // compact the reference index alongside
                    ++n_match;
                    sig = sig * 131 + (i * 1009 + j);
                }

                // Refit over the central and matched neighbours from centroids and orientations.
                double cbar_a[3] = {0, 0, 0}, cbar_b[3] = {0, 0, 0};
                const int ns = n_match + 1;
                for (int d = 0; d < 3; ++d) { cbar_a[d] += ca.c[d]; cbar_b[d] += cb.c[d]; }
                for (int s = 0; s < n_match; ++s)
                    for (int d = 0; d < 3; ++d) {
                        cbar_a[d] += ref[assign[s]].c[d];
                        cbar_b[d] += pool[match_j[s]].c[d];
                    }
                for (int d = 0; d < 3; ++d) { cbar_a[d] /= ns; cbar_b[d] /= ns; }

                double h[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                auto accumulate = [&](const Placement& pa, const Placement& pb, int q) {
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            h[i * 3 + j] += body.m * (pa.c[i] - cbar_a[i]) * (pb.c[j] - cbar_b[j]);
                    double mq[9], gm[9];
                    k_mul(pb.M, body.self_ops + q * 9, mq);   // M_b Q
                    k_mult(body.gyration, mq, gm);            // G (M_b Q)^T
                    double term[9];
                    k_mul(pa.M, gm, term);                    // M_a G (M_b Q)^T
                    for (int i = 0; i < 9; ++i) h[i] += term[i];
                };
                accumulate(ca, cb, 0);
                for (int s = 0; s < n_match; ++s)
                    accumulate(ref[assign[s]], pool[match_j[s]], match_q[s]);

                k_rotation_from_covariance(h, improper, r);
                for (int d = 0; d < 3; ++d) {
                    t[d] = cbar_b[d];
                    for (int k = 0; k < 3; ++k) t[d] -= r[d * 3 + k] * cbar_a[k];
                }

                double sum2 = 0.0;
                auto residual = [&](const Placement& pa, const Placement& pb, int q) {
                    Placement mi;
                    for (int d = 0; d < 3; ++d) {
                        mi.c[d] = t[d];
                        for (int k = 0; k < 3; ++k) mi.c[d] += r[d * 3 + k] * pa.c[k];
                    }
                    k_mul(r, pa.M, mi.M);
                    double one[9], qg[9];
                    k_mul(body.self_ops + q * 9, body.gyration, qg);
                    k_mul(pb.M, qg, one);
                    double d2 = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        const double e = mi.c[d] - pb.c[d];
                        d2 += e * e;
                    }
                    const double orient = 2.0 * (body.trace_g - k_frob(mi.M, one)) / body.m;
                    return d2 + (orient > 0 ? orient : 0.0);
                };
                sum2 += residual(ca, cb, 0);
                for (int s = 0; s < n_match; ++s)
                    sum2 += residual(ref[assign[s]], pool[match_j[s]], match_q[s]);
                fit_rmsd = Kokkos::sqrt(sum2 / ns);

                if (sig == prev_sig) break;
                prev_sig = sig;
                lanes.barrier();   // every lane has finished reading before the refill
            }

            if (grow > 0 && pass == 0 && n_match < places) {
                judged = false;
                break;
            }
        }
        if (!judged) continue;

        int matched_required = 0, matched_shell = 0;
        for (int s = 0; s < n_match; ++s) {
            if (assign[s] < required) ++matched_required;
            else ++matched_shell;
        }
        const int shell_places = places - required;
        const int score = matched_required
                        + (matched_shell < shell_places ? matched_shell : shell_places);
        if (score > best_n) {
            best_n = score;
            best_rmsd = fit_rmsd;
            best_inverted = k_det(r) < 0.0;
            if (out_cand || out_rmsd || out_matched) {
                for (int i = 0; i < n_ref; ++i) {
                    det_cand[i] = -1;
                    det_rmsd[i] = 0.0;
                    det_matched[i] = 0;
                }
                for (int s = 0; s < n_match; ++s) {
                    const int i = assign[s];
                    det_cand[i] = match_j[s];
                    det_rmsd[i] = cost[(size_t)i * n_pool + match_j[s]];
                    det_matched[i] = 1;
                }
                for (int i = 0; i < n_ref; ++i) {
                    if (det_matched[i]) continue;
                    double best_d = 1e300;
                    int arg = -1;
                    for (int j = 0; j < n_pool; ++j) {
                        const double c = cost[(size_t)i * n_pool + j];
                        if (c >= FORBIDDEN) continue;
                        if (c < best_d) { best_d = c; arg = j; }
                    }
                    if (arg >= 0) { det_cand[i] = arg; det_rmsd[i] = best_d; }
                }
            }
        }
        // All places matched: nothing can do better.
        if (best_n == places) break;
    }
    }
    if (best_n >= 0) {
        for (int i = 0; i < n_ref; ++i) {
            if (out_cand) out_cand[i] = det_cand[i];
            if (out_rmsd) out_rmsd[i] = det_rmsd[i];
            if (out_matched) out_matched[i] = det_matched[i];
        }
    }

    out.n_matched = best_n < 0 ? 0 : best_n;
    out.rmsd = best_rmsd;
    out.inverted = best_inverted;
    out.same_packing = (out.n_matched == places);
    return out;
}

// Serial entry point.
KOKKOS_INLINE_FUNCTION
PairResult k_compare_pair(const BodyView& body,
                          const Placement& ca, const Placement* ref, int n_ref,
                          const Placement& cb, const Placement* pool, int n_pool,
                          const PairOptions& opts, double* cost,
                          int* out_cand = nullptr, double* out_rmsd = nullptr,
                          unsigned char* out_matched = nullptr) {
    return k_compare_pair_lanes(SerialLanes{}, body, ca, ref, n_ref, cb, pool, n_pool,
                                opts, cost, out_cand, out_rmsd, out_matched);
}

// Team entry point.
template <class TeamMember>
KOKKOS_INLINE_FUNCTION
PairResult k_compare_pair_team(const TeamMember& team, const BodyView& body,
                               const Placement& ca, const Placement* ref, int n_ref,
                               const Placement& cb, const Placement* pool, int n_pool,
                               const PairOptions& opts, double* cost,
                               int* out_cand = nullptr, double* out_rmsd = nullptr,
                               unsigned char* out_matched = nullptr) {
    return k_compare_pair_lanes(TeamLanes<TeamMember>{team}, body, ca, ref, n_ref,
                                cb, pool, n_pool, opts, cost, out_cand, out_rmsd,
                                out_matched);
}

}  // namespace compack
}  // namespace klasp
