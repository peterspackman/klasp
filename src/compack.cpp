#include "klasp/compack.hpp"

#include "klasp/compack_kernel.hpp"
#include "klasp/types.hpp"

#include <cstdio>
#include <cstdlib>

#include <Kokkos_Core.hpp>

#include "klasp/assignment.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/element.hpp"
#include "klasp/mat3.hpp"
#include "klasp/molecule.hpp"
#include "klasp/rotation.hpp"
#include "klasp/sgdata.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace klasp {
namespace {

inline void mat_mul(const double a[9], const double b[9], double out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[i * 3 + k] * b[k * 3 + j];
            out[i * 3 + j] = s;
        }
}

// out = a^T * b
inline void mat_tmul(const double a[9], const double b[9], double out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[k * 3 + i] * b[k * 3 + j];
            out[i * 3 + j] = s;
        }
}

inline double mat_det(const double m[9]) {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

inline void apply(const double r[9], const double t[3], const double p[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = r[i * 3 + 0] * p[0] + r[i * 3 + 1] * p[1] + r[i * 3 + 2] * p[2] + t[i];
}

// Jacobi eigendecomposition of a symmetric 3x3. `vec` columns are eigenvectors,
// `val` the eigenvalues, both ascending.
void sym_eigen(const double a_in[9], double val[3], double vec[9]) {
    double a[9];
    for (int i = 0; i < 9; ++i) a[i] = a_in[i];
    for (int i = 0; i < 9; ++i) vec[i] = (i % 4 == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 32; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j) off += a[i * 3 + j] * a[i * 3 + j];
        if (off < 1e-30) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                const double apq = a[p * 3 + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0)
                               / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
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
    // ascending, carrying the eigenvectors with them
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (val[j] < val[i]) {
                std::swap(val[i], val[j]);
                for (int k = 0; k < 3; ++k) std::swap(vec[k * 3 + i], vec[k * 3 + j]);
            }
}

// Does `q` carry every atom of the body onto an atom of the same element?
// Reflection in the plane of three points, expressed about the origin the
// points are already centred on. False when they are collinear and there is no
// plane to speak of.
bool plane_reflection(const double p[9], double out[9]) {
    const double u[3] = {p[3]-p[0], p[4]-p[1], p[5]-p[2]};
    const double v[3] = {p[6]-p[0], p[7]-p[1], p[8]-p[2]};
    double n[3] = {u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]};
    const double len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (!(len > 1e-12)) return false;
    for (int i = 0; i < 3; ++i) n[i] /= len;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i*3+j] = (i == j ? 1.0 : 0.0) - 2.0 * n[i] * n[j];
    return true;
}

bool is_self_op(const RigidBody& body, const double q[9], double tol) {
    const double tol2 = tol * tol;
    for (int k = 0; k < body.m; ++k) {
        const double* p = &body.positions[k * 3];
        double moved[3];
        for (int i = 0; i < 3; ++i)
            moved[i] = q[i * 3 + 0] * p[0] + q[i * 3 + 1] * p[1] + q[i * 3 + 2] * p[2];
        bool hit = false;
        for (int l = 0; l < body.m && !hit; ++l) {
            if (body.numbers[l] != body.numbers[k]) continue;
            const double dx = moved[0] - body.positions[l * 3 + 0];
            const double dy = moved[1] - body.positions[l * 3 + 1];
            const double dz = moved[2] - body.positions[l * 3 + 2];
            if (dx * dx + dy * dy + dz * dz <= tol2) hit = true;
        }
        if (!hit) return false;
    }
    return true;
}

// Whether `q` matches an operation the body already has: ||Q1 - Q2||_F * radius
// within the is_self_op tolerance. Comparing the action on the atoms instead
// would merge a planar body's mirror with the identity.
bool already_have(const std::vector<double>& ops, const double q[9],
                  const RigidBody& body, double tol) {
    // RMS atom distance from the centroid, guarded for a single-atom body.
    const double radius = (body.m > 0 && body.trace_g > 0)
                        ? std::sqrt(body.trace_g / body.m) : 1.0;
    const double limit = tol / (radius > 1e-9 ? radius : 1.0);
    const double limit2 = limit * limit;
    for (std::size_t o = 0; o + 9 <= ops.size(); o += 9) {
        double d2 = 0.0;
        for (int i = 0; i < 9; ++i) { const double e = ops[o + i] - q[i]; d2 += e * e; }
        if (d2 < limit2) return true;
    }
    return false;
}

// (Q G)^T for every stored operation. G is symmetric, so (Q G)^T = G Q^T.
void build_qgt(RigidBody& body) {
    const int n = body.n_self_ops();
    body.qgt.assign(static_cast<std::size_t>(n) * 9, 0.0);
    for (int o = 0; o < n; ++o) {
        double qg[9];
        mat_mul(&body.self_ops[o * 9], body.gyration, qg);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) body.qgt[o * 9 + i * 3 + j] = qg[j * 3 + i];
    }
}

}  // namespace

bool heavy_atoms_suffice(const double* positions, const int* numbers, int n) {
    constexpr double kOffLine = 0.05;   // Angstrom
    int a = -1, b = -1;
    for (int i = 0; i < n; ++i) {
        if (numbers[i] <= 1) continue;
        if (a < 0) a = i;
        else if (b < 0) b = i;
    }
    if (b < 0) return false;   // fewer than two heavy atoms: no axis at all
    double u[3], len = 0;
    for (int d = 0; d < 3; ++d) {
        u[d] = positions[b * 3 + d] - positions[a * 3 + d];
        len += u[d] * u[d];
    }
    len = std::sqrt(len);
    for (int d = 0; d < 3; ++d) u[d] /= len;
    const auto off_line = [&](int i) {
        double v[3], along = 0;
        for (int d = 0; d < 3; ++d) {
            v[d] = positions[i * 3 + d] - positions[a * 3 + d];
            along += v[d] * u[d];
        }
        double perp = 0;
        for (int d = 0; d < 3; ++d) perp += (v[d] - along * u[d]) * (v[d] - along * u[d]);
        return std::sqrt(perp) > kOffLine;
    };
    bool any_off = false;
    for (int i = 0; i < n; ++i) {
        if (!off_line(i)) continue;
        if (numbers[i] > 1) return true;   // the heavy atoms span a plane
        any_off = true;
    }
    return !any_off;
}

std::vector<DedupConfig::PddBlock> pdd_symmetry_blocks(const BasisConfig& basis,
                                                       double tol) {
    std::vector<DedupConfig::PddBlock> blocks;
    const double B = static_cast<double>(BOHR_TO_ANGSTROM);
    const int n_mols = std::max(1, basis.n_mols);
    int first_row = 0;
    for (int m = 0; m < n_mols; ++m) {
        const int ns = (basis.n_mols > 1 && m < static_cast<int>(basis.mol_n_sites.size()))
                     ? basis.mol_n_sites[m] : basis.n_sites;
        std::vector<double> pos;
        std::vector<int> z;
        for (int j = 0; j < ns; ++j) {
            const int flat = m * basis.n_sites + j;
            for (int d = 0; d < 3; ++d) pos.push_back(basis.body_positions[flat][d] * B);
            z.push_back(element(basis.elements[flat]).atomic_number);
        }
        RigidBody body = make_rigid_body(pos.data(), z.data(), ns, /*heavy_only=*/false);
        detect_self_symmetry(body, tol);

        DedupConfig::PddBlock blk;
        blk.first_row = first_row;
        for (int o = 0; o < body.n_self_ops(); ++o) {
            const double* q = &body.self_ops[o * 9];
            std::vector<int> perm(ns, -1);
            for (int i = 0; i < ns; ++i) {
                double moved[3] = {0, 0, 0};
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) moved[r] += q[r * 3 + c] * body.positions[i * 3 + c];
                double best = tol;
                for (int j = 0; j < ns; ++j) {
                    if (body.numbers[j] != body.numbers[i]) continue;
                    double d2 = 0;
                    for (int d = 0; d < 3; ++d) {
                        const double e = moved[d] - body.positions[j * 3 + d];
                        d2 += e * e;
                    }
                    if (std::sqrt(d2) <= best) { best = std::sqrt(d2); perm[i] = j; }
                }
            }
            // Not a relabelling: drop it.
            if (std::find(perm.begin(), perm.end(), -1) != perm.end()) continue;
            if (std::find(blk.perms.begin(), blk.perms.end(), perm) == blk.perms.end())
                blk.perms.push_back(std::move(perm));
        }
        if (blk.perms.empty()) {
            std::vector<int> identity(ns);
            for (int i = 0; i < ns; ++i) identity[i] = i;
            blk.perms.push_back(std::move(identity));
        }
        blocks.push_back(std::move(blk));
        first_row += ns;
    }
    return blocks;
}

RigidBody make_rigid_body(const double* positions, const int* numbers, int n,
                          bool heavy_only) {
    RigidBody body;
    for (int i = 0; i < n; ++i) {
        if (heavy_only && numbers[i] <= 1) continue;
        body.numbers.push_back(numbers[i]);
        for (int d = 0; d < 3; ++d) body.positions.push_back(positions[i * 3 + d]);
    }
    body.m = static_cast<int>(body.numbers.size());
    if (body.m == 0) return body;

    // Centroid to the origin: the RMSD identity drops its cross term only when
    // sum_k p_k vanishes.
    double c[3] = {0, 0, 0};
    for (int k = 0; k < body.m; ++k)
        for (int d = 0; d < 3; ++d) c[d] += body.positions[k * 3 + d];
    for (int d = 0; d < 3; ++d) c[d] /= body.m;
    for (int k = 0; k < body.m; ++k)
        for (int d = 0; d < 3; ++d) body.positions[k * 3 + d] -= c[d];

    for (int k = 0; k < body.m; ++k) {
        const double* p = &body.positions[k * 3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) body.gyration[i * 3 + j] += p[i] * p[j];
    }
    body.trace_g = body.gyration[0] + body.gyration[4] + body.gyration[8];

    for (int d = 0; d < 3; ++d) body.origin_offset[d] = c[d];
    body.self_ops.assign({1, 0, 0, 0, 1, 0, 0, 0, 1});   // identity, always first
    build_qgt(body);
    return body;
}

void detect_self_symmetry(RigidBody& body, double tol) {
    if (body.m < 2) return;
    body.self_ops.assign({1, 0, 0, 0, 1, 0, 0, 0, 1});

    auto consider = [&](const double q[9]) {
        // No point group has more than 48 operations; the device kernel indexes
        // a fixed MAX_OPS table.
        if (body.n_self_ops() >= compack::MAX_OPS) return;
        // Must be orthogonal; is_self_op alone would accept a projection onto
        // the atoms' plane.
        if (std::fabs(std::fabs(mat_det(q)) - 1.0) > 1e-6) return;
        if (already_have(body.self_ops, q, body, tol)) return;
        if (!is_self_op(body, q, tol)) return;
        for (int i = 0; i < 9; ++i) body.self_ops.push_back(q[i]);
    };

    // Operations whose atom permutation is the identity, which a triple-based
    // search cannot find: the mirror in the plane of a planar body.
    {
        double val[3], vec[9];
        sym_eigen(body.gyration, val, vec);
        // sqrt of G's smallest eigenvalue bounds every atom's distance from the
        // plane, so nearly planar molecules within tolerance get the mirror too.
        if (std::sqrt(std::max(val[0], 0.0)) < tol) {
            const double n[3] = {vec[0], vec[3], vec[6]};   // column 0
            double q[9];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    q[i * 3 + j] = (i == j ? 1.0 : 0.0) - 2.0 * n[i] * n[j];
            consider(q);
        }
        // A collinear body's symmetry group is continuous, but G = lambda e e^T,
        // so tr(M Q G) depends only on Q e = +/- e: the identity and (when a
        // self-operation) the inversion are exact representatives.
        // The two smallest eigenvalues together bound atom distance from the axis.
        if (std::sqrt(std::max(val[0] + val[1], 0.0)) < tol) {
            const double axis[3] = {vec[2], vec[5], vec[8]};   // the non-null direction
            double q[9];
            // sigma_v, a mirror containing the axis: always a self-operation.
            const double n1[3] = {vec[1], vec[4], vec[7]};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    q[i * 3 + j] = (i == j ? 1.0 : 0.0) - 2.0 * n1[i] * n1[j];
            consider(q);
            // sigma_h, the mirror across the axis, and the inversion: self
            // operations only for a palindromic body, which is_self_op decides.
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    q[i * 3 + j] = (i == j ? 1.0 : 0.0) - 2.0 * axis[i] * axis[j];
            consider(q);
            for (int i = 0; i < 9; ++i) q[i] = (i % 4 == 0) ? -1.0 : 0.0;
            consider(q);
        }
    }

    // Operations that permute atoms, built from where a non-collinear reference
    // triple can go (an atom maps only to one of the same element and radius).
    if (body.m >= 3) {
        auto radius = [&](int k) {
            const double* p = &body.positions[k * 3];
            return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        };
        auto dist = [&](int a, int b) {
            const double dx = body.positions[a * 3 + 0] - body.positions[b * 3 + 0];
            const double dy = body.positions[a * 3 + 1] - body.positions[b * 3 + 1];
            const double dz = body.positions[a * 3 + 2] - body.positions[b * 3 + 2];
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };
        // Reference triple: furthest-out atom, then the two that span best.
        int i0 = 0;
        for (int k = 1; k < body.m; ++k) if (radius(k) > radius(i0)) i0 = k;
        int i1 = -1;
        for (int k = 0; k < body.m; ++k)
            if (k != i0 && (i1 < 0 || dist(i0, k) > dist(i0, i1))) i1 = k;
        int i2 = -1;
        double best_area = 0.0;
        for (int k = 0; k < body.m; ++k) {
            if (k == i0 || k == i1) continue;
            // |(p1-p0) x (pk-p0)|: distance of this atom from the i0-i1 line
            double u[3], v[3];
            for (int d = 0; d < 3; ++d) {
                u[d] = body.positions[i1 * 3 + d] - body.positions[i0 * 3 + d];
                v[d] = body.positions[k * 3 + d] - body.positions[i0 * 3 + d];
            }
            const double cx = u[1] * v[2] - u[2] * v[1];
            const double cy = u[2] * v[0] - u[0] * v[2];
            const double cz = u[0] * v[1] - u[1] * v[0];
            const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
            if (area > best_area) { best_area = area; i2 = k; }
        }
        if (i2 >= 0 && best_area > 1e-6) {
            const int ref[3] = {i0, i1, i2};
            const double r01 = dist(i0, i1), r02 = dist(i0, i2), r12 = dist(i1, i2);
            double from[9];
            for (int a = 0; a < 3; ++a)
                for (int d = 0; d < 3; ++d) from[a * 3 + d] = body.positions[ref[a] * 3 + d];
            for (int j0 = 0; j0 < body.m; ++j0) {
                if (body.numbers[j0] != body.numbers[i0]) continue;
                if (std::fabs(radius(j0) - radius(i0)) > tol) continue;
                for (int j1 = 0; j1 < body.m; ++j1) {
                    if (j1 == j0 || body.numbers[j1] != body.numbers[i1]) continue;
                    if (std::fabs(dist(j0, j1) - r01) > tol) continue;
                    for (int j2 = 0; j2 < body.m; ++j2) {
                        if (j2 == j0 || j2 == j1) continue;
                        if (body.numbers[j2] != body.numbers[i2]) continue;
                        if (std::fabs(dist(j0, j2) - r02) > tol) continue;
                        if (std::fabs(dist(j1, j2) - r12) > tol) continue;
                        const int img[3] = {j0, j1, j2};
                        double onto[9];
                        for (int a = 0; a < 3; ++a)
                            for (int d = 0; d < 3; ++d)
                                onto[a * 3 + d] = body.positions[img[a] * 3 + d];
                        // Both triples are about the body origin, so the
                        // translation is discarded.
                        double q[9], t[3];
                        kabsch(from, onto, 3, q, t, /*allow_improper=*/true);
                        consider(q);
                        // A centred triple cannot decide what happens across its
                        // plane, so also try the reflection in that plane; the
                        // whole body decides which is a symmetry.
                        double refl[9];
                        if (plane_reflection(from, refl)) {
                            double alt[9];
                            mat_mul(q, refl, alt);
                            consider(alt);
                        }
                    }
                }
            }
        }
    }

    // Close under composition so the stored set is the whole group.
    for (int pass = 0; pass < 4; ++pass) {
        const int n = body.n_self_ops();
        const std::vector<double> current = body.self_ops;
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b) {
                double q[9];
                mat_mul(&current[a * 9], &current[b * 9], q);
                consider(q);
            }
        if (body.n_self_ops() == n) break;
        if (body.n_self_ops() > 96) break;   // guard: no molecular point group is larger
    }
    build_qgt(body);
}

double molecule_rmsd(const RigidBody& body, const PlacedMolecule& a,
                     const PlacedMolecule& b) {
    double dc2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        const double e = a.centroid[d] - b.centroid[d];
        dc2 += e * e;
    }
    if (body.m == 0) return std::sqrt(dc2);

    // M_a^T M_b, then the best tr(M_a^T M_b Q G) over the body's own operations.
    double mtm[9];
    mat_tmul(a.orient, b.orient, mtm);
    double best = -std::numeric_limits<double>::max();
    const int n_ops = body.n_self_ops();
    for (int o = 0; o < n_ops; ++o) {
        double qg[9], mq[9];
        mat_mul(&body.self_ops[o * 9], body.gyration, qg);
        mat_mul(mtm, qg, mq);
        const double tr = mq[0] + mq[4] + mq[8];
        if (tr > best) best = tr;
    }
    const double orient_term = 2.0 * (body.trace_g - best) / body.m;
    const double v = dc2 + (orient_term > 0.0 ? orient_term : 0.0);
    return std::sqrt(v > 0.0 ? v : 0.0);
}

double kabsch(const double* from, const double* onto, int n,
              double rotation[9], double translation[3], bool allow_improper) {
    for (int i = 0; i < 9; ++i) rotation[i] = (i % 4 == 0) ? 1.0 : 0.0;
    translation[0] = translation[1] = translation[2] = 0.0;
    if (n <= 0) return 0.0;

    double cf[3] = {0, 0, 0}, co[3] = {0, 0, 0};
    for (int k = 0; k < n; ++k)
        for (int d = 0; d < 3; ++d) { cf[d] += from[k * 3 + d]; co[d] += onto[k * 3 + d]; }
    for (int d = 0; d < 3; ++d) { cf[d] /= n; co[d] /= n; }

    // H = sum_k (from_k - cf) (onto_k - co)^T
    double h[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                h[i * 3 + j] += (from[k * 3 + i] - cf[i]) * (onto[k * 3 + j] - co[j]);

    // SVD via the symmetric eigenproblem: H^T H = V S^2 V^T, then U = H V S^-1.
    double hth[9];
    mat_tmul(h, h, hth);
    double val[3], vec[9];
    sym_eigen(hth, val, vec);
    // descending, so the rank-deficient direction is last
    double s[3], v[9];
    for (int i = 0; i < 3; ++i) {
        s[i] = std::sqrt(val[2 - i] > 0 ? val[2 - i] : 0.0);
        for (int r = 0; r < 3; ++r) v[r * 3 + i] = vec[r * 3 + (2 - i)];
    }
    // Fix the handedness of V so a rank-deficient (planar) fit's determinant is
    // set by the points, not by eigenvector signs. A full-rank fit is unaffected.
    if (mat_det(v) < 0.0)
        for (int r = 0; r < 3; ++r) v[r * 3 + 2] = -v[r * 3 + 2];

    double u[9];
    const double smax = s[0] > 0 ? s[0] : 1.0;
    int rank = 0;
    // Singular values come from H^T H, squaring the condition number, so a true
    // zero appears as about sqrt(eps) * smax; the cut must sit above that.
    for (int i = 0; i < 3; ++i) if (s[i] > 1e-6 * smax) ++rank;
    for (int i = 0; i < 3; ++i) {
        if (i < rank) {
            for (int r = 0; r < 3; ++r) {
                double acc = 0.0;
                for (int c = 0; c < 3; ++c) acc += h[r * 3 + c] * v[c * 3 + i];
                u[r * 3 + i] = acc / s[i];
            }
        } else {
            for (int r = 0; r < 3; ++r) u[r * 3 + i] = 0.0;
        }
    }
    // Fill null columns of U by orthogonality (planar or collinear points). For
    // collinear points the second column is any unit vector perpendicular to the first.
    if (rank == 1) {
        int a = 0;
        for (int i = 1; i < 3; ++i)
            if ((u[i * 3] < 0 ? -u[i * 3] : u[i * 3]) < (u[a * 3] < 0 ? -u[a * 3] : u[a * 3])) a = i;
        double w[3] = {0.0, 0.0, 0.0};
        w[a] = 1.0;
        const double along = u[a * 3];
        for (int r = 0; r < 3; ++r) w[r] -= along * u[r * 3];
        const double nw = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
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

    // R = V * diag(1,1,d) * U^T
    auto build = [&](double d) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) {
                    const double vk = (k == 2) ? v[i * 3 + k] * d : v[i * 3 + k];
                    acc += vk * u[j * 3 + k];
                }
                rotation[i * 3 + j] = acc;
            }
    };
    build(1.0);
    if (!allow_improper && mat_det(rotation) < 0.0) build(-1.0);

    for (int d = 0; d < 3; ++d) {
        translation[d] = co[d];
        for (int k = 0; k < 3; ++k) translation[d] -= rotation[d * 3 + k] * cf[k];
    }

    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        double moved[3];
        apply(rotation, translation, &from[k * 3], moved);
        for (int d = 0; d < 3; ++d) {
            const double e = moved[d] - onto[k * 3 + d];
            sum += e * e;
        }
    }
    return std::sqrt(sum / n);
}

namespace {

compack::Placement to_kernel(const PlacedMolecule& m) {
    compack::Placement p;
    for (int i = 0; i < 3; ++i) p.c[i] = m.centroid[i];
    for (int i = 0; i < 9; ++i) p.M[i] = m.orient[i];
    p.component = m.component;
    return p;
}

}  // namespace

compack::BodyView body_view(const RigidBody& b) {
    compack::BodyView v;
    v.m = b.m;
    v.n_ops = b.n_self_ops();
    for (int i = 0; i < 9; ++i) v.gyration[i] = b.gyration[i];
    v.trace_g = b.trace_g;
    v.self_ops = b.self_ops.data();
    v.qgt = b.qgt.data();
    return v;
}

DeviceBody::DeviceBody(const RigidBody& b)
    : scalars_(body_view(b)),
      ops_("compack_ops", b.self_ops.size()),
      qgt_("compack_qgt", b.qgt.size()) {
    // Clear the host pointers body_view filled so scalars_ can never carry one
    // onto a device.
    scalars_.self_ops = nullptr;
    scalars_.qgt = nullptr;

    auto h_ops = Kokkos::create_mirror_view(ops_);
    auto h_qgt = Kokkos::create_mirror_view(qgt_);
    for (std::size_t k = 0; k < b.self_ops.size(); ++k) h_ops(k) = b.self_ops[k];
    for (std::size_t k = 0; k < b.qgt.size(); ++k) h_qgt(k) = b.qgt[k];
    Kokkos::deep_copy(ops_, h_ops);
    Kokkos::deep_copy(qgt_, h_qgt);
}

compack::BodyView DeviceBody::view() const {
    compack::BodyView v = scalars_;
    v.self_ops = ops_.data();
    v.qgt = qgt_.data();
    return v;
}

// The host path is an adapter over the device-callable kernel.
int kernel_reference_count(const PackingCluster& a, const CompackOptions& opts) {
    const int n_kept = static_cast<int>(a.neighbours.size());
    return a.untied >= 0 ? std::min(n_kept, compack::MAX_NEIGH)
                         : std::min({opts.n_neighbours, n_kept, compack::MAX_NEIGH});
}

compack::PairOptions kernel_pair_options(const PackingCluster& a, const CompackOptions& opts) {
    compack::PairOptions ko;
    ko.n_neighbours = a.untied >= 0 ? std::min(a.scored, kernel_reference_count(a, opts))
                                    : opts.n_neighbours;
    ko.n_untied = a.untied;
    ko.match_tolerance = opts.match_tolerance;
    ko.max_iterations = opts.max_iterations;
    ko.allow_inversion = opts.allow_inversion;
    return ko;
}

CompackResult compack_compare(const PackingCluster& a, const PackingCluster& b,
                              const RigidBody& body, const CompackOptions& opts) {
    CompackResult out;
    // Clusters are nearest-first, so capping drops only the most distant candidates.
    const int n_ref = kernel_reference_count(a, opts);
    const int n_pool = std::min<int>(static_cast<int>(b.neighbours.size()), compack::MAX_POOL);
    if (n_ref <= 0 || n_pool <= 0 || body.m == 0) return out;

    std::vector<compack::Placement> ref, pool;
    ref.reserve(n_ref);
    pool.reserve(n_pool);
    for (int i = 0; i < n_ref; ++i) ref.push_back(to_kernel(a.neighbours[i]));
    for (int j = 0; j < n_pool; ++j) pool.push_back(to_kernel(b.neighbours[j]));

    const compack::BodyView bv = body_view(body);
    const compack::PairOptions ko = kernel_pair_options(a, opts);

    std::vector<double> cost(static_cast<std::size_t>(n_ref) * n_pool);
    std::vector<int> cand(n_ref, -1);
    std::vector<double> rmsd(n_ref, 0.0);
    std::vector<unsigned char> matched(n_ref, 0);

    const compack::PairResult r = compack::k_compare_pair(
        bv, to_kernel(a.central), ref.data(), n_ref,
        to_kernel(b.central), pool.data(), n_pool, ko,
        cost.data(), cand.data(), rmsd.data(), matched.data());

    out.valid = r.valid;
    out.n_matched = r.n_matched;
    out.n_neighbours = r.n_neighbours;
    out.rmsd = r.rmsd;
    out.same_packing = r.same_packing;
    out.inverted = r.inverted;
    if (!r.valid) return out;
    for (int i = 0; i < n_ref; ++i) {
        if (cand[i] < 0) continue;
        out.matches.push_back({i, cand[i], rmsd[i], matched[i] != 0});
    }
    return out;
}

// --- clusters from a stored structure ---------------------------------------

PlacedMolecule place_molecule(const RigidBody& body, const double cart_com[3],
                              const double orient[9], int component) {
    PlacedMolecule m;
    m.component = component;
    for (int i = 0; i < 9; ++i) m.orient[i] = orient[i];
    // Shift by the rotated offset from the basis frame to the counted centroid.
    for (int i = 0; i < 3; ++i) {
        m.centroid[i] = cart_com[i];
        for (int k = 0; k < 3; ++k) m.centroid[i] += orient[i * 3 + k] * body.origin_offset[k];
    }
    return m;
}

PackingCluster build_cluster_from_cell(const double cell_in[9],
                                       const std::vector<PlacedMolecule>& cell_mols,
                                       int central, const ClusterSpec& spec) {
    PackingCluster out;
    if (cell_mols.empty()) return out;
    if (central < 0 || central >= static_cast<int>(cell_mols.size())) return out;

    // cart = cell * frac: the rows of the inverse are reciprocal vectors, and
    // |row i| * radius bounds the image range along axis i.
    double inv[9] = {};
    if (!mat3_inv(cell_in, inv)) return out;   // degenerate cell: nothing to build
    int reach[3];
    for (int i = 0; i < 3; ++i)
        reach[i] = static_cast<int>(std::ceil(spec.radius * std::sqrt(
            inv[i * 3] * inv[i * 3] + inv[i * 3 + 1] * inv[i * 3 + 1]
            + inv[i * 3 + 2] * inv[i * 3 + 2])));

    out.central = cell_mols[central];
    const PlacedMolecule& ctr = cell_mols[central];

    // Molecules are kept whole, so count each molecule's shifts from its image
    // nearest the central one.
    struct Candidate { double d2; PlacedMolecule m; };
    std::vector<Candidate> cands;
    const double r2max = spec.radius * spec.radius;
    for (std::size_t k = 0; k < cell_mols.size(); ++k) {
        double sep[3], f[3];
        for (int i = 0; i < 3; ++i) sep[i] = ctr.centroid[i] - cell_mols[k].centroid[i];
        mat3_vec(inv, sep, f);
        const int base[3] = {static_cast<int>(std::lround(f[0])),
                             static_cast<int>(std::lround(f[1])),
                             static_cast<int>(std::lround(f[2]))};
        for (int ia = base[0] - reach[0]; ia <= base[0] + reach[0]; ++ia)
            for (int ib = base[1] - reach[1]; ib <= base[1] + reach[1]; ++ib)
                for (int ic = base[2] - reach[2]; ic <= base[2] + reach[2]; ++ic) {
                    const double n[3] = {double(ia), double(ib), double(ic)};
                    double shift[3];
                    mat3_vec(cell_in, n, shift);
                    const bool self_image = (ia == 0 && ib == 0 && ic == 0
                                             && static_cast<int>(k) == central);
                    if (self_image) continue;   // the central molecule is not its own neighbour
                    Candidate c;
                    c.m = cell_mols[k];
                    double d2 = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        c.m.centroid[i] = cell_mols[k].centroid[i] + shift[i];
                        const double e = c.m.centroid[i] - ctr.centroid[i];
                        d2 += e * e;
                    }
                    if (d2 > r2max) continue;
                    c.d2 = d2;
                    cands.push_back(c);
                }
            }

    // Nearest first; stable so exact ties keep enumeration order.
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Candidate& x, const Candidate& y) { return x.d2 < y.d2; });
    // The N nearest plus every molecule tied with the Nth (within 1e-4 A).
    constexpr double kShellTie = 1e-4;   // Angstrom
    const int n = std::min<int>(spec.n_neighbours, static_cast<int>(cands.size()));
    int keep = n, untied = n;
    if (n > 0) {
        const double edge = std::sqrt(cands[n - 1].d2);
        const auto tied = [&](int i) { return std::fabs(std::sqrt(cands[i].d2) - edge) <= kShellTie; };
        while (keep < static_cast<int>(cands.size()) && tied(keep)) ++keep;
        if (keep > n)
            while (untied > 0 && tied(untied - 1)) --untied;
    }
    out.neighbours.reserve(keep);
    for (int i = 0; i < keep; ++i) out.neighbours.push_back(cands[i].m);
    out.untied = (keep > n) ? untied : -1;
    out.scored = n;
    return out;
}

PackingCluster build_cluster(const StructureGeometry& g,
                             const std::vector<RigidBody>& bodies,
                             int central, const ClusterSpec& spec) {
    PackingCluster out;
    if (g.n_mols <= 0 || !g.translations || !g.rotations) return out;
    if (central < 0 || central >= g.n_mols) return out;

    RealType cp[6];
    for (int i = 0; i < 6; ++i) cp[i] = static_cast<RealType>(g.cell_params[i]);
    RealType cell_rt[3][3];
    cell_params_to_matrix(cp, cell_rt);
    double cell[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cell[i * 3 + j] = static_cast<double>(cell_rt[i][j]);
    RealType cell_inv_rt[3][3] = {};
    mat3_inv(cell_rt, cell_inv_rt);  // zeros on a degenerate cell, as before

    const auto& sg = get_space_group(g.space_group);

    // Every molecule of the unit cell; operators may be improper.
    std::vector<PlacedMolecule> cell_mols;
    cell_mols.reserve(static_cast<std::size_t>(sg.n_symops) * g.n_mols);
    int central_index = -1;
    for (int op = 0; op < sg.n_symops; ++op) {
        double s_frac[3][3], t_sym[3];
        get_symop(sg, op, s_frac, t_sym);
        RealType s_frac_rt[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) s_frac_rt[i][j] = static_cast<RealType>(s_frac[i][j]);
        RealType s_cart_rt[3][3];
        frac_to_cart_symop(cell_rt, cell_inv_rt, s_frac_rt, s_cart_rt);

        for (int m = 0; m < g.n_mols; ++m) {
            if (m >= static_cast<int>(bodies.size())) continue;
            double frac[3];
            for (int i = 0; i < 3; ++i) {
                double f = t_sym[i];
                for (int j = 0; j < 3; ++j) f += s_frac[i][j] * g.translations[m * 3 + j];
                frac[i] = f - std::floor(f);
            }
            double cart[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) cart[i] += cell[i * 3 + j] * frac[j];

            RealType aa[3] = {static_cast<RealType>(g.rotations[m * 3 + 0]),
                              static_cast<RealType>(g.rotations[m * 3 + 1]),
                              static_cast<RealType>(g.rotations[m * 3 + 2])};
            RealType r_mol[3][3];
            angle_axis_to_matrix(aa, r_mol);
            double orient[9];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double s = 0.0;
                    for (int k = 0; k < 3; ++k)
                        s += static_cast<double>(s_cart_rt[i][k]) * static_cast<double>(r_mol[k][j]);
                    orient[i * 3 + j] = s;
                }

            if (op == 0 && m == central) central_index = static_cast<int>(cell_mols.size());
            cell_mols.push_back(place_molecule(bodies[m], cart, orient, m));
        }
    }
    if (central_index < 0) return out;
    return build_cluster_from_cell(cell, cell_mols, central_index, spec);
}

CompackResult compare_structures(const StructureGeometry& a, const StructureGeometry& b,
                                 const std::vector<RigidBody>& bodies,
                                 const CompackOptions& opts, const ClusterSpec& spec) {
    CompackResult best;
    ClusterSpec ref_spec = spec;
    ref_spec.n_neighbours = opts.n_neighbours;
    ClusterSpec pool_spec = spec;
    pool_spec.n_neighbours = candidate_pool_size(opts.n_neighbours);

    // B contributes the larger pool (see candidate_pool_size).
    const PackingCluster cb = build_cluster(b, bodies, 0, pool_spec);
    if (cb.neighbours.empty()) return best;

    for (int centre = 0; centre < a.n_mols; ++centre) {
        const PackingCluster ca = build_cluster(a, bodies, centre, ref_spec);
        if (ca.neighbours.empty()) continue;
        const CompackResult r = compack_compare(ca, cb, bodies[ca.central.component], opts);
        if (!r.valid) continue;
        const bool better = !best.valid
                         || r.n_matched > best.n_matched
                         || (r.n_matched == best.n_matched && r.rmsd < best.rmsd);
        if (better) best = r;
        if (best.same_packing) break;
    }
    return best;
}

// --- all-pairs sweep ---------------------------------------------------------

void compack_sweep(const SweepClusters& c, const RigidBody& body,
                   const CompackOptions& opts,
                   std::vector<int>& n_matched, std::vector<unsigned char>& same,
                   const std::function<void(long long, long long)>& progress) {
    const int n = c.n;
    n_matched.assign(static_cast<std::size_t>(n) * n, 0);
    same.assign(static_cast<std::size_t>(n) * n, 0);
    if (n <= 0 || body.m == 0) return;
    const int n_ref = std::min(c.n_ref, compack::MAX_NEIGH);
    const int n_pool = std::min(c.n_pool, compack::MAX_POOL);
    if (n_ref <= 0 || n_pool <= 0) return;
    const int places = std::min({opts.n_neighbours, n_ref, compack::MAX_NEIGH});
    for (int i = 0; i < n; ++i) {
        n_matched[static_cast<std::size_t>(i) * n + i] = places;
        same[static_cast<std::size_t>(i) * n + i] = 1;
    }
    if (n < 2) return;

    using Space = ExecSpace;
    using Team = Kokkos::TeamPolicy<Space>;

    // Placements and the body's operation table, device-side.
    Kokkos::View<compack::Placement*, Space> d_central("compack_central", n);
    Kokkos::View<compack::Placement*, Space> d_ref("compack_ref", (std::size_t)n * n_ref);
    Kokkos::View<compack::Placement*, Space> d_pool("compack_pool", (std::size_t)n * n_pool);
    Kokkos::View<int*, Space> d_count("compack_ref_count", n);
    Kokkos::View<int*, Space> d_untied("compack_untied", n);
    {
        auto h_count = Kokkos::create_mirror_view(d_count);
        auto h_untied = Kokkos::create_mirror_view(d_untied);
        for (int i = 0; i < n; ++i) {
            // Without a tied shell every structure is scored out of `places`;
            // missing references are padding that never matches.
            h_untied(i) = c.untied.empty() ? -1 : c.untied[i];
            h_count(i) = h_untied(i) >= 0 ? std::min(c.ref_count[i], n_ref) : places;
        }
        Kokkos::deep_copy(d_count, h_count);
        Kokkos::deep_copy(d_untied, h_untied);
    }
    {
        auto h_central = Kokkos::create_mirror_view(d_central);
        auto h_ref = Kokkos::create_mirror_view(d_ref);
        auto h_pool = Kokkos::create_mirror_view(d_pool);
        for (int i = 0; i < n; ++i) {
            h_central(i) = to_kernel(c.centrals[i]);
            for (int k = 0; k < n_ref; ++k)
                h_ref((std::size_t)i * n_ref + k) = to_kernel(c.ref[(std::size_t)i * c.n_ref + k]);
            for (int k = 0; k < n_pool; ++k)
                h_pool((std::size_t)i * n_pool + k) = to_kernel(c.pool[(std::size_t)i * c.n_pool + k]);
        }
        Kokkos::deep_copy(d_central, h_central);
        Kokkos::deep_copy(d_ref, h_ref);
        Kokkos::deep_copy(d_pool, h_pool);
    }

    // Named so the device tables outlive the launches below.
    const DeviceBody device_body(body);
    const compack::BodyView bv = device_body.view();
    compack::PairOptions ko;
    ko.n_neighbours = places;
    ko.match_tolerance = opts.match_tolerance;
    ko.max_iterations = opts.max_iterations;
    ko.allow_inversion = opts.allow_inversion;

    // The upper triangle, flattened, so every team has work.
    const long n_pairs = static_cast<long>(n) * (n - 1) / 2;
    Kokkos::View<int*, Space> d_nm("compack_nm", n_pairs);

    using ScratchD = Kokkos::View<double*, Space::scratch_memory_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const std::size_t cost_words = static_cast<std::size_t>(n_ref) * n_pool;
    const std::size_t scratch = ScratchD::shmem_size(cost_words);

    // One launch without a progress callback; with one, the range is chunked.
    long chunk = n_pairs;
    if (progress) chunk = std::max<long>(1024, n_pairs / 200);

    Team policy(static_cast<int>(std::min(chunk, n_pairs)), Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch));
    for (long begin = 0; begin < n_pairs; begin += chunk) {
        const long count = std::min(chunk, n_pairs - begin);
        Team chunk_policy(static_cast<int>(count), Kokkos::AUTO);
        chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch));
        Kokkos::parallel_for(
            "compack_sweep", chunk_policy,
            KOKKOS_LAMBDA(const Team::member_type& team) {
                const long pair = begin + team.league_rank();
                // Unflatten the upper triangle: row i, column j > i.
                long rem = pair;
                int i = 0;
                while (rem >= (n - 1 - i)) { rem -= (n - 1 - i); ++i; }
                const int j = static_cast<int>(i + 1 + rem);

                ScratchD buf(team.team_scratch(0), cost_words);
                compack::PairOptions kp = ko;
                kp.n_untied = d_untied(i);
                const compack::PairResult r = compack::k_compare_pair_team(
                    team, bv, d_central(i), &d_ref((std::size_t)i * n_ref), d_count(i),
                    d_central(j), &d_pool((std::size_t)j * n_pool), n_pool,
                    kp, buf.data());
                Kokkos::single(Kokkos::PerTeam(team),
                               [&]() { d_nm(pair) = r.valid ? r.n_matched : -1; });
            });
        if (progress) {
            // Fence per chunk so the reported count is accurate.
            Kokkos::fence();
            progress(begin + count, n_pairs);
        }
    }
    Kokkos::fence();

    auto h_nm = Kokkos::create_mirror_view(d_nm);
    Kokkos::deep_copy(h_nm, d_nm);
    long pair = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j, ++pair) {
            const int v = h_nm(pair);
            if (v < 0) continue;
            n_matched[static_cast<std::size_t>(i) * n + j] = v;
            n_matched[static_cast<std::size_t>(j) * n + i] = v;
            const unsigned char s = (v == places) ? 1 : 0;
            same[static_cast<std::size_t>(i) * n + j] = s;
            same[static_cast<std::size_t>(j) * n + i] = s;
        }
}

}  // namespace klasp
