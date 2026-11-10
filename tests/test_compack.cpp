#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/compack.hpp"
#include "klasp/constants.hpp"
#include "klasp/assignment.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;

namespace {

// Rotation about a unit axis by `angle`, row-major.
void axis_angle(const double axis[3], double angle, double r[9]) {
    const double n = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    const double x = axis[0]/n, y = axis[1]/n, z = axis[2]/n;
    const double c = std::cos(angle), s = std::sin(angle), t = 1 - c;
    r[0] = t*x*x + c;   r[1] = t*x*y - s*z; r[2] = t*x*z + s*y;
    r[3] = t*x*y + s*z; r[4] = t*y*y + c;   r[5] = t*y*z - s*x;
    r[6] = t*x*z - s*y; r[7] = t*y*z + s*x; r[8] = t*z*z + c;
}

// Reference definition: RMSD over the body's atoms placed by each molecule,
// minimised over the body's own symmetry operations.
double brute_force_rmsd(const RigidBody& body, const PlacedMolecule& a,
                        const PlacedMolecule& b) {
    double best = 1e300;
    for (int o = 0; o < body.n_self_ops(); ++o) {
        const double* q = &body.self_ops[o * 9];
        double sum = 0.0;
        for (int k = 0; k < body.m; ++k) {
            const double* p = &body.positions[k * 3];
            double qp[3];
            for (int i = 0; i < 3; ++i)
                qp[i] = q[i*3+0]*p[0] + q[i*3+1]*p[1] + q[i*3+2]*p[2];
            for (int i = 0; i < 3; ++i) {
                const double pa = a.centroid[i]
                    + a.orient[i*3+0]*p[0] + a.orient[i*3+1]*p[1] + a.orient[i*3+2]*p[2];
                const double pb = b.centroid[i]
                    + b.orient[i*3+0]*qp[0] + b.orient[i*3+1]*qp[1] + b.orient[i*3+2]*qp[2];
                const double e = pa - pb;
                sum += e * e;
            }
        }
        best = std::min(best, std::sqrt(sum / body.m));
    }
    return best;
}

// Planar hexagon of carbons in the xy plane — benzene's heavy skeleton, D6h.
RigidBody benzene_ring() {
    std::vector<double> p;
    std::vector<int> z;
    for (int k = 0; k < 6; ++k) {
        const double a = k * M_PI / 3.0;
        p.push_back(1.39 * std::cos(a));
        p.push_back(1.39 * std::sin(a));
        p.push_back(0.0);
        z.push_back(6);
    }
    RigidBody b = make_rigid_body(p.data(), z.data(), 6, /*heavy_only=*/true);
    detect_self_symmetry(b);
    return b;
}

// A lopsided, non-planar body with no symmetry.
RigidBody asymmetric_body() {
    const double p[] = {0.0, 0.0, 0.0,  1.5, 0.1, 0.2,  -0.4, 1.3, 0.7,  0.3, -0.2, -1.6};
    const int z[] = {6, 8, 7, 6};
    RigidBody b = make_rigid_body(p, z, 4, true);
    detect_self_symmetry(b);
    return b;
}

}  // namespace

TEST_CASE("Assignment returns the permutation, not just the cost", "[compack][assignment]") {
    // Cheapest choice is the anti-diagonal; a greedy pass would take (0,0).
    const double cost[9] = {1, 2, 3,
                            2, 4, 6,
                            3, 6, 2};
    std::vector<int> perm;
    const double total = min_cost_assignment(cost, 3, 3, &perm);
    REQUIRE(perm.size() == 3);
    std::vector<int> seen = perm;
    std::sort(seen.begin(), seen.end());
    REQUIRE(seen == std::vector<int>{0, 1, 2});
    double check = 0;
    for (int i = 0; i < 3; ++i) check += cost[i * 3 + perm[i]];
    REQUIRE_THAT(total, WithinAbs(check, 1e-12));
    // Optimal: brute force over all 6 permutations.
    std::vector<int> idx{0, 1, 2};
    double best = 1e300;
    do {
        double s = 0;
        for (int i = 0; i < 3; ++i) s += cost[i * 3 + idx[i]];
        best = std::min(best, s);
    } while (std::next_permutation(idx.begin(), idx.end()));
    REQUIRE_THAT(total, WithinAbs(best, 1e-12));
}

TEST_CASE("Assignment handles a pool wider than the reference", "[compack][assignment]") {
    // 2 rows, 4 columns, the overlay's shape.
    const double cost[8] = {9, 1, 9, 9,
                            9, 9, 9, 2};
    std::vector<int> perm;
    const double total = min_cost_assignment(cost, 2, 4, &perm);
    REQUIRE(perm == std::vector<int>{1, 3});
    REQUIRE_THAT(total, WithinAbs(3.0, 1e-12));
}

TEST_CASE("Kabsch recovers a known rotation", "[compack][kabsch]") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    std::vector<double> from(30), onto(30);
    for (auto& x : from) x = u(rng);

    const double axis[3] = {0.3, -0.5, 0.8};
    double r[9];
    axis_angle(axis, 0.7, r);
    const double t[3] = {1.5, -0.25, 3.0};
    for (int k = 0; k < 10; ++k)
        for (int i = 0; i < 3; ++i)
            onto[k*3+i] = r[i*3+0]*from[k*3+0] + r[i*3+1]*from[k*3+1] + r[i*3+2]*from[k*3+2] + t[i];

    double rr[9], tt[3];
    const double rmsd = kabsch(from.data(), onto.data(), 10, rr, tt, false);
    REQUIRE_THAT(rmsd, WithinAbs(0.0, 1e-9));
    for (int i = 0; i < 9; ++i) REQUIRE_THAT(rr[i], WithinAbs(r[i], 1e-9));
    for (int i = 0; i < 3; ++i) REQUIRE_THAT(tt[i], WithinAbs(t[i], 1e-9));
}

TEST_CASE("Kabsch stays proper on a planar point set unless allowed otherwise",
          "[compack][kabsch]") {
    // Planar inputs leave one direction undetermined; the fit must still be a
    // proper rotation.
    const double from[] = {0,0,0,  1,0,0,  0,1,0,  1,1,0};
    const double onto[] = {0,0,0, -1,0,0,  0,1,0, -1,1,0};   // mirrored in x
    double r[9], t[3];
    const double rmsd_proper = kabsch(from, onto, 4, r, t, /*allow_improper=*/false);
    const double det = r[0]*(r[4]*r[8]-r[5]*r[7]) - r[1]*(r[3]*r[8]-r[5]*r[6])
                     + r[2]*(r[3]*r[7]-r[4]*r[6]);
    REQUIRE(det > 0.0);
    // A planar set's mirror is reachable by a 180 degree turn out of plane.
    REQUIRE(rmsd_proper < 1e-9);

    double ri[9], ti[3];
    kabsch(from, onto, 4, ri, ti, /*allow_improper=*/true);
    const double deti = ri[0]*(ri[4]*ri[8]-ri[5]*ri[7]) - ri[1]*(ri[3]*ri[8]-ri[5]*ri[6])
                      + ri[2]*(ri[3]*ri[7]-ri[4]*ri[6]);
    REQUIRE(std::fabs(std::fabs(deti) - 1.0) < 1e-9);
}

TEST_CASE("Self-symmetry detection finds the known point groups", "[compack][symmetry]") {
    SECTION("a benzene ring is D6h: 24 operations") {
        REQUIRE(benzene_ring().n_self_ops() == 24);
    }
    SECTION("water's heavy atom alone has no orientation to speak of") {
        const double p[] = {0.0, 0.0, 0.0};
        const int z[] = {8};
        RigidBody b = make_rigid_body(p, z, 1, true);
        detect_self_symmetry(b);
        REQUIRE(b.m == 1);
    }
    SECTION("a planar C2v triatomic is C2v: E, C2, and two mirrors") {
        // Water with heavy-atom-only off, so all three atoms count.
        const double p[] = {0.0, 0.0, 0.117,  0.0, 0.757, -0.469,  0.0, -0.757, -0.469};
        const int z[] = {8, 1, 1};
        RigidBody b = make_rigid_body(p, z, 3, /*heavy_only=*/false);
        detect_self_symmetry(b);
        REQUIRE(b.n_self_ops() == 4);
    }
    SECTION("a lopsided body has only the identity") {
        REQUIRE(asymmetric_body().n_self_ops() == 1);
    }
    SECTION("every detected operation is orthogonal") {
        RigidBody b = benzene_ring();
        for (int o = 0; o < b.n_self_ops(); ++o) {
            const double* q = &b.self_ops[o * 9];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double s = 0;
                    for (int k = 0; k < 3; ++k) s += q[k*3+i] * q[k*3+j];
                    REQUIRE_THAT(s, WithinAbs(i == j ? 1.0 : 0.0, 1e-9));
                }
        }
    }
}

TEST_CASE("The closed-form RMSD agrees with the atom-by-atom definition",
          "[compack][rmsd]") {
    // Check the closed form against a literal sum over atoms, for symmetric and
    // asymmetric bodies and proper and improper placements.
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(-3.0, 3.0), ang(0.0, 6.283);

    for (RigidBody body : {benzene_ring(), asymmetric_body()}) {
        for (int trial = 0; trial < 40; ++trial) {
            PlacedMolecule a, b;
            const double ax1[3] = {u(rng), u(rng), u(rng)};
            const double ax2[3] = {u(rng), u(rng), u(rng)};
            axis_angle(ax1, ang(rng), a.orient);
            axis_angle(ax2, ang(rng), b.orient);
            for (int d = 0; d < 3; ++d) { a.centroid[d] = u(rng); b.centroid[d] = u(rng); }
            // Half the trials make B an improper copy.
            if (trial % 2) for (int i = 0; i < 9; ++i) b.orient[i] = -b.orient[i];

            INFO("body m=" << body.m << " ops=" << body.n_self_ops() << " trial=" << trial);
            REQUIRE_THAT(molecule_rmsd(body, a, b),
                         WithinAbs(brute_force_rmsd(body, a, b), 1e-9));
        }
    }
}

TEST_CASE("A planar body matches its own mirror image", "[compack][rmsd][symmetry]") {
    // A planar skeleton is carried onto itself by reflection in its plane, so
    // proper and improper copies at the same place are the same placement.
    RigidBody ring = benzene_ring();
    REQUIRE(ring.n_self_ops() > 1);

    PlacedMolecule a, b;
    const double axis[3] = {0.2, 0.9, -0.3};
    axis_angle(axis, 1.1, a.orient);
    for (int i = 0; i < 9; ++i) b.orient[i] = a.orient[i];
    for (int d = 0; d < 3; ++d) { a.centroid[d] = 0.0; b.centroid[d] = 0.0; }
    // b is a's mirror image through the ring plane: same placement.
    const double n[3] = {0, 0, 1};
    double refl[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) refl[i*3+j] = (i == j ? 1.0 : 0.0) - 2.0*n[i]*n[j];
    double mirrored[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += a.orient[i*3+k] * refl[k*3+j];
            mirrored[i*3+j] = s;
        }
    for (int i = 0; i < 9; ++i) b.orient[i] = mirrored[i];

    REQUIRE_THAT(molecule_rmsd(ring, a, b), WithinAbs(0.0, 1e-9));

    // A body with no such operation must not match its mirror.
    RigidBody lop = asymmetric_body();
    REQUIRE(lop.n_self_ops() == 1);
    PlacedMolecule c = a, d = a;
    for (int i = 0; i < 9; ++i) d.orient[i] = mirrored[i];
    REQUIRE(molecule_rmsd(lop, c, d) > 0.1);
}

namespace {

// A simple cluster: the central molecule at the origin and neighbours on a
// lattice around it, all with the same orientation.
PackingCluster lattice_cluster(const double r_orient[9], double a_len, int shells) {
    PackingCluster c;
    for (int i = 0; i < 9; ++i) c.central.orient[i] = r_orient[i];
    struct Nb { double d; PlacedMolecule m; };
    std::vector<Nb> nbs;
    for (int i = -shells; i <= shells; ++i)
        for (int j = -shells; j <= shells; ++j)
            for (int k = -shells; k <= shells; ++k) {
                if (i == 0 && j == 0 && k == 0) continue;
                PlacedMolecule m;
                for (int q = 0; q < 9; ++q) m.orient[q] = r_orient[q];
                m.centroid[0] = i * a_len; m.centroid[1] = j * a_len; m.centroid[2] = k * a_len;
                nbs.push_back({std::sqrt(double(i*i + j*j + k*k)) * a_len, m});
            }
    std::stable_sort(nbs.begin(), nbs.end(), [](const Nb& x, const Nb& y) { return x.d < y.d; });
    for (const auto& n : nbs) c.neighbours.push_back(n.m);
    return c;
}

}  // namespace

TEST_CASE("A cluster matches itself completely", "[compack]") {
    RigidBody body = asymmetric_body();
    const double eye[9] = {1,0,0, 0,1,0, 0,0,1};
    PackingCluster c = lattice_cluster(eye, 6.0, 2);

    CompackOptions opts;
    opts.n_neighbours = 15;
    CompackResult r = compack_compare(c, c, body, opts);
    REQUIRE(r.valid);
    REQUIRE(r.n_neighbours == 15);
    REQUIRE(r.n_matched == 15);
    REQUIRE(r.same_packing);
    // Not tighter than 1e-6: the orientation term is a sqrt of a cancelling
    // difference, so self-comparison has an absolute error near 1e-8.
    REQUIRE_THAT(r.rmsd, WithinAbs(0.0, 1e-6));
}

TEST_CASE("A rigidly moved cluster still matches completely", "[compack]") {
    // Same packing described in a different frame: the overlay has to find it.
    RigidBody body = asymmetric_body();
    const double eye[9] = {1,0,0, 0,1,0, 0,0,1};
    PackingCluster a = lattice_cluster(eye, 6.0, 2);

    const double axis[3] = {0.4, 0.6, -0.7};
    double rot[9];
    axis_angle(axis, 0.9, rot);
    const double shift[3] = {3.1, -2.2, 0.8};
    PackingCluster b;
    auto move = [&](const PlacedMolecule& m) {
        PlacedMolecule o;
        for (int i = 0; i < 3; ++i)
            o.centroid[i] = rot[i*3+0]*m.centroid[0] + rot[i*3+1]*m.centroid[1]
                          + rot[i*3+2]*m.centroid[2] + shift[i];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += rot[i*3+k] * m.orient[k*3+j];
                o.orient[i*3+j] = s;
            }
        return o;
    };
    b.central = move(a.central);
    for (const auto& m : a.neighbours) b.neighbours.push_back(move(m));

    CompackOptions opts;
    opts.n_neighbours = 15;
    CompackResult r = compack_compare(a, b, body, opts);
    REQUIRE(r.n_matched == 15);
    REQUIRE(r.same_packing);
    REQUIRE_THAT(r.rmsd, WithinAbs(0.0, 1e-7));
}

TEST_CASE("A genuinely different packing does not match", "[compack]") {
    RigidBody body = asymmetric_body();
    const double eye[9] = {1,0,0, 0,1,0, 0,0,1};
    PackingCluster a = lattice_cluster(eye, 6.0, 2);
    // A third of the spacing gone in one direction is far beyond 1 A.
    PackingCluster b = a;
    for (auto& m : b.neighbours) m.centroid[2] *= 0.65;

    CompackOptions opts;
    opts.n_neighbours = 15;
    CompackResult r = compack_compare(a, b, body, opts);
    REQUIRE(r.valid);
    REQUIRE_FALSE(r.same_packing);
    REQUIRE(r.n_matched < 15);
    // Unmatched references are still reported, with their nearest counterpart.
    REQUIRE(static_cast<int>(r.matches.size()) == r.n_neighbours);
}

TEST_CASE("Linear bodies are handled exactly, not conservatively", "[compack][symmetry]") {
    // A collinear body: G is rank one, so an operation can only send the axis to
    // +/- e, and the inversion is a self-operation.
    SECTION("a palindromic body (O=C=O) is invariant under inversion") {
        const double p[] = {0, 0, -1.16,  0, 0, 0.0,  0, 0, 1.16};
        const int z[] = {8, 6, 8};
        RigidBody b = make_rigid_body(p, z, 3, /*heavy_only=*/false);
        detect_self_symmetry(b);
        bool has_inversion = false;
        for (int o = 0; o < b.n_self_ops(); ++o) {
            const double* q = &b.self_ops[o * 9];
            bool inv = true;
            for (int i = 0; i < 9; ++i) {
                const double want = (i % 4 == 0) ? -1.0 : 0.0;
                if (std::fabs(q[i] - want) > 1e-9) inv = false;
            }
            if (inv) has_inversion = true;
        }
        REQUIRE(has_inversion);

        PlacedMolecule x, y;
        const double axis[3] = {0.3, 0.5, 0.8};
        axis_angle(axis, 0.6, x.orient);
        for (int i = 0; i < 9; ++i) y.orient[i] = -x.orient[i];
        REQUIRE_THAT(molecule_rmsd(b, x, y), WithinAbs(0.0, 1e-9));
    }

    SECTION("a non-palindromic body (H-C-N) is not") {
        const double p[] = {0, 0, -1.20,  0, 0, -0.14,  0, 0, 1.02};
        const int z[] = {1, 6, 7};
        RigidBody b = make_rigid_body(p, z, 3, /*heavy_only=*/false);
        detect_self_symmetry(b);
        PlacedMolecule x, y;
        const double axis[3] = {0.3, 0.5, 0.8};
        axis_angle(axis, 0.6, x.orient);
        for (int i = 0; i < 9; ++i) y.orient[i] = -x.orient[i];
        // Without the inversion, reversal is a different placement.
        REQUIRE(molecule_rmsd(b, x, y) > 0.5);
    }

    SECTION("the closed form still matches the definition for a linear body") {
        const double p[] = {0, 0, -1.16,  0, 0, 0.0,  0, 0, 1.16};
        const int z[] = {8, 6, 8};
        RigidBody b = make_rigid_body(p, z, 3, false);
        detect_self_symmetry(b);
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> u(-3.0, 3.0), ang(0.0, 6.283);
        for (int trial = 0; trial < 30; ++trial) {
            PlacedMolecule x, y;
            const double a1[3] = {u(rng), u(rng), u(rng)}, a2[3] = {u(rng), u(rng), u(rng)};
            axis_angle(a1, ang(rng), x.orient);
            axis_angle(a2, ang(rng), y.orient);
            for (int d = 0; d < 3; ++d) { x.centroid[d] = u(rng); y.centroid[d] = u(rng); }
            if (trial % 2) for (int i = 0; i < 9; ++i) y.orient[i] = -y.orient[i];
            INFO("trial " << trial);
            REQUIRE_THAT(molecule_rmsd(b, x, y), WithinAbs(brute_force_rmsd(b, x, y), 1e-9));
        }
    }
}

namespace {

StructureGeometry cubic_p1(double a, const double* tr, const double* rv) {
    StructureGeometry g;
    g.cell_params[0] = g.cell_params[1] = g.cell_params[2] = a;
    g.cell_params[3] = g.cell_params[4] = g.cell_params[5] = 90.0;
    g.space_group = 1;
    g.n_mols = 1;
    g.translations = tr;
    g.rotations = rv;
    return g;
}

}  // namespace

TEST_CASE("A cluster built from a stored row matches itself", "[compack][cluster]") {
    std::vector<RigidBody> bodies{asymmetric_body()};
    const double tr[3] = {0.1, 0.2, 0.3};
    const double rv[3] = {0.3, -0.4, 0.2};
    StructureGeometry g = cubic_p1(7.0, tr, rv);

    ClusterSpec spec;
    spec.n_neighbours = 15;
    spec.radius = 16.0;
    PackingCluster c = build_cluster(g, bodies, 0, spec);
    REQUIRE(c.scored == 15);
    REQUIRE(static_cast<int>(c.neighbours.size()) >= 15);
    // Nearest first, and the central molecule is not among its own neighbours.
    for (std::size_t i = 1; i < c.neighbours.size(); ++i) {
        auto d2 = [&](std::size_t k) {
            double s = 0;
            for (int j = 0; j < 3; ++j) {
                const double e = c.neighbours[k].centroid[j] - c.central.centroid[j];
                s += e * e;
            }
            return s;
        };
        REQUIRE(d2(i) >= d2(i - 1) - 1e-9);
        REQUIRE(d2(i) > 1e-9);
    }

    CompackOptions opts;
    opts.n_neighbours = 15;
    CompackResult r = compare_structures(g, g, bodies, opts, spec);
    REQUIRE(r.valid);
    REQUIRE(r.same_packing);
    REQUIRE(r.n_matched == 15);
    // Same floor as in "A cluster matches itself completely" above.
    REQUIRE_THAT(r.rmsd, WithinAbs(0.0, 1e-6));
}

TEST_CASE("The comparison is invariant to the cell origin", "[compack][cluster]") {
    // A uniform fractional shift is a change of origin: identical packing.
    std::vector<RigidBody> bodies{asymmetric_body()};
    const double tr_a[3] = {0.1, 0.2, 0.3};
    const double tr_b[3] = {0.6, 0.45, 0.85};
    const double rv[3] = {0.3, -0.4, 0.2};
    StructureGeometry a = cubic_p1(7.0, tr_a, rv);
    StructureGeometry b = cubic_p1(7.0, tr_b, rv);

    CompackOptions opts;
    opts.n_neighbours = 15;
    ClusterSpec spec; spec.radius = 16.0;
    CompackResult r = compare_structures(a, b, bodies, opts, spec);
    REQUIRE(r.same_packing);
}

TEST_CASE("A different cell is not the same packing", "[compack][cluster]") {
    std::vector<RigidBody> bodies{asymmetric_body()};
    const double tr[3] = {0.1, 0.2, 0.3};
    const double rv[3] = {0.3, -0.4, 0.2};
    StructureGeometry a = cubic_p1(7.0, tr, rv);
    StructureGeometry b = cubic_p1(9.0, tr, rv);   // 30% longer edges

    CompackOptions opts;
    opts.n_neighbours = 15;
    ClusterSpec spec; spec.radius = 20.0;
    CompackResult r = compare_structures(a, b, bodies, opts, spec);
    REQUIRE(r.valid);
    REQUIRE_FALSE(r.same_packing);
}

TEST_CASE("A centrosymmetric group places improper copies", "[compack][cluster]") {
    // P-1: half the molecules are inversion images with det -1 orientations.
    std::vector<RigidBody> bodies{asymmetric_body()};
    const double tr[3] = {0.13, 0.27, 0.31};
    const double rv[3] = {0.3, -0.4, 0.2};
    StructureGeometry g = cubic_p1(7.0, tr, rv);
    g.space_group = 2;

    ClusterSpec spec; spec.n_neighbours = 15; spec.radius = 16.0;
    PackingCluster c = build_cluster(g, bodies, 0, spec);
    REQUIRE(c.scored == 15);
    REQUIRE(static_cast<int>(c.neighbours.size()) >= 15);
    int improper = 0;
    for (const auto& m : c.neighbours) {
        const double* r = m.orient;
        const double det = r[0]*(r[4]*r[8]-r[5]*r[7]) - r[1]*(r[3]*r[8]-r[5]*r[6])
                         + r[2]*(r[3]*r[7]-r[4]*r[6]);
        if (det < 0) ++improper;
    }
    REQUIRE(improper > 0);

    CompackOptions opts; opts.n_neighbours = 15;
    CompackResult r = compare_structures(g, g, bodies, opts, spec);
    REQUIRE(r.same_packing);
}

#include "klasp/compack_kernel.hpp"

namespace {

// The kernel takes plain arrays; adapt a host cluster onto it.
klasp::compack::Placement to_placement(const PlacedMolecule& m) {
    klasp::compack::Placement p;
    for (int i = 0; i < 3; ++i) p.c[i] = m.centroid[i];
    for (int i = 0; i < 9; ++i) p.M[i] = m.orient[i];
    p.component = m.component;
    return p;
}

// Uses the library's own conversion.
klasp::compack::BodyView to_view(const RigidBody& b) { return body_view(b); }

// The one-pair path compiled for the device, as the reference for the sweep.
int device_serial_n_matched(const PackingCluster& a, const PackingCluster& b,
                            const RigidBody& body, const CompackOptions& opts) {
    using Space = klasp::ExecSpace;
    std::vector<klasp::compack::Placement> ref, pool;
    for (int q = 0; q < kernel_reference_count(a, opts); ++q)
        ref.push_back(to_placement(a.neighbours[q]));
    for (const auto& m : b.neighbours) pool.push_back(to_placement(m));
    const int n_ref = static_cast<int>(ref.size()), n_pool = static_cast<int>(pool.size());
    const klasp::compack::PairOptions kopts = kernel_pair_options(a, opts);

    Kokkos::View<klasp::compack::Placement*, Space> d_ref("d_ref", ref.size());
    Kokkos::View<klasp::compack::Placement*, Space> d_pool("d_pool", pool.size());
    Kokkos::View<klasp::compack::Placement*, Space> d_cen("d_cen", 2);
    auto h_ref = Kokkos::create_mirror_view(d_ref);
    auto h_pool = Kokkos::create_mirror_view(d_pool);
    auto h_cen = Kokkos::create_mirror_view(d_cen);
    for (size_t q = 0; q < ref.size(); ++q) h_ref(q) = ref[q];
    for (size_t q = 0; q < pool.size(); ++q) h_pool(q) = pool[q];
    h_cen(0) = to_placement(a.central);
    h_cen(1) = to_placement(b.central);
    Kokkos::deep_copy(d_ref, h_ref);
    Kokkos::deep_copy(d_pool, h_pool);
    Kokkos::deep_copy(d_cen, h_cen);

    Kokkos::View<double*, Space> d_cost("d_cost", (size_t)n_ref * n_pool);
    Kokkos::View<int*, Space> d_out("d_out", 1);
    const klasp::DeviceBody device_body(body);
    const klasp::compack::BodyView bv = device_body.view();
    Kokkos::parallel_for("compack_serial_on_device", Kokkos::RangePolicy<Space>(0, 1),
                         KOKKOS_LAMBDA(int) {
        const auto r = klasp::compack::k_compare_pair(
            bv, d_cen(0), d_ref.data(), n_ref, d_cen(1), d_pool.data(), n_pool,
            kopts, d_cost.data());
        d_out(0) = r.valid ? r.n_matched : -1;
    });
    Kokkos::fence();
    return Kokkos::create_mirror_view_and_copy(klasp::HostMemSpace{}, d_out)(0);
}

}  // namespace

TEST_CASE("The device-callable kernel agrees with the host implementation",
          "[compack][kernel]") {
    // The device kernel must reproduce the host path, including the atom-free
    // covariance.
    std::vector<RigidBody> bodies{asymmetric_body(), benzene_ring()};
    std::mt19937 rng(23);
    std::uniform_real_distribution<double> u01(0.0, 1.0), rot(-1.5, 1.5),
                                           cellv(6.0, 9.0);

    for (int which = 0; which < 2; ++which) {
        std::vector<RigidBody> one{bodies[which]};
        for (int sg : {1, 2, 14}) {
            for (int trial = 0; trial < 6; ++trial) {
                double tra[3] = {u01(rng), u01(rng), u01(rng)};
                double rva[3] = {rot(rng), rot(rng), rot(rng)};
                double trb[3] = {u01(rng), u01(rng), u01(rng)};
                double rvb[3] = {rot(rng), rot(rng), rot(rng)};
                const double a_len = cellv(rng);

                StructureGeometry ga = cubic_p1(a_len, tra, rva);
                ga.space_group = sg;
                StructureGeometry gb = cubic_p1(a_len, trb, rvb);
                gb.space_group = sg;

                CompackOptions opts;
                opts.n_neighbours = 12;
                opts.allow_inversion = (trial % 2 == 0);
                ClusterSpec spec;
                spec.radius = 18.0;

                ClusterSpec rs = spec; rs.n_neighbours = opts.n_neighbours;
                ClusterSpec ps = spec; ps.n_neighbours = candidate_pool_size(opts.n_neighbours);
                PackingCluster ca = build_cluster(ga, one, 0, rs);
                PackingCluster cb = build_cluster(gb, one, 0, ps);
                if (ca.neighbours.empty() || cb.neighbours.empty()) continue;

                const CompackResult host = compack_compare(ca, cb, one[0], opts);

                std::vector<klasp::compack::Placement> ref, pool;
                for (int q = 0; q < kernel_reference_count(ca, opts); ++q)
                    ref.push_back(to_placement(ca.neighbours[q]));
                for (const auto& m : cb.neighbours) pool.push_back(to_placement(m));
                klasp::compack::BodyView bv = to_view(one[0]);
                const klasp::compack::PairOptions kopts = kernel_pair_options(ca, opts);

                std::vector<double> cost((size_t)ref.size() * pool.size());
                const klasp::compack::PairResult dev = klasp::compack::k_compare_pair(
                    bv, to_placement(ca.central), ref.data(), (int)ref.size(),
                    to_placement(cb.central), pool.data(), (int)pool.size(),
                    kopts, cost.data());

                INFO("body " << which << " sg" << sg << " trial " << trial
                     << " host " << host.n_matched << " dev " << dev.n_matched);
                REQUIRE(dev.valid == host.valid);
                REQUIRE(dev.n_matched == host.n_matched);
                REQUIRE(dev.same_packing == host.same_packing);
                if (host.n_matched > 0) REQUIRE_THAT(dev.rmsd, WithinAbs(host.rmsd, 1e-7));
            }
        }
    }
}

TEST_CASE("The all-pairs sweep reproduces the one-pair path", "[compack][sweep]") {
    // The team split must not change the answer: checked exactly against the
    // one-pair path compiled for the same device. Across backends the refinement
    // can reach a different fixed point, so only the verdict must agree and the
    // matched count may differ by one.
    std::vector<RigidBody> one{benzene_ring()};   // 24 self-ops: the hard case
    std::mt19937 rng(31);
    std::uniform_real_distribution<double> u01(0.0, 1.0), rot(-1.5, 1.5);

    const int n = 7;
    CompackOptions opts;
    opts.n_neighbours = 12;
    opts.allow_inversion = true;
    ClusterSpec rs; rs.radius = 20.0; rs.n_neighbours = opts.n_neighbours;
    ClusterSpec ps; ps.radius = 20.0; ps.n_neighbours = candidate_pool_size(opts.n_neighbours);

    std::vector<StructureGeometry> geom(n);
    std::vector<std::vector<double>> tr(n), rv(n);
    SweepClusters sc;
    sc.n = n;
    sc.n_ref = std::min(candidate_pool_size(opts.n_neighbours), klasp::compack::MAX_NEIGH);
    sc.n_pool = candidate_pool_size(opts.n_neighbours);
    sc.centrals.resize(n);
    sc.ref.resize((size_t)n * sc.n_ref);
    sc.pool.resize((size_t)n * sc.n_pool);
    sc.ref_count.assign(n, 0);
    sc.untied.assign(n, -1);

    std::vector<PackingCluster> refs(n), pools(n);
    for (int i = 0; i < n; ++i) {
        tr[i] = {u01(rng), u01(rng), u01(rng)};
        rv[i] = {rot(rng), rot(rng), rot(rng)};
        geom[i] = cubic_p1(8.0, tr[i].data(), rv[i].data());
        geom[i].space_group = (i % 2) ? 2 : 14;
        refs[i] = build_cluster(geom[i], one, 0, rs);
        pools[i] = build_cluster(geom[i], one, 0, ps);
        REQUIRE(refs[i].scored == opts.n_neighbours);
        REQUIRE((int)pools[i].neighbours.size() >= sc.n_pool);
        sc.centrals[i] = refs[i].central;
        sc.ref_count[i] = std::min<int>(refs[i].neighbours.size(), sc.n_ref);
        sc.untied[i] = refs[i].untied;
        for (int k = 0; k < sc.ref_count[i]; ++k) sc.ref[(size_t)i * sc.n_ref + k] = refs[i].neighbours[k];
        for (int k = 0; k < sc.n_pool; ++k) sc.pool[(size_t)i * sc.n_pool + k] = pools[i].neighbours[k];
        pools[i].neighbours.resize(sc.n_pool);   // the one-pair paths see the sweep's pool
    }

    std::vector<int> nm;
    std::vector<unsigned char> same;
    compack_sweep(sc, one[0], opts, nm, same);
    REQUIRE((int)nm.size() == n * n);

    for (int i = 0; i < n; ++i) {
        REQUIRE(nm[(size_t)i * n + i] == opts.n_neighbours);
        for (int j = i + 1; j < n; ++j) {
            const CompackResult serial = compack_compare(refs[i], pools[j], one[0], opts);
            const int dev_serial = device_serial_n_matched(refs[i], pools[j], one[0], opts);
            INFO("pair " << i << "," << j << "  sweep " << nm[(size_t)i * n + j]
                 << "  device serial " << dev_serial
                 << "  host serial " << serial.n_matched);

            REQUIRE(nm[(size_t)i * n + j] == dev_serial);
            REQUIRE(nm[(size_t)j * n + i] == dev_serial);

            // Against the host: same verdict, count within one.
            REQUIRE((same[(size_t)i * n + j] != 0) == serial.same_packing);
            REQUIRE(std::abs(nm[(size_t)i * n + j] - serial.n_matched) <= 1);
        }
    }
}

TEST_CASE("the device body view never points into host memory", "[compack][sweep]") {
    // A correctly built device BodyView aliases nothing the RigidBody owns:
    // host pointers must not reach the device.
    for (RigidBody body : {benzene_ring(), asymmetric_body()}) {
        REQUIRE_FALSE(body.self_ops.empty());
        REQUIRE_FALSE(body.qgt.empty());

        const DeviceBody device_body(body);
        const compack::BodyView bv = device_body.view();

        CHECK(bv.self_ops != nullptr);
        CHECK(bv.qgt != nullptr);
        CHECK(bv.self_ops != body.self_ops.data());
        CHECK(bv.qgt != body.qgt.data());

        // The scalars and tables must still describe the same body.
        const compack::BodyView host = body_view(body);
        CHECK(bv.m == host.m);
        CHECK(bv.n_ops == host.n_ops);
        CHECK(bv.trace_g == host.trace_g);
        for (int i = 0; i < 9; ++i) CHECK(bv.gyration[i] == host.gyration[i]);

        // Read the device allocations back; bv is only valid inside the kernel.
        auto ops = Kokkos::View<const double*, ExecSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            bv.self_ops, body.self_ops.size());
        auto qgt = Kokkos::View<const double*, ExecSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            bv.qgt, body.qgt.size());
        auto h_ops = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ops);
        auto h_qgt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, qgt);
        for (std::size_t k = 0; k < body.self_ops.size(); ++k)
            CHECK(h_ops(k) == body.self_ops[k]);
        for (std::size_t k = 0; k < body.qgt.size(); ++k)
            CHECK(h_qgt(k) == body.qgt[k]);
    }
}

TEST_CASE("Heavy atoms alone are used only when they fix the orientation",
          "[compack][heavy]") {
    SECTION("ethylene's carbons lie on a line its hydrogens leave") {
        const double p[] = { 0.000, 0.0,  0.667,   0.000, 0.0, -0.667,
                             0.923, 0.0,  1.238,  -0.923, 0.0,  1.238,
                             0.923, 0.0, -1.238,  -0.923, 0.0, -1.238};
        const int z[] = {6, 6, 1, 1, 1, 1};
        CHECK_FALSE(heavy_atoms_suffice(p, z, 6));
    }
    SECTION("CO2 is linear as a whole, so its heavy atoms are all there is") {
        const double p[] = {0.0, 0.0, 0.0,  0.0, 0.0, 1.16,  0.0, 0.0, -1.16};
        const int z[] = {6, 8, 8};
        CHECK(heavy_atoms_suffice(p, z, 3));
    }
    SECTION("water has one heavy atom, which cannot orient anything") {
        const double p[] = {0.0, 0.0, 0.117,  0.0, 0.757, -0.469,  0.0, -0.757, -0.469};
        const int z[] = {8, 1, 1};
        CHECK_FALSE(heavy_atoms_suffice(p, z, 3));
    }
    SECTION("acetic acid's heavy atoms span a plane") {
        const double p[] = { 0.000,  0.000, 0.0,   1.500,  0.000, 0.0,
                             2.100,  1.050, 0.0,   2.150, -1.150, 0.0,
                            -0.360,  1.020, 0.0};
        const int z[] = {6, 6, 8, 8, 1};
        CHECK(heavy_atoms_suffice(p, z, 5));
    }
}

TEST_CASE("PDD row blocks carry the relabellings a molecule's symmetry makes",
          "[compack][pdd]") {
    // Urea, planar and C2v about y: the mirror x -> -x swaps the two NH2 groups,
    // and the molecular plane relabels nothing.
    const double angstrom[8][3] = {{0.0, 0.0, 0.0},    {0.0, 1.25, 0.0},
                                   {1.15, -0.70, 0.0}, {-1.15, -0.70, 0.0},
                                   {2.05, -0.25, 0.0}, {1.15, -1.70, 0.0},
                                   {-2.05, -0.25, 0.0}, {-1.15, -1.70, 0.0}};
    BasisConfig basis{};
    basis.n_sites = 8;
    basis.n_mols = 1;
    basis.elements = {"C", "O", "N", "N", "H", "H", "H", "H"};
    for (const auto& p : angstrom)
        basis.body_positions.push_back({p[0] / BOHR_TO_ANGSTROM, p[1] / BOHR_TO_ANGSTROM,
                                        p[2] / BOHR_TO_ANGSTROM});

    const auto blocks = pdd_symmetry_blocks(basis);
    REQUIRE(blocks.size() == 1u);
    CHECK(blocks[0].first_row == 0);
    REQUIRE(blocks[0].perms.size() == 2u);
    const std::vector<int> identity{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<int> swapped{0, 1, 3, 2, 6, 7, 4, 5};
    CHECK(std::find(blocks[0].perms.begin(), blocks[0].perms.end(), identity)
          != blocks[0].perms.end());
    CHECK(std::find(blocks[0].perms.begin(), blocks[0].perms.end(), swapped)
          != blocks[0].perms.end());
}

TEST_CASE("A shell tied at the edge of the cluster fills the last places, nothing more",
          "[compack][shell]") {
    RigidBody body = asymmetric_body();
    const double eye[9] = {1,0,0, 0,1,0, 0,0,1};
    const PackingCluster pool = lattice_cluster(eye, 6.0, 3);

    // The first 14 of the lattice, then two candidates for the 15th place: one
    // on the lattice, one moved 3 A off it.
    PackingCluster ref;
    ref.central = pool.central;
    for (int i = 0; i < 15; ++i) ref.neighbours.push_back(pool.neighbours[i]);
    PlacedMolecule stray = pool.neighbours[14];
    stray.centroid[0] += 3.0;
    ref.neighbours.push_back(stray);
    ref.untied = 14;
    ref.scored = 15;

    CompackOptions opts;
    opts.n_neighbours = 15;

    SECTION("either tied molecule may take the 15th place") {
        const CompackResult r = compack_compare(ref, pool, body, opts);
        CHECK(r.n_neighbours == 15);
        CHECK(r.n_matched == 15);
        CHECK(r.same_packing);
        std::swap(ref.neighbours[14], ref.neighbours[15]);
        CHECK(compack_compare(ref, pool, body, opts).n_matched == 15);
    }
    SECTION("a matched tied molecule does not stand in for a nearer one") {
        ref.neighbours[0].centroid[0] += 3.0;
        ref.neighbours[15] = pool.neighbours[15];   // both tied candidates now match
        const CompackResult r = compack_compare(ref, pool, body, opts);
        CHECK(r.n_matched == 14);
        CHECK_FALSE(r.same_packing);
    }
}

TEST_CASE("A nearly planar body's operations are orthogonal", "[compack][symmetry]") {
    // Ethyl carbamate's heavy atoms, 0.02 A from planar: the mirror comes from
    // fitting triangles, whose covariance is rank 2.
    const double p[6][3] = {
        {-0.082068, 1.967206, 1.710924},
        {0.552122, 1.151814, 0.631942},
        {-0.064807, -0.772559, -0.610599},
        {-0.997925, -1.702149, -0.869951},
        {-0.417478, 0.098201, 0.316087},
        {1.010156, -0.742514, -1.178403}};
    const int z[6] = {6, 6, 6, 7, 8, 8};
    RigidBody b = make_rigid_body(&p[0][0], z, 6, /*heavy_only=*/true);
    detect_self_symmetry(b);
    REQUIRE(b.n_self_ops() == 2);
    for (int o = 0; o < b.n_self_ops(); ++o) {
        const double* q = &b.self_ops[o * 9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += q[k * 3 + i] * q[k * 3 + j];
                CHECK_THAT(s, WithinAbs(i == j ? 1.0 : 0.0, 1e-9));
            }
    }
}

TEST_CASE("Kabsch aligns collinear points along their axis", "[compack][kabsch]") {
    // Only the axis is determined; it must land on the target's.
    const double from[9] = {-1.16, 0, 0,  0, 0, 0,  1.16, 0, 0};
    const double len = std::sqrt(0.3 * 0.3 + 0.5 * 0.5 + 0.8 * 0.8);
    const double axis[3] = {0.3 / len, -0.5 / len, 0.8 / len};
    double onto[9];
    for (int k = 0; k < 3; ++k)
        for (int d = 0; d < 3; ++d) onto[k * 3 + d] = from[k * 3] * axis[d] + 2.0;
    double r[9], t[3];
    const double rmsd = kabsch(from, onto, 3, r, t, /*allow_improper=*/false);
    CHECK(rmsd < 1e-9);
    for (int k = 0; k < 3; ++k)
        for (int d = 0; d < 3; ++d) {
            double x = t[d];
            for (int c = 0; c < 3; ++c) x += r[d * 3 + c] * from[k * 3 + c];
            CHECK_THAT(x, WithinAbs(onto[k * 3 + d], 1e-9));
        }
}

TEST_CASE("A slightly bent linear molecule keeps its linear symmetry", "[compack][symmetry]") {
    // CO2 bent by half a degree: within tolerance, so it has the operations of
    // the straight molecule.
    const double s = 1.16 * std::sin(0.25 * M_PI / 180.0);
    const double c = 1.16 * std::cos(0.25 * M_PI / 180.0);
    const double p[9] = {0, 0, 0,  c, s, 0,  -c, s, 0};
    const int z[3] = {6, 8, 8};
    RigidBody b = make_rigid_body(p, z, 3, /*heavy_only=*/true);
    detect_self_symmetry(b);
    CHECK(b.n_self_ops() >= 4);
}
