// The Verlet neighbour cache must contain every edge the uncached search finds:
// the cached search must size its image box for cutoff + skin, not cutoff.
// Pure geometry, no model.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>

#include "klasp/crystal_dof.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <tuple>
#include <vector>

using namespace klasp;

namespace {

using Edge = std::tuple<int, int, int, int, int>;   // i, j, shift a/b/c

// Enumerate pairs within `keep` using image ranges sized for `search`, as both
// paths in build_nef_device do.
std::set<Edge> edges(const std::vector<std::array<double, 3>>& pos,
                     const double cell[9], double search, double keep) {
    int rng[3];
    crystdof::image_ranges_rows(cell, search, rng);
    const double keep2 = keep * keep;
    const int n = static_cast<int>(pos.size());
    std::set<Edge> out;
    for (int i = 0; i < n; ++i)
        for (int sa = -rng[0]; sa <= rng[0]; ++sa)
        for (int sb = -rng[1]; sb <= rng[1]; ++sb)
        for (int sc = -rng[2]; sc <= rng[2]; ++sc) {
            const double shx = sa*cell[0] + sb*cell[3] + sc*cell[6];
            const double shy = sa*cell[1] + sb*cell[4] + sc*cell[7];
            const double shz = sa*cell[2] + sb*cell[5] + sc*cell[8];
            for (int j = 0; j < n; ++j) {
                const double vx = pos[j][0] + shx - pos[i][0];
                const double vy = pos[j][1] + shy - pos[i][1];
                const double vz = pos[j][2] + shz - pos[i][2];
                const double d2 = vx*vx + vy*vy + vz*vz;
                if (d2 < 1e-24 || d2 > keep2) continue;
                out.insert({i, j, sa, sb, sc});
            }
        }
    return out;
}

}  // namespace

TEST_CASE("the cached neighbour topology covers the direct search",
          "[pet][neighbors]") {
    std::mt19937 rng(20260811);
    // Small cells relative to cutoff + skin, so the two radii give different
    // image boxes (ceil(radius / interplanar spacing)). The final assertion
    // checks this regime is still reached.
    std::uniform_real_distribution<double> len(4.5, 9.0);    // Angstrom
    std::uniform_real_distribution<double> ang(70.0, 110.0);
    std::uniform_real_distribution<double> frac(-1.5, 1.5);  // unwrapped

    const double cutoff = 5.0;   // PET's, Angstrom
    const double skin = 1.0;     // build_nef_device's

    int checked = 0, would_have_missed = 0;
    for (int trial = 0; trial < 120; ++trial) {
        // Lattice vectors as rows, as in the PET frame.
        const double a = len(rng), b = len(rng), c = len(rng);
        const double al = ang(rng)*M_PI/180, be = ang(rng)*M_PI/180, ga = ang(rng)*M_PI/180;
        const double ca = std::cos(al), cb = std::cos(be), cg = std::cos(ga);
        const double sg = std::sin(ga);
        const double v2 = 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg;
        if (v2 <= 0.05) continue;                      // skip near-degenerate cells
        const double cz = c * std::sqrt(v2) / sg;
        const double cell[9] = {a, 0, 0,
                                b*cg, b*sg, 0,
                                c*cb, c*(ca - cb*cg)/sg, cz};

        std::vector<std::array<double, 3>> pos(12);
        for (auto& p : pos) {
            const double f0 = frac(rng), f1 = frac(rng), f2 = frac(rng);
            for (int k = 0; k < 3; ++k)
                p[k] = f0*cell[k] + f1*cell[3+k] + f2*cell[6+k];
        }

        // The uncached result, and what each cache build holds once filtered to
        // the cutoff.
        const auto direct = edges(pos, cell, cutoff, cutoff);
        const auto cached_fixed = edges(pos, cell, cutoff + skin, cutoff);

        for (const auto& e : direct) {
            INFO("trial " << trial);
            REQUIRE(cached_fixed.count(e) == 1);
        }
        ++checked;

        // At cutoff + skin the cache holds strictly more.
        const auto held_fixed = edges(pos, cell, cutoff + skin, cutoff + skin);
        const auto held_old = edges(pos, cell, cutoff, cutoff + skin);
        if (held_old.size() < held_fixed.size()) ++would_have_missed;
        for (const auto& e : held_old) REQUIRE(held_fixed.count(e) == 1);
    }

    std::printf("  %d cells checked; %d would have cached an incomplete "
                "cutoff+skin shell with the old range\n", checked, would_have_missed);
    REQUIRE(checked > 100);
    // If this is ever 0 the cells no longer exercise the image-box difference.
    REQUIRE(would_have_missed > 0);
}

// An isolated molecule must not be wrapped into the padded box built around it
// (pet_isolated_reference). Quantifies the hazard: wrapping a molecule that
// straddles its box breaks internal distances.
TEST_CASE("an isolated molecule survives the neighbour build", "[pet][neighbors]") {
    // A bent molecule centred on the origin, so half its coordinates are
    // negative, as in the body frame.
    const std::vector<std::array<double, 3>> mol = {
        {-1.20, 0.00, 0.00}, {0.00, 0.00, 0.00}, {0.65, 1.15, 0.00},
        {0.65, -1.15, 0.35}, {-1.85, 0.95, 0.10}};
    // The padded box pet_isolated_reference builds: extent + 20 Bohr.
    const double L = 14.5;

    auto wrap = [&](std::array<double, 3> p) {
        std::array<double, 3> f{p[0]/L, p[1]/L, p[2]/L};
        for (int k = 0; k < 3; ++k) f[k] -= std::floor(f[k]);
        return std::array<double, 3>{f[0]*L, f[1]*L, f[2]*L};
    };
    auto dist = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return std::sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])
                       + (a[2]-b[2])*(a[2]-b[2]));
    };

    // Wrapping it tears it apart.
    double worst_wrapped = 0;
    for (size_t i = 0; i < mol.size(); ++i)
        for (size_t j = i + 1; j < mol.size(); ++j)
            worst_wrapped = std::max(worst_wrapped,
                std::abs(dist(wrap(mol[i]), wrap(mol[j])) - dist(mol[i], mol[j])));
    std::printf("  wrapping an isolated molecule distorts a distance by %.2f A\n",
                worst_wrapped);
    REQUIRE(worst_wrapped > 5.0);

    // Does not exercise the kernel (needs device staging and a model); the
    // kernel is guarded by the `sper` branch in build_nef_device.
}
