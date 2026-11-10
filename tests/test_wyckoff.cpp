#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include "klasp/wyckoff.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/cell.hpp"

#include <algorithm>

using namespace klasp;

namespace {

// The multiplicities of a space group's special positions, deduped and sorted
// (representative coordinates depend on the origin choice).
std::vector<int> multiplicities(int sg_num) {
    const auto& sg = get_space_group(sg_num);
    auto sites = special_positions(sg);
    std::vector<int> m;
    for (const auto& s : sites) m.push_back(s.multiplicity);
    std::sort(m.begin(), m.end());
    m.erase(std::unique(m.begin(), m.end()), m.end());
    return m;
}

}  // namespace

TEST_CASE("the site symmetry of a general position is the identity alone",
          "[wyckoff]") {
    const auto& sg = get_space_group(194);
    // A point on no symmetry element: denominators the group cannot fix.
    const double x[3] = {0.137, 0.291, 0.077};
    REQUIRE(site_symmetry_ops(sg, x).size() == 1);
    REQUIRE(static_cast<int>(coset_representatives(sg, x).size()) == sg.n_symops);
}

TEST_CASE("multiplicity times site order is the group order", "[wyckoff]") {
    for (int sg_num : {2, 14, 19, 62, 148, 156, 176, 194, 221, 225}) {
        const auto& sg = get_space_group(sg_num);
        for (const auto& s : special_positions(sg)) {
            INFO("SG " << sg_num << " site order " << s.site_order
                 << " multiplicity " << s.multiplicity);
            REQUIRE(s.multiplicity * s.site_order == sg.n_symops);
            REQUIRE(static_cast<int>(s.coset_reps.size()) == s.multiplicity);
            REQUIRE(static_cast<int>(s.site_ops.size()) == s.site_order);
        }
    }
}

TEST_CASE("special positions match the International Tables", "[wyckoff]") {
    // P-1: the eight inversion centres, all multiplicity 1.
    REQUIRE(multiplicities(2) == std::vector<int>{1});

    // P2_1/c: four inversion centres at multiplicity 2; screw and glide have no
    // fixed points.
    REQUIRE(multiplicities(14) == std::vector<int>{2});

    // P2_1 2_1 2_1: screws only, so no special position at all.
    REQUIRE(multiplicities(19).empty());

    // Pbca: inversion centres only.
    REQUIRE(multiplicities(61) == std::vector<int>{4});
}

TEST_CASE("P3m1 has a site that holds a C3v molecule alone in the cell",
          "[wyckoff]") {
    const auto& sg = get_space_group(156);
    auto sites = special_positions(sg);
    REQUIRE(sg.n_symops == 6);
    // The 3m site: the whole group fixes it, so one molecule fills the cell.
    auto it = std::find_if(sites.begin(), sites.end(),
                           [](const WyckoffSite& s) { return s.site_order == 6; });
    REQUIRE(it != sites.end());
    REQUIRE(it->multiplicity == 1);
    // A line along c: the molecule may slide along the 3-fold.
    REQUIRE(it->n_free == 1);
}

TEST_CASE("P6_3/mmc has a 3m site of multiplicity 4", "[wyckoff]") {
    const auto& sg = get_space_group(194);
    REQUIRE(sg.n_symops == 24);
    auto sites = special_positions(sg);
    auto it = std::find_if(sites.begin(), sites.end(),
                           [](const WyckoffSite& s) { return s.multiplicity == 4; });
    REQUIRE(it != sites.end());
    REQUIRE(it->site_order == 6);
    // Ammonia is C3v, order 6: it fits here, and Z drops from 24 to 4.
}

// --- matching a molecule to a site ------------------------------------------

namespace {

// The symmetry operations of an idealised C3v molecule with its 3-fold along z
// and one mirror normal along x: E, 2C3, 3 sigma_v.
std::vector<double> c3v_ops() {
    std::vector<double> ops;
    auto push = [&](double m[9]) { ops.insert(ops.end(), m, m + 9); };
    double I[9] = {1,0,0, 0,1,0, 0,0,1};
    push(I);
    for (int k = 1; k <= 2; ++k) {
        const double a = 2.0 * M_PI * k / 3.0, c = std::cos(a), s = std::sin(a);
        double r[9] = {c,-s,0, s,c,0, 0,0,1};
        push(r);
    }
    for (int k = 0; k < 3; ++k) {
        // Mirror whose normal is turned by k*60 degrees about z.
        const double a = M_PI * k / 3.0;
        const double n[3] = {std::cos(a), std::sin(a), 0};
        double m[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i*3+j] = (i == j ? 1.0 : 0.0) - 2.0 * n[i] * n[j];
        push(m);
    }
    return ops;
}

// C1: no symmetry beyond the identity.
std::vector<double> c1_ops() { return {1,0,0, 0,1,0, 0,0,1}; }

}  // namespace

TEST_CASE("a site's Cartesian operations do not depend on the free cell ratios",
          "[wyckoff]") {
    // canonical_cell relies on this: a hexagonal operation's Cartesian form does
    // not depend on c/a.
    const auto& sg = get_space_group(194);
    auto sites = special_positions(sg);
    REQUIRE(!sites.empty());

    RealType p1[6] = {1, 1, 1.37, 90, 90, 120};
    RealType p2[6] = {1, 1, 4.10, 90, 90, 120};
    RealType m1[3][3], m2[3][3];
    cell_params_to_matrix(p1, m1);
    cell_params_to_matrix(p2, m2);
    double c1[3][3], c2[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) { c1[i][j] = m1[i][j]; c2[i][j] = m2[i][j]; }

    for (const auto& s : sites) {
        auto a = site_ops_cartesian(sg, s, c1);
        auto b = site_ops_cartesian(sg, s, c2);
        REQUIRE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            INFO("site order " << s.site_order << " element " << i);
            REQUIRE(std::abs(a[i] - b[i]) < 1e-9);
        }
    }
}

TEST_CASE("a site's Cartesian operations are orthogonal", "[wyckoff]") {
    for (int sg_num : {14, 62, 156, 176, 194, 221}) {
        const auto& sg = get_space_group(sg_num);
        double cell[3][3];
        canonical_cell(sg.lattice, cell);
        for (const auto& s : special_positions(sg)) {
            auto ops = site_ops_cartesian(sg, s, cell);
            for (size_t o = 0; o < ops.size(); o += 9) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        double dot = 0;
                        for (int k = 0; k < 3; ++k) dot += ops[o+k*3+i] * ops[o+k*3+j];
                        INFO("SG " << sg_num << " site order " << s.site_order);
                        REQUIRE(std::abs(dot - (i == j ? 1.0 : 0.0)) < 1e-9);
                    }
            }
        }
    }
}

TEST_CASE("a C1 molecule takes no special position", "[wyckoff]") {
    const auto& sg = get_space_group(194);
    double cell[3][3];
    canonical_cell(sg.lattice, cell);
    for (const auto& s : special_positions(sg)) {
        auto w = site_ops_cartesian(sg, s, cell);
        int dof = -1;
        auto oris = orientations_on_site(w, c1_ops(), &dof);
        INFO("site order " << s.site_order);
        REQUIRE(oris.empty());
    }
}

TEST_CASE("a C3v molecule fits P6_3/mmc's order-6 site", "[wyckoff]") {
    const auto& sg = get_space_group(194);
    double cell[3][3];
    canonical_cell(sg.lattice, cell);
    auto sites = special_positions(sg);
    auto it = std::find_if(sites.begin(), sites.end(),
                           [](const WyckoffSite& s) { return s.multiplicity == 4; });
    REQUIRE(it != sites.end());
    REQUIRE(it->site_order == 6);

    auto w = site_ops_cartesian(sg, *it, cell);
    int dof = -1;
    auto oris = orientations_on_site(w, c3v_ops(), &dof);
    INFO("orientations found: " << oris.size() / 9);
    REQUIRE(!oris.empty());

    // Every orientation returned must make the molecule invariant: R^-1 W R lies
    // in the molecule's own operations.
    const auto m = c3v_ops();
    for (size_t r = 0; r < oris.size(); r += 9) {
        for (size_t o = 0; o < w.size(); o += 9) {
            double Rt[9], tmp[9], conj[9];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) Rt[i*3+j] = oris[r + j*3+i];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double acc = 0;
                    for (int k = 0; k < 3; ++k) acc += Rt[i*3+k] * w[o+k*3+j];
                    tmp[i*3+j] = acc;
                }
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double acc = 0;
                    for (int k = 0; k < 3; ++k) acc += tmp[i*3+k] * oris[r+k*3+j];
                    conj[i*3+j] = acc;
                }
            bool found = false;
            for (size_t q = 0; q < m.size() && !found; q += 9) {
                bool same = true;
                for (int i = 0; i < 9 && same; ++i)
                    if (std::abs(m[q+i] - conj[i]) > 1e-4) same = false;
                found = same;
            }
            REQUIRE(found);
        }
    }
}

TEST_CASE("a C3v molecule does not fit a site needing more symmetry than it has",
          "[wyckoff]") {
    // P6_3/mmc's order-12 site asks for twelve operations; C3v has six.
    const auto& sg = get_space_group(194);
    double cell[3][3];
    canonical_cell(sg.lattice, cell);
    auto sites = special_positions(sg);
    auto it = std::find_if(sites.begin(), sites.end(),
                           [](const WyckoffSite& s) { return s.site_order == 12; });
    REQUIRE(it != sites.end());
    auto w = site_ops_cartesian(sg, *it, cell);
    REQUIRE(orientations_on_site(w, c3v_ops()).empty());
}
