#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>

#include "klasp/fft_generator.hpp"
#include "klasp/sampling_prior.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/constants.hpp"
#include "klasp/volume_budget.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <set>
#include <vector>

using namespace klasp;

// `--collision fft_pack`: the overlap map and the structures it emits.

namespace {

// Volume shared by two spheres whose centres are `d` apart.
double lens_volume(double d, double r1, double r2) {
    if (d >= r1 + r2) return 0.0;
    if (d <= std::fabs(r1 - r2)) {
        const double r = std::min(r1, r2);
        return (4.0 / 3.0) * M_PI * r * r * r;
    }
    return M_PI * (r1 + r2 - d) * (r1 + r2 - d)
         * (d*d + 2*d*r2 - 3*r2*r2 + 2*d*r1 + 6*r1*r2 - 3*r1*r1) / (12.0 * d);
}

void upper_inverse(const double cell[3][3], double ci[3][3]) {
    const double det = cell[0][0] * cell[1][1] * cell[2][2];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) ci[i][j] = 0;
    ci[0][0] = cell[1][1]*cell[2][2] / det;
    ci[0][1] = -cell[0][1]*cell[2][2] / det;
    ci[0][2] = (cell[0][1]*cell[1][2] - cell[0][2]*cell[1][1]) / det;
    ci[1][1] = cell[0][0]*cell[2][2] / det;
    ci[1][2] = -cell[0][0]*cell[1][2] / det;
    ci[2][2] = cell[0][0]*cell[1][1] / det;
}

// The FFT map's quantity summed pair by pair in real space over +/-2 lattice images.
double real_space_overlap(const SpaceGroupInfo& sg, int id_op,
                          const double cell[3][3], const double R[3][3],
                          const double* body, const double* rad, int n,
                          const double x[3]) {
    double ci[3][3];
    upper_inverse(cell, ci);
    double total = 0.0;
    for (int s = 0; s < sg.n_symops; ++s) {
        if (s == id_op) continue;
        double tmp[3][3] = {}, Rcs[3][3] = {}, Rc[3][3] = {};
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k)
            tmp[i][j] += sg.symops[s].rot[i][k] * ci[k][j];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k)
            Rcs[i][j] += cell[i][k] * tmp[k][j];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k)
            Rc[i][j] += Rcs[i][k] * R[k][j];

        double fs[3];
        for (int d = 0; d < 3; ++d) {
            fs[d] = sg.symops[s].trans[d];
            for (int k = 0; k < 3; ++k) fs[d] += sg.symops[s].rot[d][k] * x[k];
        }
        for (int nx=-2;nx<=2;++nx) for (int ny=-2;ny<=2;++ny) for (int nz=-2;nz<=2;++nz) {
            const double fsn[3] = {fs[0]+nx, fs[1]+ny, fs[2]+nz};
            double comb[3] = {0,0,0}, com0[3] = {0,0,0};
            for (int d=0;d<3;++d) for (int k=0;k<3;++k) {
                comb[d] += cell[d][k] * fsn[k];
                com0[d] += cell[d][k] * x[k];
            }
            for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                double pi[3], pj[3];
                for (int d = 0; d < 3; ++d) {
                    pi[d] = com0[d] + R[d][0]*body[i*3] + R[d][1]*body[i*3+1] + R[d][2]*body[i*3+2];
                    pj[d] = comb[d] + Rc[d][0]*body[j*3] + Rc[d][1]*body[j*3+1] + Rc[d][2]*body[j*3+2];
                }
                double d2 = 0;
                for (int d = 0; d < 3; ++d) { const double e = pi[d] - pj[d]; d2 += e*e; }
                total += lens_volume(std::sqrt(d2), rad[i], rad[j]);
            }
        }
    }
    return total;
}

struct Mol {
    static constexpr int n = 3;
    double body[9] = {0.0, 0.0, 0.0,  1.1, 0.2, 0.0,  -0.6, 0.9, 0.3};
    double cov[3]  = {0.76, 0.31, 0.31};
    double vdw[3]  = {1.70, 1.20, 1.20};
    View2D<RealType> d_body{"body", n, 3};
    View1D<RealType> d_cov{"cov", n};
    View1D<RealType> d_vdw{"vdw", n};
    Mol() {
        auto hb = Kokkos::create_mirror_view(d_body);
        auto hc = Kokkos::create_mirror_view(d_cov);
        auto hv = Kokkos::create_mirror_view(d_vdw);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < 3; ++d) hb(i, d) = static_cast<RealType>(body[i*3+d]);
            hc(i) = static_cast<RealType>(cov[i]);
            hv(i) = static_cast<RealType>(vdw[i]);
        }
        Kokkos::deep_copy(d_body, hb);
        Kokkos::deep_copy(d_cov, hc);
        Kokkos::deep_copy(d_vdw, hv);
    }
};

}  // namespace

TEST_CASE("the overlap map converges to a real-space sphere overlap",
          "[fft_pack][map]") {
    // A correct map is a truncated convergent series, so its error against the
    // real-space sum falls as 1/n with grid size; a wrong map stays at a fixed error.
    const auto& sg = get_space_group(14);
    const int id_op = find_identity_symop(sg);
    Mol mol;

    // Monoclinic, not axis-aligned in the ac plane.
    const double cell[3][3] = {{7.0, 0.0, -1.2}, {0.0, 6.0, 0.0}, {0.0, 0.0, 8.0}};
    const double R[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    // Same radii the map is given: covalent plus the buffer.
    const double buffer = 0.74;
    const double rad[3] = {mol.cov[0] + buffer, mol.cov[1] + buffer, mol.cov[2] + buffer};

    // Fixed fractional points across every grid.
    std::vector<std::array<int,3>> pts;
    for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) for (int c = 0; c < 8; ++c)
        pts.push_back({a, b, c});

    std::vector<double> ref(pts.size());
    double ref_max = 0;
    for (std::size_t q = 0; q < pts.size(); ++q) {
        const double x[3] = {pts[q][0]/8.0, pts[q][1]/8.0, pts[q][2]/8.0};
        ref[q] = real_space_overlap(sg, id_op, cell, R, mol.body, rad, Mol::n, x);
        ref_max = std::max(ref_max, ref[q]);
    }
    REQUIRE(ref_max > 1.0);   // the case has to have real overlap in it

    auto worst_error = [&](int ng) {
        std::vector<double> omap;
        fft_overlap_map(sg, id_op, cell, R, mol.d_body, mol.d_cov, Mol::n,
                        buffer, ng, ng, ng, omap);
        double worst = 0;
        for (std::size_t q = 0; q < pts.size(); ++q) {
            const int ix = pts[q][0]*ng/8, iy = pts[q][1]*ng/8, iz = pts[q][2]*ng/8;
            const double f = omap[(static_cast<std::size_t>(ix)*ng + iy)*ng + iz];
            worst = std::max(worst, std::fabs(f - ref[q]));
        }
        return worst;
    };

    const double e32 = worst_error(32);
    const double e64 = worst_error(64);

    INFO("max |FFT - real space| : 32^3 " << e32 << ", 64^3 " << e64
         << ", map range " << ref_max);
    CHECK(e64 < 0.02 * ref_max);
    // Halving the spacing must roughly halve the error.
    CHECK(e64 < 0.65 * e32);
}

TEST_CASE("one trial's structures get distinct identities", "[fft_pack][seeds]") {
    // The database key is (seed, space_group, stage) with INSERT OR REPLACE, so each
    // trial owns a block of k distinct identities.
    SECTION("the mapping is a bijection onto blocks") {
        for (int k : {1, 4, 8}) {
            std::set<int> seen;
            for (int trial = 0; trial < 50; ++trial)
                for (int slot = 0; slot < k; ++slot) {
                    const int id = fft_identity(trial, slot, k);
                    REQUIRE(seen.insert(id).second);          // never collides
                    REQUIRE(fft_trial_of(id, k) == trial);    // and says where it came from
                }
        }
    }

    SECTION("a generated batch has no repeats") {
        Mol mol;
        FFTGeneratorConfig cfg;
        cfg.tvp = 1.2;
        constexpr int cap = 256;
        View2D<RealType> cp("cp", cap, 6);
        View3D<RealType> tr("tr", cap, 1, 3), rv("rv", cap, 1, 3);
        View1D<int> sd("sd", cap);

        const int n = fft_generate_batch(14, 0, 24, cfg, mol.d_body, mol.d_cov,
                                         mol.d_vdw, Mol::n, cp, tr, rv, sd, cap);
        REQUIRE(n > 0);
        auto h_sd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sd);
        std::set<int> ids;
        for (int i = 0; i < n; ++i) REQUIRE(ids.insert(h_sd(i)).second);
    }
}

TEST_CASE("the generator yields structures at the shipped default",
          "[fft_pack][generate]") {
    // Sized against the vdW bounding box like every other generator, the default
    // --tvp must yield structures.
    Mol mol;
    FFTGeneratorConfig cfg;      // shipped defaults, tvp included
    constexpr int cap = 512;
    View2D<RealType> cp("cp", cap, 6);
    View3D<RealType> tr("tr", cap, 1, 3), rv("rv", cap, 1, 3);
    View1D<int> sd("sd", cap);

    for (int sg_num : {2, 14, 19}) {
        const int n = fft_generate_batch(sg_num, 0, 32, cfg, mol.d_body, mol.d_cov,
                                         mol.d_vdw, Mol::n, cp, tr, rv, sd, cap);
        INFO("space group " << sg_num);
        CHECK(n > 0);
    }
}

TEST_CASE("every emitted structure is clash-free and inside the prior",
          "[fft_pack][generate]") {
    // Emitted structures must be clash-free when measured directly, and their cells
    // within the sampling prior.
    Mol mol;
    FFTGeneratorConfig cfg;
    constexpr int cap = 512;
    View2D<RealType> cp("cp", cap, 6);
    View3D<RealType> tr("tr", cap, 1, 3), rv("rv", cap, 1, 3);
    View1D<int> sd("sd", cap);

    const int sg_num = 14;
    const auto& sg = get_space_group(sg_num);
    const int n = fft_generate_batch(sg_num, 0, 32, cfg, mol.d_body, mol.d_cov,
                                     mol.d_vdw, Mol::n, cp, tr, rv, sd, cap);
    REQUIRE(n > 0);

    auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cp);
    auto h_tr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tr);
    auto h_rv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rv);

    const int mol_sites[1] = {Mol::n}, mol_offsets[1] = {0};
    const VolumeBudget budget = volume_budget(mol.body, mol.vdw, mol_sites, mol_offsets,
                                              1, sg.n_symops, sg.lattice, cfg.tvp);
    const SamplingPrior prior = make_sampling_prior(sg_num, 1, budget.max_volume);

    for (int i = 0; i < n; ++i) {
        INFO("structure " << i);
        const double a = h_cp(i,0) * BOHR_TO_ANGSTROM;
        const double b = h_cp(i,1) * BOHR_TO_ANGSTROM;
        const double c = h_cp(i,2) * BOHR_TO_ANGSTROM;
        const double al = h_cp(i,3) * M_PI / 180.0;
        const double be = h_cp(i,4) * M_PI / 180.0;
        const double ga = h_cp(i,5) * M_PI / 180.0;
        CHECK(prior_accepts_cell(prior, a, b, c, al, be, ga));

        // The molecule must clear its symmetry images and its own lattice translates.
        double cell[3][3];
        const double ca = std::cos(al), cb = std::cos(be), cg = std::cos(ga);
        const double sg_val = std::sin(ga);
        const double sqrt_x = std::sqrt(std::max(0.0,
            1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg));
        cell[0][0] = a; cell[0][1] = b*cg;     cell[0][2] = c*cb;
        cell[1][0] = 0; cell[1][1] = b*sg_val; cell[1][2] = c*(ca - cb*cg)/sg_val;
        cell[2][0] = 0; cell[2][1] = 0;        cell[2][2] = a*b*c*sqrt_x/(a*b*sg_val);

        double R[3][3];
        {
            const RealType v[3] = {static_cast<RealType>(h_rv(i,0,0)),
                                   static_cast<RealType>(h_rv(i,0,1)),
                                   static_cast<RealType>(h_rv(i,0,2))};
            RealType M[3][3];
            angle_axis_to_matrix(v, M);   // rot_vectors are angle-axis, not Shoemake [0,1)^3
            for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q)
                R[p][q] = static_cast<double>(M[p][q]);
        }
        const double x[3] = {h_tr(i,0,0), h_tr(i,0,1), h_tr(i,0,2)};
        const double rad[3] = {mol.cov[0] + 0.25, mol.cov[1] + 0.25, mol.cov[2] + 0.25};

        // Zero overlap, measured in real space.
        const double ov = real_space_overlap(sg, find_identity_symop(sg), cell, R,
                                             mol.body, rad, Mol::n, x);
        CHECK(ov < 1e-9);
    }
}
