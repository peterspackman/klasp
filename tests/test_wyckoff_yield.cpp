#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include "klasp/wyckoff.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace klasp;

// Does placing the molecule on a special position make these space groups
// generatable? Same molecule, collision rule and volume per molecule; only the
// source of the copies (whole group vs one site's cosets) differs.

namespace {

constexpr int N_SITES = 4;

std::vector<double> ammonia_body() {          // Angstrom, 3-fold along z
    const double rh = 0.9377, zh = -0.3263, zn = 0.0704;
    std::vector<double> p = {0, 0, zn};
    for (int k = 0; k < 3; ++k) {
        const double a = 2.0 * M_PI * k / 3.0;
        p.push_back(rh * std::cos(a));
        p.push_back(rh * std::sin(a));
        p.push_back(zh);
    }
    return p;
}

std::vector<double> ammonia_ops() {
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

// Closest approach between atoms of different molecules, over the 27 nearest
// cells.
double min_intermolecular(const std::vector<double>& cart, int n_mol,
                          const double cell[3][3]) {
    double best = 1e9;
    for (int i = 0; i < n_mol; ++i)
    for (int j = 0; j < n_mol; ++j)
    for (int sa = -1; sa <= 1; ++sa)
    for (int sb = -1; sb <= 1; ++sb)
    for (int sc = -1; sc <= 1; ++sc) {
        if (i == j && sa == 0 && sb == 0 && sc == 0) continue;
        double sh[3];
        for (int d = 0; d < 3; ++d)
            sh[d] = cell[d][0]*sa + cell[d][1]*sb + cell[d][2]*sc;
        for (int ai = 0; ai < N_SITES; ++ai)
        for (int aj = 0; aj < N_SITES; ++aj) {
            double d2 = 0;
            for (int d = 0; d < 3; ++d) {
                const double e = cart[(i*N_SITES+ai)*3+d]
                               - (cart[(j*N_SITES+aj)*3+d] + sh[d]);
                d2 += e * e;
            }
            best = std::min(best, d2);
        }
    }
    return std::sqrt(best);
}

void frac_to_cart(const double cell[3][3], const double f[3], double out[3]) {
    for (int d = 0; d < 3; ++d)
        out[d] = cell[d][0]*f[0] + cell[d][1]*f[1] + cell[d][2]*f[2];
}

// Build the copies of one molecule under a list of operations.
std::vector<double> build(const SpaceGroupInfo& sg, const std::vector<int>& ops,
                          const double cell[3][3], const double inv[3][3],
                          const double site_frac[3], const double R[9],
                          const std::vector<double>& body) {
    std::vector<double> cart;
    for (int op : ops) {
        // Fractional rotation acts on the molecule's Cartesian frame as A S A^-1.
        double AS[9], Q[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0;
                for (int k = 0; k < 3; ++k) acc += sg.symops[op].rot[i][k] * inv[k][j];
                AS[i*3+j] = acc;
            }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0;
                for (int k = 0; k < 3; ++k) acc += cell[i][k] * AS[k*3+j];
                Q[i*3+j] = acc;
            }
        double fc[3];
        for (int d = 0; d < 3; ++d) {
            fc[d] = sg.symops[op].trans[d];
            for (int k = 0; k < 3; ++k) fc[d] += sg.symops[op].rot[d][k] * site_frac[k];
        }
        double cen[3];
        frac_to_cart(cell, fc, cen);
        for (int s = 0; s < N_SITES; ++s) {
            double oriented[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k) oriented[i] += R[i*3+k] * body[s*3+k];
            double placed[3];
            for (int i = 0; i < 3; ++i) {
                placed[i] = 0;
                for (int k = 0; k < 3; ++k) placed[i] += Q[i*3+k] * oriented[k];
            }
            for (int d = 0; d < 3; ++d) cart.push_back(cen[d] + placed[d]);
        }
    }
    return cart;
}

}  // namespace

TEST_CASE("a special position makes the mirror-heavy groups generatable",
          "[wyckoff][yield]") {
    const auto body = ammonia_body();
    const auto mol_ops = ammonia_ops();
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double CLASH = 2.0;
    const int N = 1200;
    double vol_per_mol = 120.0;

    // Solid ammonia is about 35 A^3 per molecule; looser cells are vacuum slabs.
    const std::vector<double> vols = {35.0, 60.0, 120.0};

    std::string report = "ammonia, " + std::to_string(N)
        + " draws per cell, clash under 2 A, % clash-free\n";
    report += "                       ";
    for (double v : vols) { char b[32]; std::snprintf(b, sizeof b, "%7.0f", v); report += b; }
    report += "   A^3/mol\n";

    std::map<int, double> dense_general, dense_special;   // rate at 35 A^3

    for (int SG : {156, 183, 191, 194}) {
        const SpaceGroupInfo& sg = get_space_group(SG);
        double ccell[3][3];
        canonical_cell(sg.lattice, ccell);

        auto trial = [&](int n_mol, const std::vector<int>& ops,
                         const WyckoffSite* site, const std::vector<double>& oris) {
            const double vol = vol_per_mol * n_mol;
            const double ratio = 0.45 + 2.6 * u01(rng);       // wide shape range
            const double a = std::cbrt(vol / (0.8660254 * ratio));
            const double c = ratio * a;
            RealType pr[6] = {(RealType)a, (RealType)a, (RealType)c, 90, 90, 120};
            RealType m[3][3];
            cell_params_to_matrix(pr, m);
            double cell[3][3], inv[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) cell[i][j] = m[i][j];
            if (!mat3_inv(cell, inv)) return false;

            double sitef[3], R[9];
            if (site) {
                for (int d = 0; d < 3; ++d) sitef[d] = site->rep[d];
                for (int f = 0; f < site->n_free; ++f) {
                    const double t = u01(rng);
                    for (int d = 0; d < 3; ++d) sitef[d] += t * site->free_basis[f][d];
                }
                const int pick = static_cast<int>(u01(rng) * (oris.size() / 9));
                for (int k = 0; k < 9; ++k) R[k] = oris[pick * 9 + k];
            } else {
                for (int d = 0; d < 3; ++d) sitef[d] = u01(rng);
                const double q0=u01(rng)*2-1,q1=u01(rng)*2-1,q2=u01(rng)*2-1,q3=u01(rng)*2-1;
                const double n = std::sqrt(q0*q0+q1*q1+q2*q2+q3*q3);
                const double x=q1/n,y=q2/n,z=q3/n,ww=q0/n;
                R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-z*ww); R[2]=2*(x*z+y*ww);
                R[3]=2*(x*y+z*ww);  R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-x*ww);
                R[6]=2*(x*z-y*ww);  R[7]=2*(y*z+x*ww);  R[8]=1-2*(x*x+y*y);
            }
            const auto cart = build(sg, ops, cell, inv, sitef, R, body);
            return min_intermolecular(cart, n_mol, cell) > CLASH;
        };

        std::vector<int> all_ops(sg.n_symops);
        for (int i = 0; i < sg.n_symops; ++i) all_ops[i] = i;

        char hdr[64];
        std::snprintf(hdr, sizeof hdr, "  SG %-3d general Z=%-3d ", SG, sg.n_symops);
        report += hdr;
        for (double v : vols) {
            vol_per_mol = v;
            int g = 0;
            for (int k = 0; k < N; ++k) if (trial(sg.n_symops, all_ops, nullptr, {})) ++g;
            char b[32]; std::snprintf(b, sizeof b, "%6.2f%%", 100.0*g/N); report += b;
            if (v == vols.front()) dense_general[SG] = double(g) / N;
        }
        report += "\n";

        for (const auto& site : special_positions(sg)) {
            const auto w = site_ops_cartesian(sg, site, ccell);
            const auto oris = orientations_on_site(w, mol_ops);
            if (oris.empty()) continue;
            std::snprintf(hdr, sizeof hdr, "         site  Z=%-3d ", site.multiplicity);
            report += hdr;
            for (double v : vols) {
                vol_per_mol = v;
                int sp = 0;
                for (int k = 0; k < N; ++k)
                    if (trial(site.multiplicity, site.coset_reps, &site, oris)) ++sp;
                char b[32]; std::snprintf(b, sizeof b, "%6.2f%%", 100.0*sp/N); report += b;
                if (v == vols.front()) dense_special[SG] = double(sp) / N;
            }
            report += "\n";
            break;                                 // one site per group is enough
        }
    }
    WARN(report);
    // P3m1: a general position cannot place ammonia at all at crystal density;
    // its full-group site places it easily.
    INFO("P3m1 general " << dense_general[156] << " special " << dense_special[156]);
    REQUIRE(dense_general[156] < 0.01);
    REQUIRE(dense_special[156] > 0.10);
    // P6_3/mmc: a high-symmetry group that stalls on general positions.
    INFO("P6_3/mmc general " << dense_general[194] << " special " << dense_special[194]);
    REQUIRE(dense_special[194] > 20 * dense_general[194]);
}
