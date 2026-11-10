#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kokkos_listener.hpp"
#include "klasp/csp.hpp"
#include "klasp/generator.hpp"
#include "klasp/landscape_generator.hpp"
#include "klasp/sampling_prior.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/rotation.hpp"
#include "klasp/generate_kernel.hpp"
#include "klasp/constants.hpp"

using namespace klasp;

namespace {

// A small rigid molecule (Angstrom): a bent 5-atom cluster spanning ~3 A.
struct Mol {
    std::vector<RealType> pos;  // interleaved x,y,z
    std::vector<RealType> cov, vdw;
    int n() const { return (int)cov.size(); }
};

Mol makeMol() {
    Mol m;
    const RealType p[5][3] = {
        {0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}, {0.75, 1.30, 0.0},
        {2.25, 1.30, 0.0}, {1.5, 2.60, 0.3}};
    for (auto& a : p) { m.pos.insert(m.pos.end(), {a[0], a[1], a[2]}); }
    m.cov.assign(5, 0.77);
    m.vdw.assign(5, 1.70);
    return m;
}

// A thin rod (~7 A along x, centred on the origin): tiny input-frame box in
// y/z, but a generic rotation needs a much larger cell.
Mol makeRodMol() {
    Mol m;
    const int N = 6;
    for (int i = 0; i < N; ++i)
        m.pos.insert(m.pos.end(), {RealType(1.4 * i - 3.5), RealType(0), RealType(0)});
    m.cov.assign(N, 0.77);
    m.vdw.assign(N, 1.70);
    return m;
}

// Reconstruct the supercell from one generated (cell, trans, rot) and count
// atom clashes using the SAME cov-radius + buffer criterion as the generator.
int countClashes(const Mol& mol, const RealType cp[6],
                 const RealType trans[3], const RealType aa[3],
                 int sg_num, RealType buffer) {
    const int ns = mol.n();
    const auto& sg = get_space_group(sg_num);
    const int nop = sg.n_symops;

    const RealType a = cp[0] * BOHR_TO_ANGSTROM;
    const RealType b = cp[1] * BOHR_TO_ANGSTROM;
    const RealType c = cp[2] * BOHR_TO_ANGSTROM;
    const RealType al = cp[3] * M_PI / 180.0;
    const RealType be = cp[4] * M_PI / 180.0;
    const RealType ga = cp[5] * M_PI / 180.0;
    const RealType ca = std::cos(al), cb = std::cos(be), cg = std::cos(ga);
    const RealType sgam = std::sin(ga);
    const RealType x = 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg;
    const RealType sx = std::sqrt(x);

    RealType cell[3][3], cell_inv[3][3];
    gpu_build_cell(a, b, c, ca, cb, cg, sgam, sx, cell, cell_inv);

    RealType R[3][3];
    angle_axis_to_matrix(aa, R);
    std::vector<RealType> rotated(3 * ns);
    for (int i = 0; i < ns; ++i)
        for (int d = 0; d < 3; ++d)
            rotated[i*3+d] = R[d][0]*mol.pos[i*3+0] + R[d][1]*mol.pos[i*3+1] +
                             R[d][2]*mol.pos[i*3+2];

    // symops in natural order (clash set is order-independent)
    std::vector<RealType> sym_rot(9 * nop), sym_trans(3 * nop);
    for (int op = 0; op < nop; ++op) {
        double rot[3][3], tr[3];
        get_symop(sg, op, rot, tr);
        for (int r = 0; r < 3; ++r)
            for (int cc = 0; cc < 3; ++cc) sym_rot[op*9 + r*3 + cc] = rot[r][cc];
        for (int d = 0; d < 3; ++d) sym_trans[op*3 + d] = tr[d];
    }

    std::vector<RealType> all_cart(3 * ns * nop);
    std::vector<RealType> mol0_frac(3 * ns);
    gpu_build_supercell(rotated.data(), ns, trans, cell, cell_inv,
                        sym_rot.data(), sym_trans.data(), nop,
                        all_cart.data(), mol0_frac.data(), nullptr);

    const RealType tol = 1e-5;  // absorb reconstruction rounding
    int clashes = 0;
    for (int dx = -2; dx <= 2; ++dx)
    for (int dy = -2; dy <= 2; ++dy)
    for (int dz = -2; dz <= 2; ++dz) {
        RealType shift[3] = {0, 0, 0};
        for (int k = 0; k < 3; ++k)
            shift[k] = dx*cell[k][0] + dy*cell[k][1] + dz*cell[k][2];
        const bool primary = (dx == 0 && dy == 0 && dz == 0);
        for (int mi = 0; mi < nop; ++mi)
        for (int mj = mi; mj < nop; ++mj) {
            if (mi == mj && primary) continue;
            for (int si = 0; si < ns; ++si)
            for (int sj = 0; sj < ns; ++sj) {
                RealType dr[3];
                for (int k = 0; k < 3; ++k)
                    dr[k] = all_cart[(mi*ns+si)*3+k] -
                            (all_cart[(mj*ns+sj)*3+k] + shift[k]);
                const RealType r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                const RealType sum = mol.cov[si] + mol.cov[sj] + buffer - tol;
                if (r2 < sum * sum) clashes++;
            }
        }
    }
    return clashes;
}

// Upload a molecule + per-component metadata to device (n_indep = 1).
struct DevMol {
    View2D<RealType> bp; View1D<RealType> cov, vdw;
    View1D<int> mns, mso; View1D<RealType> mR; int ns;
};
DevMol upload(const Mol& mol) {
    const int ns = mol.n();
    DevMol d{View2D<RealType>("bp", ns, 3), View1D<RealType>("cov", ns),
             View1D<RealType>("vdw", ns), View1D<int>("mns", 1),
             View1D<int>("mso", 1), View1D<RealType>("mR", 1), ns};
    auto h_bp = Kokkos::create_mirror_view(d.bp);
    auto h_cov = Kokkos::create_mirror_view(d.cov);
    auto h_vdw = Kokkos::create_mirror_view(d.vdw);
    RealType cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < ns; ++i) {
        for (int k = 0; k < 3; ++k) h_bp(i, k) = mol.pos[i*3+k];
        h_cov(i) = mol.cov[i]; h_vdw(i) = mol.vdw[i];
        cx += mol.pos[i*3]; cy += mol.pos[i*3+1]; cz += mol.pos[i*3+2];
    }
    cx /= ns; cy /= ns; cz /= ns;
    RealType Rmax = 0;
    for (int i = 0; i < ns; ++i) {
        RealType dx = mol.pos[i*3]-cx, dy = mol.pos[i*3+1]-cy, dz = mol.pos[i*3+2]-cz;
        Rmax = std::max(Rmax, std::sqrt(dx*dx+dy*dy+dz*dz) + mol.vdw[i]);
    }
    Kokkos::deep_copy(d.bp, h_bp); Kokkos::deep_copy(d.cov, h_cov); Kokkos::deep_copy(d.vdw, h_vdw);
    auto h_mns = Kokkos::create_mirror_view(d.mns);
    auto h_mso = Kokkos::create_mirror_view(d.mso);
    auto h_mR = Kokkos::create_mirror_view(d.mR);
    h_mns(0) = ns; h_mso(0) = 0; h_mR(0) = Rmax;
    Kokkos::deep_copy(d.mns, h_mns); Kokkos::deep_copy(d.mso, h_mso); Kokkos::deep_copy(d.mR, h_mR);
    return d;
}

}  // namespace

TEST_CASE("generator: structures depend only on seed, not rank/nranks", "[csp][generate]") {
    // Seeds are a contiguous global index, not strided by rank, so the same
    // seed_start yields identical structures however work is split.
    Mol mol = makeMol();
    DevMol d = upload(mol);
    const int sg_num = 14, n = 256, seed_start = 1;
    const RealType buffer = 0.5; const double tvp = 1.2;

    auto gen = [&](int rank, int nranks) {
        View2D<RealType> cp("cp", n, 6);
        View3D<RealType> tr("tr", n, 1, 3), rv("rv", n, 1, 3);
        View1D<int> vf("vf", n);
        batch_generate_structures(sg_num, seed_start, n, d.bp, d.cov, d.vdw,
            d.ns, tvp, buffer, cp, tr, rv, vf, CollisionMethod::DIRECT, 1,
            d.mns, d.mso, d.mR, rank, nranks);
        auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cp);
        auto h_vf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vf);
        std::vector<int> vfv(n); std::vector<double> cpv(n*6);
        for (int i = 0; i < n; ++i) {
            vfv[i] = h_vf(i);
            for (int k = 0; k < 6; ++k) cpv[i*6+k] = h_cp(i, k);
        }
        return std::make_pair(vfv, cpv);
    };

    auto [vf_a, cp_a] = gen(/*rank=*/0, /*nranks=*/1);
    auto [vf_b, cp_b] = gen(/*rank=*/5, /*nranks=*/16);

    REQUIRE(vf_a == vf_b);   // same seeds -> same validity pattern
    REQUIRE(cp_a == cp_b);   // ...and bit-identical cells
}

TEST_CASE("generator: accepted structures are clash-free", "[csp][generate]") {
    Mol mol = makeMol();
    const int ns = mol.n();
    const int sg_num = 14;      // P2_1/c, monoclinic, 4 symops
    const RealType buffer = 0.5;
    const double tvp = 1.2;

    View2D<RealType> d_bp("bp", ns, 3);
    View1D<RealType> d_cov("cov", ns), d_vdw("vdw", ns);
    {
        auto h_bp = Kokkos::create_mirror_view(d_bp);
        auto h_cov = Kokkos::create_mirror_view(d_cov);
        auto h_vdw = Kokkos::create_mirror_view(d_vdw);
        for (int i = 0; i < ns; ++i) {
            for (int d = 0; d < 3; ++d) h_bp(i, d) = mol.pos[i*3+d];
            h_cov(i) = mol.cov[i];
            h_vdw(i) = mol.vdw[i];
        }
        Kokkos::deep_copy(d_bp, h_bp);
        Kokkos::deep_copy(d_cov, h_cov);
        Kokkos::deep_copy(d_vdw, h_vdw);
    }

    View1D<int> d_mns("mns", 1), d_mso("mso", 1);
    View1D<RealType> d_mR("mR", 1);
    {
        // bounding sphere radius: centroid -> max(|atom-c| + vdw)
        RealType cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < ns; ++i) { cx += mol.pos[i*3]; cy += mol.pos[i*3+1]; cz += mol.pos[i*3+2]; }
        cx /= ns; cy /= ns; cz /= ns;
        RealType Rmax = 0;
        for (int i = 0; i < ns; ++i) {
            RealType dx = mol.pos[i*3]-cx, dy = mol.pos[i*3+1]-cy, dz = mol.pos[i*3+2]-cz;
            Rmax = std::max(Rmax, std::sqrt(dx*dx+dy*dy+dz*dz) + mol.vdw[i]);
        }
        auto h_mns = Kokkos::create_mirror_view(d_mns);
        auto h_mso = Kokkos::create_mirror_view(d_mso);
        auto h_mR  = Kokkos::create_mirror_view(d_mR);
        h_mns(0) = ns; h_mso(0) = 0; h_mR(0) = Rmax;
        Kokkos::deep_copy(d_mns, h_mns);
        Kokkos::deep_copy(d_mso, h_mso);
        Kokkos::deep_copy(d_mR, h_mR);
    }

    const int n_seeds = 512;
    View2D<RealType> cp("cp", n_seeds, 6);
    View3D<RealType> tr("tr", n_seeds, 1, 3);
    View3D<RealType> rv("rv", n_seeds, 1, 3);
    View1D<int> vf("vf", n_seeds);

    int n_valid = batch_generate_structures(
        sg_num, /*seed_start=*/1, n_seeds, d_bp, d_cov, d_vdw,
        ns, tvp, buffer, cp, tr, rv, vf,
        CollisionMethod::DIRECT, /*n_indep=*/1, d_mns, d_mso, d_mR);

    REQUIRE(n_valid > 0);

    auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cp);
    auto h_tr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tr);
    auto h_rv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rv);
    auto h_vf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vf);

    int total_clashes = 0, checked = 0;
    double vol_sum = 0;
    for (int i = 0; i < n_seeds; ++i) {
        if (!h_vf(i)) continue;
        RealType cpv[6], trv[3], aav[3];
        for (int k = 0; k < 6; ++k) cpv[k] = h_cp(i, k);
        for (int k = 0; k < 3; ++k) { trv[k] = h_tr(i, 0, k); aav[k] = h_rv(i, 0, k); }
        int cl = countClashes(mol, cpv, trv, aav, sg_num, buffer);
        total_clashes += cl;
        // volume in A^3 from the cell params
        const RealType a = cpv[0]*BOHR_TO_ANGSTROM, b = cpv[1]*BOHR_TO_ANGSTROM, c = cpv[2]*BOHR_TO_ANGSTROM;
        const RealType al = cpv[3]*M_PI/180, be = cpv[4]*M_PI/180, ga = cpv[5]*M_PI/180;
        const RealType ca = std::cos(al), cb = std::cos(be), cg = std::cos(ga);
        vol_sum += a*b*c*std::sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
        checked++;
    }
    std::printf("[generate] SG %d: %d/%d valid, mean cell volume %.1f A^3, "
                "total clashes %d\n", sg_num, n_valid, n_seeds,
                checked ? vol_sum/checked : 0.0, total_clashes);

    REQUIRE(total_clashes == 0);
}

TEST_CASE("generator: space groups with more than 48 symops", "[csp][generate]") {
    // Groups with 49 to 128 operations (F- and I-centred cubic) must be checked
    // against their real operators.
    Mol mol = makeMol();
    DevMol d = upload(mol);
    const int sg_num = 202;   // Fm-3: 96 operations, inside the window
    const auto& sg = get_space_group(sg_num);
    REQUIRE(sg.n_symops > 48);
    REQUIRE(sg.n_symops <= 128);   // above this the MAX_MOLS_CC guard rejects it

    const RealType buffer = 0.5;
    const double tvp = 1.2;
    const int n_seeds = 256;
    View2D<RealType> cp("cp", n_seeds, 6);
    View3D<RealType> tr("tr", n_seeds, 1, 3), rv("rv", n_seeds, 1, 3);
    View1D<int> vf("vf", n_seeds);
    const int n_valid = batch_generate_structures(
        sg_num, /*seed_start=*/1, n_seeds, d.bp, d.cov, d.vdw,
        d.ns, tvp, buffer, cp, tr, rv, vf,
        CollisionMethod::DIRECT, /*n_indep=*/1, d.mns, d.mso, d.mR);
    REQUIRE(n_valid > 0);

    auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cp);
    auto h_tr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tr);
    auto h_rv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rv);
    auto h_vf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vf);

    // Reconstructed against the real operators, every accepted cell is clash-free.
    int total_clashes = 0;
    for (int i = 0; i < n_seeds; ++i) {
        if (!h_vf(i)) continue;
        RealType cpv[6], trv[3], aav[3];
        for (int k = 0; k < 6; ++k) cpv[k] = h_cp(i, k);
        for (int k = 0; k < 3; ++k) { trv[k] = h_tr(i, 0, k); aav[k] = h_rv(i, 0, k); }
        total_clashes += countClashes(mol, cpv, trv, aav, sg_num, buffer);
    }
    std::printf("[generate] SG %d (%d symops): %d/%d valid, total clashes %d\n",
                sg_num, sg.n_symops, n_valid, n_seeds, total_clashes);
    REQUIRE(total_clashes == 0);
}

TEST_CASE("generator: P1 acceptance not starved by volume budget", "[csp][generate]") {
    Mol mol = makeRodMol();
    const int ns = mol.n();
    const RealType buffer = 0.5;
    const double tvp = 1.2;
    const int n_seeds = 2000;

    View2D<RealType> d_bp("bp", ns, 3);
    View1D<RealType> d_cov("cov", ns), d_vdw("vdw", ns);
    {
        auto h_bp = Kokkos::create_mirror_view(d_bp);
        auto h_cov = Kokkos::create_mirror_view(d_cov);
        auto h_vdw = Kokkos::create_mirror_view(d_vdw);
        for (int i = 0; i < ns; ++i) {
            for (int d = 0; d < 3; ++d) h_bp(i, d) = mol.pos[i*3+d];
            h_cov(i) = mol.cov[i]; h_vdw(i) = mol.vdw[i];
        }
        Kokkos::deep_copy(d_bp, h_bp);
        Kokkos::deep_copy(d_cov, h_cov);
        Kokkos::deep_copy(d_vdw, h_vdw);
    }
    View1D<int> d_mns("mns", 1), d_mso("mso", 1);
    View1D<RealType> d_mR("mR", 1);
    {
        RealType cx=0,cy=0,cz=0;
        for (int i=0;i<ns;++i){cx+=mol.pos[i*3];cy+=mol.pos[i*3+1];cz+=mol.pos[i*3+2];}
        cx/=ns; cy/=ns; cz/=ns; RealType Rmax=0;
        for (int i=0;i<ns;++i){RealType dx=mol.pos[i*3]-cx,dy=mol.pos[i*3+1]-cy,dz=mol.pos[i*3+2]-cz;
            Rmax=std::max(Rmax,std::sqrt(dx*dx+dy*dy+dz*dz)+mol.vdw[i]);}
        auto h_mns=Kokkos::create_mirror_view(d_mns); auto h_mso=Kokkos::create_mirror_view(d_mso);
        auto h_mR=Kokkos::create_mirror_view(d_mR);
        h_mns(0)=ns; h_mso(0)=0; h_mR(0)=Rmax;
        Kokkos::deep_copy(d_mns,h_mns); Kokkos::deep_copy(d_mso,h_mso); Kokkos::deep_copy(d_mR,h_mR);
    }

    View2D<RealType> cp("cp", n_seeds, 6);
    View3D<RealType> tr("tr", n_seeds, 1, 3), rv("rv", n_seeds, 1, 3);
    View1D<int> vf("vf", n_seeds);
    auto accept = [&](int sg) {
        int nv = batch_generate_structures(sg, 1, n_seeds, d_bp, d_cov, d_vdw,
            ns, tvp, buffer, cp, tr, rv, vf, CollisionMethod::DIRECT, 1, d_mns, d_mso, d_mR);
        return double(nv) / n_seeds;
    };
    const double p1  = accept(1);   // P1,  triclinic, 1 symop  (cell holds 1 molecule)
    const double pm1 = accept(2);   // P-1, triclinic, 2 symops
    std::printf("[generate] P1 accept %.1f%%  P-1 accept %.1f%%\n", 100*p1, 100*pm1);

    // P1 (one molecule in a box) must not lag P-1 in acceptance; the volume
    // budget is rotation-invariant.
    REQUIRE(pm1 > 0.0);
    REQUIRE(p1 > 0.30);
    REQUIRE(p1 >= 0.5 * pm1);
}

// cspy's MOI-oriented mol5: cov all 0.68, vdw per atom.
Mol makeCspyMol() {
    Mol m;
    const RealType p[5][3] = {
        {-1.534649, -0.478707,  0.028115},
        {-0.756477,  0.803143,  0.064281},
        {-0.038715, -0.505873, -0.089972},
        { 0.739456,  0.775978, -0.053806},
        { 1.480355, -0.555499,  0.090203}};
    for (auto& a : p) m.pos.insert(m.pos.end(), {a[0], a[1], a[2]});
    m.cov.assign(5, 0.68);
    m.vdw = {1.70, 1.70, 1.52, 1.55, 1.70};
    return m;
}

TEST_CASE("generator: cspy distribution compare", "[csp][generate][.compat]") {
    Mol mol = makeCspyMol();
    const int ns = mol.n();
    const RealType buffer = 0.5;
    const double tvp = 2.5;  // cspy default
    const int n_seeds = 4000;

    View2D<RealType> d_bp("bp", ns, 3);
    View1D<RealType> d_cov("cov", ns), d_vdw("vdw", ns);
    {
        auto h_bp=Kokkos::create_mirror_view(d_bp); auto h_cov=Kokkos::create_mirror_view(d_cov); auto h_vdw=Kokkos::create_mirror_view(d_vdw);
        for (int i=0;i<ns;++i){ for(int d=0;d<3;++d) h_bp(i,d)=mol.pos[i*3+d]; h_cov(i)=mol.cov[i]; h_vdw(i)=mol.vdw[i]; }
        Kokkos::deep_copy(d_bp,h_bp); Kokkos::deep_copy(d_cov,h_cov); Kokkos::deep_copy(d_vdw,h_vdw);
    }
    View1D<int> d_mns("mns",1), d_mso("mso",1); View1D<RealType> d_mR("mR",1);
    {
        RealType cx=0,cy=0,cz=0; for(int i=0;i<ns;++i){cx+=mol.pos[i*3];cy+=mol.pos[i*3+1];cz+=mol.pos[i*3+2];}
        cx/=ns;cy/=ns;cz/=ns; RealType Rmax=0;
        for(int i=0;i<ns;++i){RealType dx=mol.pos[i*3]-cx,dy=mol.pos[i*3+1]-cy,dz=mol.pos[i*3+2]-cz; Rmax=std::max(Rmax,std::sqrt(dx*dx+dy*dy+dz*dz)+mol.vdw[i]);}
        auto h_mns=Kokkos::create_mirror_view(d_mns); auto h_mso=Kokkos::create_mirror_view(d_mso); auto h_mR=Kokkos::create_mirror_view(d_mR);
        h_mns(0)=ns; h_mso(0)=0; h_mR(0)=Rmax;
        Kokkos::deep_copy(d_mns,h_mns); Kokkos::deep_copy(d_mso,h_mso); Kokkos::deep_copy(d_mR,h_mR);
    }

    View2D<RealType> cp("cp",n_seeds,6); View3D<RealType> tr("tr",n_seeds,1,3), rv("rv",n_seeds,1,3); View1D<int> vf("vf",n_seeds);
    const char* nbE = std::getenv("KLASP_GEN_NUDGE"); const char* ctE = std::getenv("KLASP_GEN_CONTRACT");
    const int nudge = nbE ? std::atoi(nbE) : 16;
    const bool contract = ctE ? (std::atoi(ctE) != 0) : true;
    std::printf("[compare] config nudge=%d contract=%d tvp=%.1f  (cspy ref: SG2 72%% med135 min59 | SG14 70%% med236 min121 | SG19 90%% med194 min131)\n",
                nudge, contract, tvp);
    for (int sg : {2, 14, 19}) {
        int nv = batch_generate_structures(sg, 1, n_seeds, d_bp, d_cov, d_vdw,
            ns, tvp, buffer, cp, tr, rv, vf, CollisionMethod::DIRECT, 1, d_mns, d_mso, d_mR,
            0, 1, nudge, contract);
        auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cp);
        auto h_vf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vf);
        std::vector<double> vols, aspect, minedge;
        for (int i=0;i<n_seeds;++i) if (h_vf(i)) {
            double a=h_cp(i,0)*BOHR_TO_ANGSTROM,b=h_cp(i,1)*BOHR_TO_ANGSTROM,c=h_cp(i,2)*BOHR_TO_ANGSTROM;
            double al=h_cp(i,3)*M_PI/180,be=h_cp(i,4)*M_PI/180,ga=h_cp(i,5)*M_PI/180;
            double ca=std::cos(al),cb=std::cos(be),cg=std::cos(ga);
            vols.push_back(a*b*c*std::sqrt(1-ca*ca-cb*cb-cg*cg+2*ca*cb*cg));
            double e[3]={a,b,c}; std::sort(e,e+3);
            aspect.push_back(e[2]/e[0]); minedge.push_back(e[0]);
        }
        std::sort(vols.begin(), vols.end()); std::sort(aspect.begin(),aspect.end()); std::sort(minedge.begin(),minedge.end());
        auto pc=[](std::vector<double>&v,double q){ return v.empty()?0.0:v[(size_t)(q*(v.size()-1))]; };
        double mean=0; for(double v:vols) mean+=v; if(!vols.empty()) mean/=vols.size();
        std::printf("  SG %2d: accept %5.1f%%  vol mean=%.1f med=%.1f min=%.1f  aspect med=%.2f p90=%.2f max=%.2f  minedge med=%.2f min=%.2f\n",
                    sg, 100.0*nv/n_seeds, mean, pc(vols,0.5), vols.empty()?0.0:vols.front(),
                    pc(aspect,0.5), pc(aspect,0.9), aspect.empty()?0.0:aspect.back(),
                    pc(minedge,0.5), minedge.empty()?0.0:minedge.front());
    }
    REQUIRE(true);
}

TEST_CASE("generator: throughput", "[csp][generate][.bench]") {
    Mol mol = makeMol();
    const int ns = mol.n();
    const int sg_num = 14;
    const RealType buffer = 0.5;
    const double tvp = 1.2;

    View2D<RealType> d_bp("bp", ns, 3);
    View1D<RealType> d_cov("cov", ns), d_vdw("vdw", ns);
    {
        auto h_bp = Kokkos::create_mirror_view(d_bp);
        auto h_cov = Kokkos::create_mirror_view(d_cov);
        auto h_vdw = Kokkos::create_mirror_view(d_vdw);
        for (int i = 0; i < ns; ++i) {
            for (int d = 0; d < 3; ++d) h_bp(i, d) = mol.pos[i*3+d];
            h_cov(i) = mol.cov[i]; h_vdw(i) = mol.vdw[i];
        }
        Kokkos::deep_copy(d_bp, h_bp);
        Kokkos::deep_copy(d_cov, h_cov);
        Kokkos::deep_copy(d_vdw, h_vdw);
    }
    View1D<int> d_mns("mns", 1), d_mso("mso", 1);
    View1D<RealType> d_mR("mR", 1);
    {
        RealType cx=0,cy=0,cz=0;
        for (int i=0;i<ns;++i){cx+=mol.pos[i*3];cy+=mol.pos[i*3+1];cz+=mol.pos[i*3+2];}
        cx/=ns; cy/=ns; cz/=ns; RealType Rmax=0;
        for (int i=0;i<ns;++i){RealType dx=mol.pos[i*3]-cx,dy=mol.pos[i*3+1]-cy,dz=mol.pos[i*3+2]-cz;
            Rmax=std::max(Rmax,std::sqrt(dx*dx+dy*dy+dz*dz)+mol.vdw[i]);}
        auto h_mns=Kokkos::create_mirror_view(d_mns); auto h_mso=Kokkos::create_mirror_view(d_mso);
        auto h_mR=Kokkos::create_mirror_view(d_mR);
        h_mns(0)=ns; h_mso(0)=0; h_mR(0)=Rmax;
        Kokkos::deep_copy(d_mns,h_mns); Kokkos::deep_copy(d_mso,h_mso); Kokkos::deep_copy(d_mR,h_mR);
    }

    const int n_seeds = 20000;
    View2D<RealType> cp("cp", n_seeds, 6);
    View3D<RealType> tr("tr", n_seeds, 1, 3), rv("rv", n_seeds, 1, 3);
    View1D<int> vf("vf", n_seeds);

    const char* nbE = std::getenv("KLASP_GEN_NUDGE"); const char* ctE = std::getenv("KLASP_GEN_CONTRACT");
    const int nudge = nbE ? std::atoi(nbE) : 16;
    const bool contract = ctE ? (std::atoi(ctE) != 0) : true;
    auto run = [&](int seed_start) {
        return batch_generate_structures(sg_num, seed_start, n_seeds, d_bp, d_cov, d_vdw,
            ns, tvp, buffer, cp, tr, rv, vf, CollisionMethod::DIRECT, 1, d_mns, d_mso, d_mR,
            0, 1, nudge, contract);
    };
    run(1);  // warm up
    const int reps = 3;
    double best_ms = 1e30; int nv = 0;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        nv = run(1 + r*n_seeds);
        auto t1 = std::chrono::steady_clock::now();
        best_ms = std::min(best_ms,
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::printf("[bench] %d seeds (serial): %.2f ms  (%.0f seeds/s)  accept %d/%d\n",
                n_seeds, best_ms, n_seeds / (best_ms/1000.0), nv, n_seeds);
    REQUIRE(nv > 0);
}


TEST_CASE("generator contract: the request form is the free function", "[csp][generate]") {
    // QuasiRandomGenerator must produce the identical batch to the raw kernel.
    Mol mol = makeMol();
    DevMol d = upload(mol);
    const int sg_num = 14, n = 256;
    const double tvp = 1.2, buffer = 0.5;

    auto alloc = [&](const char* tag) {
        DofBatch b;
        b.cell_params  = View2D<RealType>(std::string("cp") + tag, n, 6);
        b.translations = View3D<RealType>(std::string("tr") + tag, n, 1, 3);
        b.rot_vectors  = View3D<RealType>(std::string("rv") + tag, n, 1, 3);
        b.seeds        = View1D<int>(std::string("sd") + tag, n);
        b.capacity = n;
        b.n_indep = 1;
        return b;
    };

    DofBatch direct = alloc("d");
    int raw_direct = 0;
    const int n_direct = generate_and_compact_on_device(
        sg_num, /*seed_start=*/1, n, d.bp, d.cov, d.vdw, d.ns, tvp, buffer,
        CollisionMethod::DIRECT,
        direct.cell_params, direct.translations, direct.rot_vectors, direct.seeds,
        n, 1, d.mns, d.mso, d.mR, 0, 1, 16, true, &raw_direct);

    QuasiRandomGenerator gen;
    gen.body_positions_ang = d.bp;
    gen.cov_radii = d.cov;
    gen.vdw_radii = d.vdw;
    gen.n_sites = d.ns;
    gen.tvp = tvp;
    gen.collision_buffer = buffer;
    gen.mol_n_sites = d.mns;
    gen.mol_site_offsets = d.mso;
    gen.mol_bounding_radius = d.mR;

    DofBatch via = alloc("g");
    int raw_via = 0;
    const int n_via = gen.generate({sg_num, /*seed_start=*/1, n}, via, &raw_via);

    REQUIRE(n_via == n_direct);
    REQUIRE(raw_via == raw_direct);
    REQUIRE(n_via > 0);

    auto h_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), direct.cell_params);
    auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), via.cell_params);
    auto s_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), direct.seeds);
    auto s_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), via.seeds);
    for (int i = 0; i < n_via; ++i) {
        REQUIRE(s_a(i) == s_b(i));
        for (int k = 0; k < 6; ++k) REQUIRE(h_a(i, k) == h_b(i, k));
    }
}

TEST_CASE("sampling prior: the extracted draw is the generator's own", "[csp][generate]") {
    // Orientation and cell angles survive clash resolution untouched, so for
    // every accepted seed they must equal what prior_draw_pose produces.
    Mol mol = makeMol();
    DevMol d = upload(mol);
    const int n = 256, seed_start = 1;

    for (int sg_num : {2, 14, 19, 61}) {
        View2D<RealType> cp("cp", n, 6);
        View3D<RealType> tr("tr", n, 1, 3), rv("rv", n, 1, 3);
        View1D<int> vf("vf", n);
        const int nv = batch_generate_structures(
            sg_num, seed_start, n, d.bp, d.cov, d.vdw, d.ns, 1.2, 0.5,
            cp, tr, rv, vf, CollisionMethod::DIRECT, 1, d.mns, d.mso, d.mR);
        REQUIRE(nv > 0);

        auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cp);
        auto h_rv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rv);
        auto h_vf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vf);

        // max_volume only bounds the lengths, which this case does not check.
        const SamplingPrior prior = make_sampling_prior(sg_num, 1, 0.0);
        REQUIRE(prior.ndim() <= PRIOR_MAX_DIM);

        // The reference draw runs on the device, like the generator: host and
        // device compilations can differ in the last bit, and the check is exact.
        View3D<RealType> ref_rv("ref_rv", n, 1, 3);
        View2D<RealType> ref_ang("ref_ang", n, 3);
        View1D<int> ref_ok("ref_ok", n);
        Kokkos::parallel_for("prior_reference", Kokkos::RangePolicy<ExecSpace>(0, n),
                             KOKKOS_LAMBDA(int i) {
            RealType qr[PRIOR_MAX_DIM];
            kgf_point(seed_start + i, prior.ndim(), qr);
            PriorDraw dof;
            if (!prior_draw_pose(prior, qr, dof)) { ref_ok(i) = 0; return; }
            ref_ok(i) = 1;
            for (int k = 0; k < 3; ++k) ref_rv(i, 0, k) = dof.rot_vec[0][k];
            ref_ang(i, 0) = static_cast<RealType>(dof.alpha * 180.0 / M_PI);
            ref_ang(i, 1) = static_cast<RealType>(dof.beta  * 180.0 / M_PI);
            ref_ang(i, 2) = static_cast<RealType>(dof.gamma * 180.0 / M_PI);
        });
        Kokkos::fence();
        auto h_ref_rv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_rv);
        auto h_ref_ang = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_ang);
        auto h_ref_ok = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_ok);

        for (int i = 0; i < n; ++i) {
            if (!h_vf(i)) continue;
            INFO("sg " << sg_num << " seed " << (seed_start + i));
            REQUIRE(h_ref_ok(i) == 1);      // an accepted seed cannot be a rejected draw
            for (int k = 0; k < 3; ++k) REQUIRE(h_rv(i, 0, k) == h_ref_rv(i, 0, k));
            REQUIRE(h_cp(i, 3) == h_ref_ang(i, 0));
            REQUIRE(h_cp(i, 4) == h_ref_ang(i, 1));
            REQUIRE(h_cp(i, 5) == h_ref_ang(i, 2));
        }
    }
}

TEST_CASE("landscape generator: a seed window of stored structures", "[csp][generate]") {
    // Every row a landscape generator emits carries the seed it was stored
    // under, and the request's window is a window of those seeds.
    LandscapeGenerator lg;
    auto row = [](int seed, int sg, int n_mols) {
        StructureResult r{};
        r.seed = seed; r.space_group = sg; r.stage = 2; r.n_mols = n_mols;
        for (int k = 0; k < 6; ++k) r.cell_params[k] = 10.0 + seed + k;
        for (int k = 0; k < 3 * n_mols; ++k) {
            r.translation[k] = 0.01 * (seed + k);
            r.rotation[k] = 0.1 * (seed - k);
        }
        return r;
    };
    // Out of seed order and mixed across space groups, as read_landscape
    // returns them (sorted by energy).
    lg.rows = {row(7, 14, 1), row(3, 2, 1), row(4, 14, 2), row(1, 14, 1),
               row(9, 14, 1), row(5, 14, 1)};
    std::sort(lg.rows.begin(), lg.rows.end(),
              [](const StructureResult& a, const StructureResult& b) {
                  return std::make_pair(a.space_group, a.seed) <
                         std::make_pair(b.space_group, b.seed);
              });

    REQUIRE(lg.seed_end(14) == 10);
    REQUIRE(lg.seed_end(2) == 4);
    REQUIRE(lg.seed_end(33) == 0);

    const int cap = 8;
    DofBatch out;
    out.cell_params  = View2D<RealType>("cp", cap, 6);
    out.translations = View3D<RealType>("tr", cap, 1, 3);
    out.rot_vectors  = View3D<RealType>("rv", cap, 1, 3);
    out.seeds        = View1D<int>("sd", cap);
    out.capacity = cap;
    out.n_indep = 1;

    // Window [1, 8): seeds 1, 5, 7 of SG 14. Seed 3 is another space group, and
    // seed 4 is a Z'=2 row that a Z'=1 destination cannot hold.
    const int n = lg.generate({14, 1, 7}, out);
    REQUIRE(n == 3);

    auto h_sd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out.seeds);
    auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out.cell_params);
    REQUIRE(h_sd(0) == 1);
    REQUIRE(h_sd(1) == 5);
    REQUIRE(h_sd(2) == 7);
    for (int i = 0; i < n; ++i)
        REQUIRE(h_cp(i, 0) == static_cast<RealType>(10.0 + h_sd(i)));

    // The window is half-open and clamps to the batch.
    REQUIRE(lg.generate({14, 9, 1}, out) == 1);
    REQUIRE(lg.generate({14, 10, 100}, out) == 0);
    REQUIRE(lg.generate({99, 0, 100}, out) == 0);
    out.capacity = 2;
    REQUIRE(lg.generate({14, 0, 100}, out) == 2);
}
