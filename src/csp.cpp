#include "klasp/log.hpp"
#include "klasp/csp.hpp"
#include "klasp/sobol.hpp"
#include "klasp/kgf.hpp"
#include "klasp/rotation.hpp"
#include "klasp/constants.hpp"
#include "klasp/batch.hpp"
#include "klasp/generate_kernel.hpp"
#include "klasp/sampling_prior.hpp"
#include "klasp/sphere_mtv.hpp"
#include "klasp/volume_budget.hpp"

#include <fmt/base.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace klasp {

// Gap (Angstrom) above which the generator shifts content along an axis before
// contracting. Well below warnings.hpp's WARN_VACUUM_GAP_A.
static constexpr double kGenVacuumTriggerA = 4.0;

// On-device structure generator, one thread per seed: sample DOFs from a KGF
// quasi-random sequence, apply a minimum-volume bound from per-component bounding
// boxes, then resolve clashes (bounding-sphere then per-atom filter) by nudging
// and expanding until clash-free or the iteration/volume cap is hit. Per-seed
// scratch lives in device Views sized to the basis. Handles any n_indep >= 1.

int batch_generate_structures(
    int space_group, int seed_start, int n_seeds,
    const View2D<RealType>& body_positions_ang,
    const View1D<RealType>& cov_radii,
    const View1D<RealType>& vdw_radii,
    int n_sites, double tvp, double collision_buffer,
    View2D<RealType> cell_params_out,
    View3D<RealType> translations_out,
    View3D<RealType> rot_vectors_out,
    View1D<int> valid_flags_out,
    CollisionMethod collision_method,
    int n_indep,
    const View1D<int>& mol_n_sites,
    const View1D<int>& mol_site_offsets,
    const View1D<RealType>& mol_bounding_radius,
    int rank, int nranks,
    int nudge_budget, bool contract,
    SamplingPrior* prior_out) {

    using ExecSpace = Kokkos::DefaultExecutionSpace;

    if (n_indep > MAX_INDEP) {
        logging::warn("batch_generate_structures_multi: n_indep={} exceeds MAX_INDEP={}\n",
                   n_indep, MAX_INDEP);
        return 0;
    }
    if (mol_n_sites.extent(0) < static_cast<size_t>(n_indep) ||
        mol_site_offsets.extent(0) < static_cast<size_t>(n_indep)) {
        logging::warn("batch_generate_structures_multi: mol_n_sites/offsets Views too small\n");
        return 0;
    }
    // RIGID_PRESS runs after this kernel and FFT_PACK has no multi-component
    // path, so both use the direct collision check here.
    (void)collision_method;

    auto& sg_info = get_space_group(space_group);
    int n_symops = sg_info.n_symops;
    LatticeType lattice = sg_info.lattice;

    // Crystal operators, identity first, via the same builder as
    // build_csp_molecule: the packing is read back through that table.
    const SymmetryOps gen_ops = crystal_symmetry_ops(sg_info);
    View2D<RealType> d_sym_rot   = gen_ops.sym_rotations;
    View2D<RealType> d_sym_trans = gen_ops.sym_translations;

    // Per-component bounding box and max vdW (host side).
    auto h_bp = to_host(body_positions_ang);
    auto h_vdw = to_host(vdw_radii);
    auto h_mns = to_host(mol_n_sites);
    auto h_mso = to_host(mol_site_offsets);

    // Volume budget and sizes from volume_budget.hpp, shared by every generator.
    std::vector<double> bp_flat(static_cast<std::size_t>(h_bp.extent(0)) * 3);
    std::vector<double> vdw_flat(h_vdw.extent(0));
    for (std::size_t i = 0; i < h_bp.extent(0); ++i)
        for (int d = 0; d < 3; ++d)
            bp_flat[i * 3 + d] = static_cast<double>(h_bp(i, d));
    for (std::size_t i = 0; i < h_vdw.extent(0); ++i)
        vdw_flat[i] = static_cast<double>(h_vdw(i));
    std::vector<int> mns_flat(n_indep), mso_flat(n_indep);
    for (int m = 0; m < n_indep; ++m) { mns_flat[m] = h_mns(m); mso_flat[m] = h_mso(m); }

    const VolumeBudget budget = volume_budget(bp_flat.data(), vdw_flat.data(),
                                              mns_flat.data(), mso_flat.data(),
                                              n_indep, n_symops, lattice, tvp);
    const double max_vdw = budget.max_vdw;
    const int total_real_sites = budget.total_real_sites;
    const int max_ns_m = budget.max_sites_per_mol;
    double max_volume = budget.max_volume;

    const int total_atoms_max = total_real_sites * n_symops;

    // Per-seed scratch for the generator's large arrays.
    // WithoutInitializing: the kernel writes before reading every slot.
    View3D<RealType> d_gen_rotated(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "gen_rotated"),
        n_seeds, total_real_sites, 3);
    View3D<RealType> d_gen_all_cart(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "gen_all_cart"),
        n_seeds, total_atoms_max, 3);
    Kokkos::View<int**, Kokkos::LayoutRight, ExecSpace> d_gen_atom_site(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "gen_atom_site"),
        n_seeds, total_atoms_max);
    // mol0_frac: gpu_build_supercell's fractional-position scratch, sized to the
    // largest independent molecule, one slice per seed.
    View3D<RealType> d_gen_mol0_frac(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "gen_mol0_frac"),
        n_seeds, max_ns_m, 3);

    // Per-seed scratch for the hull-MTV clash nudge (A's radii; B's shifted
    // centres + radii). Molecules larger than MTV_CAP atoms skip the nudge.
    constexpr int MTV_CAP = 96;
    const int mtv_cap = std::max(1, std::min(max_ns_m, MTV_CAP));
    View2D<RealType> d_gen_mtv_rA(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "gen_mtv_rA"),
        n_seeds, mtv_cap);
    View3D<RealType> d_gen_mtv_cB(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "gen_mtv_cB"),
        n_seeds, mtv_cap, 3);
    View2D<RealType> d_gen_mtv_rB(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "gen_mtv_rB"),
        n_seeds, mtv_cap);

    // The sampling prior this generator draws from.
    SamplingPrior prior = make_sampling_prior(space_group, n_indep, max_volume);
    if (prior_out) *prior_out = prior;

    RealType coll_buf = collision_buffer;

    Kokkos::deep_copy(valid_flags_out, 0);

    auto bp = body_positions_ang;
    auto cr = cov_radii;
    auto sr = d_sym_rot;
    auto st = d_sym_trans;
    auto mns = mol_n_sites;
    auto mso = mol_site_offsets;
    auto mR  = mol_bounding_radius;  // (n_indep,) Angstrom; empty View → filter disabled
    const bool have_mR = (mR.extent(0) >= static_cast<size_t>(n_indep));
    auto cp_out = cell_params_out;
    auto tr_out = translations_out;
    auto rv_out = rot_vectors_out;
    auto vf_out = valid_flags_out;
    auto s_rot  = d_gen_rotated;
    auto s_cart = d_gen_all_cart;
    auto s_site = d_gen_atom_site;
    auto s_m0f  = d_gen_mol0_frac;
    auto s_mtv_rA = d_gen_mtv_rA;
    auto s_mtv_cB = d_gen_mtv_cB;
    auto s_mtv_rB = d_gen_mtv_rB;
    const int mtv_cap_k = mtv_cap;
    RealType max_vol = max_volume;
    RealType mv = max_vdw;
    int n_indep_k = n_indep;
    (void)rank; (void)nranks;   // seed is now a rank-independent global index

    // Clash-resolution controls: hull-MTV nudge and expand-then-contract
    // densification.
    const int nudge_budget_k = nudge_budget;
    const bool do_contract_k = contract;

    Kokkos::parallel_for("batch_generate_cc",
        Kokkos::RangePolicy<ExecSpace>(0, n_seeds),
        KOKKOS_LAMBDA(const int idx) {
            // Contiguous global seed, not strided by rank, so the sampled set is
            // independent of rank count.
            int seed = seed_start + idx;

            RealType qr[PRIOR_MAX_DIM];
            kgf_point(seed, prior.ndim(), qr);

            // Poses and cell angles from the sampling prior; the rest is clash
            // resolution, which mutates trans_m and a/b/c (the draw's storage) in place.
            PriorDraw dof;
            if (!prior_draw_pose(prior, qr, dof)) return;

            RealType (&trans_m)[MAX_INDEP][3]  = dof.trans;
            RealType (&R_mol_m)[MAX_INDEP][3][3] = dof.rot;
            RealType (&aa_m)[MAX_INDEP][3]     = dof.rot_vec;
            const RealType alpha = dof.alpha, beta = dof.beta, gamma = dof.gamma;
            const RealType ca = dof.ca, cb = dof.cb, cg = dof.cg;
            const RealType sg_val = dof.sg_val, sqrt_x = dof.sqrt_x;

            // Rotated body positions, concatenated over components:
            // rotated_total[global_site*3 + d], comp0_site0..N0, comp1_site0..N1, ...
            RealType* rotated = &s_rot(idx, 0, 0);
            int rot_site_offsets[MAX_INDEP];
            int roff = 0;
            for (int m = 0; m < n_indep_k; ++m) {
                rot_site_offsets[m] = roff;
                int ns_m = mns(m);
                int bp_off = mso(m);
                for (int i = 0; i < ns_m; ++i) {
                    RealType bpx = bp(bp_off + i, 0);
                    RealType bpy = bp(bp_off + i, 1);
                    RealType bpz = bp(bp_off + i, 2);
                    int g = roff + i;
                    rotated[g*3+0] = R_mol_m[m][0][0]*bpx + R_mol_m[m][0][1]*bpy + R_mol_m[m][0][2]*bpz;
                    rotated[g*3+1] = R_mol_m[m][1][0]*bpx + R_mol_m[m][1][1]*bpy + R_mol_m[m][1][2]*bpz;
                    rotated[g*3+2] = R_mol_m[m][2][0]*bpx + R_mol_m[m][2][1]*bpy + R_mol_m[m][2][2]*bpz;
                }
                roff += ns_m;
            }

            // Cell minimum-size bound from union f_min/f_max
            RealType unit_inv[3][3] = {};
            unit_inv[0][0] = 1.0;
            unit_inv[1][0] = -cg / sg_val;
            unit_inv[1][1] = 1.0 / sg_val;
            unit_inv[2][0] = (ca * cg - cb) / (sqrt_x * sg_val);
            unit_inv[2][1] = (cb * cg - ca) / (sqrt_x * sg_val);
            unit_inv[2][2] = sg_val / sqrt_x;

            RealType f_min[3] = {1e30, 1e30, 1e30}, f_max[3] = {-1e30, -1e30, -1e30};
            for (int i = 0; i < total_real_sites; ++i) {
                RealType f[3] = {0, 0, 0};
                for (int d = 0; d < 3; ++d)
                    for (int k = 0; k < 3; ++k)
                        f[d] += rotated[i*3+k] * unit_inv[k][d];
                for (int d = 0; d < 3; ++d) {
                    f_min[d] = Kokkos::fmin(f_min[d], f[d]);
                    f_max[d] = Kokkos::fmax(f_max[d], f[d]);
                }
            }
            RealType s_mins[3];
            for (int d = 0; d < 3; ++d) s_mins[d] = (f_max[d] - f_min[d]) + mv * 2;
            if (!prior_accepts_volume(prior, s_mins, sqrt_x)) return;

            // Cell lengths are conditioned on s_mins.
            prior_draw_lengths(prior, qr, seed, s_mins, dof);
            RealType &a = dof.a, &b = dof.b, &c = dof.c;

            // Collision resolution: per-molecule bounding spheres, then per-atom.
            constexpr int MAX_MOLS_CC = 128;  // n_indep × n_symops upper bound
            RealType* all_cart = &s_cart(idx, 0, 0);
            int* atom_global_site = &s_site(idx, 0);
            int total_atoms = 0;
            int total_mols  = n_indep_k * n_symops;
            for (int m = 0; m < n_indep_k; ++m)
                total_atoms += n_symops * mns(m);
            if (total_atoms > total_atoms_max) return;  // safety (shouldn't trigger)
            if (total_mols  > MAX_MOLS_CC)     return;

            // Per-molecule bookkeeping (cell-invariant).
            // mol_first_atom[m_id] = where mol m_id's atoms start in all_cart
            // mol_n_atoms[m_id]    = number of atoms in mol m_id (= mns(m))
            // mol_indep_id[m_id]   = which indep mol m this is (for vdw/cov lookup)
            int mol_first_atom[MAX_MOLS_CC];
            int mol_n_atoms[MAX_MOLS_CC];
            int mol_indep_id[MAX_MOLS_CC];
            {
                int write_off = 0;
                int mid = 0;
                for (int m = 0; m < n_indep_k; ++m) {
                    int ns_m = mns(m);
                    int bp_off = mso(m);
                    for (int op = 0; op < n_symops; ++op) {
                        mol_first_atom[mid] = write_off + op * ns_m;
                        mol_n_atoms[mid]    = ns_m;
                        mol_indep_id[mid]   = m;
                        for (int i = 0; i < ns_m; ++i)
                            atom_global_site[write_off + op * ns_m + i] = bp_off + i;
                        ++mid;
                    }
                    write_off += n_symops * ns_m;
                }
            }

            // Symops read directly from the LayoutRight device Views.
            const RealType* sym_rot_local   = &sr(0, 0);
            const RealType* sym_trans_local = &st(0, 0);

            // Per-indep-mol bounding sphere radii (Angstrom). Without mR, fall
            // back to a loose bound (max_vdw x n_sites_max).
            RealType mol_R_local[MAX_INDEP];
            for (int m = 0; m < n_indep_k; ++m) {
                if (have_mR) mol_R_local[m] = mR(m);
                else         mol_R_local[m] = mv * RealType(mns(m));  // crude fallback
            }

            // Per-component Cartesian COM nudge (applied via gpu_build_supercell).
            RealType nudge_comp[MAX_INDEP][3];
            for (int m = 0; m < n_indep_k; ++m)
                nudge_comp[m][0] = nudge_comp[m][1] = nudge_comp[m][2] = 0;

            // Fill all_cart for a trial cell. Shared by the collision test and
            // the vacuum probe so both see the same geometry.
            auto build_at = [&](RealType la, RealType lb, RealType lc,
                                RealType cell[3][3], RealType cell_inv[3][3]) {
                gpu_build_cell(la, lb, lc, ca, cb, cg, sg_val, sqrt_x, cell, cell_inv);
                RealType* m0f = &s_m0f(idx, 0, 0);
                int wo = 0;
                for (int m = 0; m < n_indep_k; ++m) {
                    int ns_m = mns(m);
                    RealType* rot_m = &rotated[rot_site_offsets[m] * 3];
                    gpu_build_supercell(rot_m, ns_m, trans_m[m], cell, cell_inv,
                                        sym_rot_local, sym_trans_local, n_symops,
                                        &all_cart[wo * 3], m0f, &nudge_comp[m][0]);
                    wo += n_symops * ns_m;
                }
            };

            // Widest empty layer in the cell (Angstrom) over the 13 lattice planes
            // with h,k,l in {-1,0,1}, one supercell build for all. Plane spacing
            // comes from the reciprocal vector, not the cell edge.
            auto max_gap_at = [&](RealType la, RealType lb, RealType lc) -> RealType {
                RealType cell[3][3], cell_inv[3][3];
                build_at(la, lb, lc, cell, cell_inv);
                constexpr int NDIR = 13;
                constexpr int DIR[NDIR][3] = {
                    {0,0,1},{0,1,0},{0,1,1},{0,1,-1},
                    {1,0,0},{1,0,1},{1,0,-1},{1,1,0},{1,-1,0},
                    {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1}};
                constexpr int NB = 64;
                RealType worst = 0;
                for (int dcount = 0; dcount < NDIR; ++dcount) {
                    const int h = DIR[dcount][0], k = DIR[dcount][1], l = DIR[dcount][2];
                    // G = h a* + k b* + l c*; rows of cell_inv are the reciprocal
                    // basis, so G.cart is exactly h*fx + k*fy + l*fz.
                    RealType G[3];
                    for (int j = 0; j < 3; ++j)
                        G[j] = h*cell_inv[0][j] + k*cell_inv[1][j] + l*cell_inv[2][j];
                    const RealType gn2 = G[0]*G[0] + G[1]*G[1] + G[2]*G[2];
                    if (gn2 <= RealType(1e-20)) continue;
                    const RealType d = RealType(1) / Kokkos::sqrt(gn2);
                    bool occ[NB];
                    for (int i = 0; i < NB; ++i) occ[i] = false;
                    for (int i = 0; i < total_atoms; ++i) {
                        RealType u = G[0]*all_cart[i*3+0] + G[1]*all_cart[i*3+1]
                                   + G[2]*all_cart[i*3+2];
                        u -= Kokkos::floor(u);
                        int bn = int(u * NB);
                        if (bn < 0) bn = 0; else if (bn >= NB) bn = NB - 1;
                        occ[bn] = true;
                    }
                    int first = -1;
                    for (int i = 0; i < NB; ++i) if (occ[i]) { first = i; break; }
                    if (first < 0) continue;                 // nothing placed
                    int run = 0, best = 0;
                    for (int n = 0; n < NB; ++n) {
                        const int i = (first + n) % NB;
                        if (occ[i]) { if (run > best) best = run; run = 0; }
                        else ++run;
                    }
                    if (run > best) best = run;
                    const RealType gap = RealType(best + 1) / RealType(NB) * d;
                    if (gap > worst) worst = gap;
                }
                return worst;
            };

            auto detect_at = [&](RealType la, RealType lb, RealType lc) -> bool {
                RealType cell[3][3], cell_inv[3][3];
                build_at(la, lb, lc, cell, cell_inv);
                RealType mcx[MAX_MOLS_CC], mcy[MAX_MOLS_CC], mcz[MAX_MOLS_CC];
                for (int mid = 0; mid < total_mols; ++mid) {
                    int base = mol_first_atom[mid], na = mol_n_atoms[mid];
                    RealType cx = 0, cy = 0, cz = 0;
                    for (int s = 0; s < na; ++s) {
                        cx += all_cart[(base+s)*3+0];
                        cy += all_cart[(base+s)*3+1];
                        cz += all_cart[(base+s)*3+2];
                    }
                    RealType inv = RealType(1) / RealType(na);
                    mcx[mid]=cx*inv; mcy[mid]=cy*inv; mcz[mid]=cz*inv;
                }
                for (int dx = -2; dx <= 2; ++dx)
                for (int dy = -2; dy <= 2; ++dy)
                for (int dz = -2; dz <= 2; ++dz) {
                    RealType shift[3] = {0,0,0};
                    for (int k = 0; k < 3; ++k)
                        shift[k] = dx*cell[k][0] + dy*cell[k][1] + dz*cell[k][2];
                    const bool primary = (dx==0&&dy==0&&dz==0);
                    // Only each independent molecule's identity image needs
                    // testing against all others: by symmetry the orbit follows.
                    for (int cm = 0; cm < n_indep_k; ++cm) {
                        const int mi = cm * n_symops;
                        const RealType Ri = mol_R_local[cm];
                        for (int mj = 0; mj < total_mols; ++mj) {
                            if (mi == mj && primary) continue;
                            const RealType Rj = mol_R_local[mol_indep_id[mj]];
                            const RealType crit = Ri + Rj + coll_buf;
                            const RealType ddx = mcx[mi]-(mcx[mj]+shift[0]);
                            const RealType ddy = mcy[mi]-(mcy[mj]+shift[1]);
                            const RealType ddz = mcz[mi]-(mcz[mj]+shift[2]);
                            if (ddx*ddx+ddy*ddy+ddz*ddz > crit*crit) continue;
                            const int bi = mol_first_atom[mi], bj = mol_first_atom[mj];
                            const int nai = mol_n_atoms[mi], naj = mol_n_atoms[mj];
                            for (int si = 0; si < nai; ++si) {
                                const RealType covi = cr(atom_global_site[bi+si]);
                                for (int sj = 0; sj < naj; ++sj) {
                                    RealType drx = all_cart[(bi+si)*3+0]-(all_cart[(bj+sj)*3+0]+shift[0]);
                                    RealType dry = all_cart[(bi+si)*3+1]-(all_cart[(bj+sj)*3+1]+shift[1]);
                                    RealType drz = all_cart[(bi+si)*3+2]-(all_cart[(bj+sj)*3+2]+shift[2]);
                                    RealType r2 = drx*drx+dry*dry+drz*drz;
                                    RealType sc = covi + cr(atom_global_site[bj+sj]) + coll_buf;
                                    if (r2 < sc*sc) return true;
                                }
                            }
                        }
                    }
                }
                return false;
            };

            // Clash resolution, nudge then scale: find the deepest clashing pair,
            // compute the hull MTV between their atom spheres and try to separate
            // them by moving the asymmetric-unit COM; when symmetry cannot realise
            // that translation, expand the shortest axis. The nudge budget is
            // bounded so termination rests on the monotone expansion.
            int nudge_left = nudge_budget_k;

            bool success = false;
            for (int iter = 0; iter < 60; ++iter) {
                RealType vol = a * b * c * sqrt_x;
                if (vol > max_vol) break;

                RealType cell[3][3], cell_inv[3][3];
                gpu_build_cell(a, b, c, ca, cb, cg, sg_val, sqrt_x, cell, cell_inv);

                // Build per-component supercell fragments into all_cart with the
                // accumulated COM nudge.
                RealType* mol0_frac_buf = &s_m0f(idx, 0, 0);
                int write_off = 0;
                for (int m = 0; m < n_indep_k; ++m) {
                    int ns_m = mns(m);
                    RealType* rot_m = &rotated[rot_site_offsets[m] * 3];
                    gpu_build_supercell(rot_m, ns_m, trans_m[m], cell, cell_inv,
                                        sym_rot_local, sym_trans_local, n_symops,
                                        &all_cart[write_off * 3], mol0_frac_buf,
                                        &nudge_comp[m][0]);
                    write_off += n_symops * ns_m;
                }

                // Compute per-molecule centroids from all_cart (Angstrom).
                RealType mol_cx[MAX_MOLS_CC], mol_cy[MAX_MOLS_CC], mol_cz[MAX_MOLS_CC];
                for (int mid = 0; mid < total_mols; ++mid) {
                    int base = mol_first_atom[mid];
                    int na   = mol_n_atoms[mid];
                    RealType cx = 0, cy = 0, cz = 0;
                    for (int s = 0; s < na; ++s) {
                        cx += all_cart[(base + s)*3 + 0];
                        cy += all_cart[(base + s)*3 + 1];
                        cz += all_cart[(base + s)*3 + 2];
                    }
                    RealType inv_n = RealType(1) / RealType(na);
                    mol_cx[mid] = cx * inv_n;
                    mol_cy[mid] = cy * inv_n;
                    mol_cz[mid] = cz * inv_n;
                }

                // Collision scan: bounding-sphere filter, then atom pairs. Track
                // the deepest clashing pair (no early exit) for the nudge.
                bool coll = false;
                RealType best_pen = 0;
                int best_mi = 0, best_mj = 0;
                RealType best_shift[3] = {0, 0, 0};
                for (int dx = -2; dx <= 2; ++dx)
                for (int dy = -2; dy <= 2; ++dy)
                for (int dz = -2; dz <= 2; ++dz) {
                    RealType shift[3] = {0, 0, 0};
                    for (int k = 0; k < 3; ++k)
                        shift[k] = dx * cell[k][0] + dy * cell[k][1] + dz * cell[k][2];
                    const bool primary_image = (dx == 0 && dy == 0 && dz == 0);

                    for (int mi = 0; mi < total_mols; ++mi) {
                        const RealType R_i = mol_R_local[mol_indep_id[mi]];
                        for (int mj = mi; mj < total_mols; ++mj) {
                            // Skip same molecule in primary cell.
                            if (mi == mj && primary_image) continue;

                            // Level 1: bounding-sphere filter.
                            const RealType R_j = mol_R_local[mol_indep_id[mj]];
                            const RealType crit = R_i + R_j + coll_buf;
                            const RealType crit_sq = crit * crit;
                            const RealType ddx = mol_cx[mi] - (mol_cx[mj] + shift[0]);
                            const RealType ddy = mol_cy[mi] - (mol_cy[mj] + shift[1]);
                            const RealType ddz = mol_cz[mi] - (mol_cz[mj] + shift[2]);
                            const RealType d2_c = ddx*ddx + ddy*ddy + ddz*ddz;
                            if (d2_c > crit_sq) continue;

                            // Level 2: detailed atom-pair check.
                            const int base_i = mol_first_atom[mi];
                            const int base_j = mol_first_atom[mj];
                            const int na_i   = mol_n_atoms[mi];
                            const int na_j   = mol_n_atoms[mj];
                            for (int si = 0; si < na_i; ++si) {
                                const int gi = atom_global_site[base_i + si];
                                const RealType cov_i = cr(gi);
                                for (int sj = 0; sj < na_j; ++sj) {
                                    const int gj = atom_global_site[base_j + sj];
                                    RealType drx = all_cart[(base_i + si)*3 + 0]
                                                 - (all_cart[(base_j + sj)*3 + 0] + shift[0]);
                                    RealType dry = all_cart[(base_i + si)*3 + 1]
                                                 - (all_cart[(base_j + sj)*3 + 1] + shift[1]);
                                    RealType drz = all_cart[(base_i + si)*3 + 2]
                                                 - (all_cart[(base_j + sj)*3 + 2] + shift[2]);
                                    RealType r2 = drx*drx + dry*dry + drz*drz;
                                    RealType sum_cov = cov_i + cr(gj) + coll_buf;
                                    if (r2 < sum_cov * sum_cov) {
                                        coll = true;
                                        const RealType pen = sum_cov - Kokkos::sqrt(r2);
                                        if (pen > best_pen) {
                                            best_pen = pen;
                                            best_mi = mi; best_mj = mj;
                                            best_shift[0] = shift[0];
                                            best_shift[1] = shift[1];
                                            best_shift[2] = shift[2];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (!coll) { success = true; break; }

                // --- NUDGE: separate the deepest pair via a COM translation ---
                bool did_nudge = false;
                const int na_i = mol_n_atoms[best_mi];
                const int na_j = mol_n_atoms[best_mj];
                if (nudge_left > 0 && na_i <= mtv_cap_k && na_j <= mtv_cap_k) {
                    const int base_i = mol_first_atom[best_mi];
                    const int base_j = mol_first_atom[best_mj];
                    RealType* rA = &s_mtv_rA(idx, 0);
                    RealType* cB = &s_mtv_cB(idx, 0, 0);
                    RealType* rB = &s_mtv_rB(idx, 0);
                    for (int si = 0; si < na_i; ++si)
                        rA[si] = cr(atom_global_site[base_i + si]);
                    for (int sj = 0; sj < na_j; ++sj) {
                        cB[3*sj+0] = all_cart[(base_j + sj)*3 + 0] + best_shift[0];
                        cB[3*sj+1] = all_cart[(base_j + sj)*3 + 1] + best_shift[1];
                        cB[3*sj+2] = all_cart[(base_j + sj)*3 + 2] + best_shift[2];
                        rB[sj] = cr(atom_global_site[base_j + sj]);
                    }
                    const RealType* cA = &all_cart[base_i * 3];
                    MTVResult mr = sphere_set_mtv(cA, rA, na_i, cB, rB, na_j, 24);
                    if (mr.depth > 0) {
                        // Requested relative displacement of B(=mj) w.r.t A(=mi).
                        const RealType target = mr.depth + coll_buf + RealType(1e-3);
                        RealType rhs[3] = {target*mr.dir[0], target*mr.dir[1],
                                           target*mr.dir[2]};
                        const int op_a = best_mi % n_symops;
                        const int op_b = best_mj % n_symops;
                        const int comp_i = mol_indep_id[best_mi];
                        const int comp_j = mol_indep_id[best_mj];
                        RealType Sa[3][3], Sb[3][3], Ra[3][3], Rb[3][3];
                        for (int r = 0; r < 3; ++r)
                            for (int q = 0; q < 3; ++q) {
                                Sa[r][q] = sym_rot_local[op_a*9 + r*3 + q];
                                Sb[r][q] = sym_rot_local[op_b*9 + r*3 + q];
                            }
                        frac_to_cart_symop(cell, cell_inv, Sa, Ra);
                        frac_to_cart_symop(cell, cell_inv, Sb, Rb);
                        // d(pos_mj - pos_mi)/d(delta) = M.  Same component: both
                        // images move -> M = Rb - Ra.  Different components: move
                        // only comp_j -> M = Rb (delta = Rb^T rhs).
                        RealType M[3][3];
                        int target_comp;
                        if (comp_i == comp_j) {
                            for (int r = 0; r < 3; ++r)
                                for (int q = 0; q < 3; ++q)
                                    M[r][q] = Rb[r][q] - Ra[r][q];
                            target_comp = comp_i;
                        } else {
                            for (int r = 0; r < 3; ++r)
                                for (int q = 0; q < 3; ++q)
                                    M[r][q] = Rb[r][q];
                            target_comp = comp_j;
                        }
                        RealType fro = 0;
                        for (int r = 0; r < 3; ++r)
                            for (int q = 0; q < 3; ++q) fro += M[r][q]*M[r][q];
                        if (fro > RealType(1e-10)) {
                            // delta = (M^T M + lambda I)^{-1} M^T rhs
                            RealType MtM[3][3];
                            for (int r = 0; r < 3; ++r)
                                for (int q = 0; q < 3; ++q) {
                                    RealType s = 0;
                                    for (int k = 0; k < 3; ++k) s += M[k][r]*M[k][q];
                                    MtM[r][q] = s;
                                }
                            MtM[0][0] += RealType(1e-6);
                            MtM[1][1] += RealType(1e-6);
                            MtM[2][2] += RealType(1e-6);
                            // Unreachable: the 1e-6 diagonal shift makes MtM invertible.
                            RealType MtMinv[3][3] = {};
                            (void)mat3_inv(MtM, MtMinv);
                            RealType Mtr[3];
                            for (int r = 0; r < 3; ++r) {
                                RealType s = 0;
                                for (int k = 0; k < 3; ++k) s += M[k][r]*rhs[k];
                                Mtr[r] = s;
                            }
                            RealType delta[3];
                            mat3_vec(MtMinv, Mtr, delta);
                            // Reject the nudge if too much of the requested
                            // separation lies in M's null space (lattice
                            // translates / shared axis) -> let scale handle it.
                            RealType Md[3];
                            mat3_vec(M, delta, Md);
                            RealType res2 = 0, rhs2 = 0;
                            for (int k = 0; k < 3; ++k) {
                                RealType e = rhs[k] - Md[k];
                                res2 += e*e; rhs2 += rhs[k]*rhs[k];
                            }
                            if (res2 <= RealType(0.34*0.34) * rhs2) {
                                nudge_comp[target_comp][0] += delta[0];
                                nudge_comp[target_comp][1] += delta[1];
                                nudge_comp[target_comp][2] += delta[2];
                                nudge_left--;
                                did_nudge = true;
                            }
                        }
                    }
                }

                // --- SCALE: expand shortest length by 1 Å when the nudge can't
                //     separate the pair (or the nudge budget is spent). ---
                if (!did_nudge) {
                    RealType min_len = Kokkos::fmin(Kokkos::fmin(a, b), c);
                    if (a == min_len) a += 1; else if (b == min_len) b += 1; else c += 1;
                    if (lattice == TETRAGONAL || lattice == HEXAGONAL) b = a;
                    else if (lattice == RHOMBOHEDRAL || lattice == CUBIC) b = c = a;
                }
            }

            if (!success) return;

            // --- CONTRACT (expand-then-contract densification) ---
            // First close vacuum layers: contraction alone cannot, since it scales
            // slab and void together. Shift the asymmetric unit relative to the
            // symmetry elements instead, scoring 16 shifts and collision-testing
            // only the winner. Then shrink lengths by coordinate descent until
            // collision, for lattices whose three lengths vary independently.
            if (do_contract_k && max_gap_at(a, b, c) > kGenVacuumTriggerA) {
                // Shifts are along fractional axes, but the objective is the
                // worst gap over all directions.
                for (int ax = 0; ax < 3; ++ax) {
                    RealType best_shift = 0, best_gap = max_gap_at(a, b, c);
                    for (int t = 1; t < 16; ++t) {
                        const RealType d = RealType(t) / RealType(16);
                        for (int m = 0; m < n_indep_k; ++m) trans_m[m][ax] += d;
                        const RealType g = max_gap_at(a, b, c);
                        for (int m = 0; m < n_indep_k; ++m) trans_m[m][ax] -= d;
                        if (g < best_gap) { best_gap = g; best_shift = d; }
                    }
                    if (best_shift > 0) {
                        for (int m = 0; m < n_indep_k; ++m) trans_m[m][ax] += best_shift;
                        // A shift that creates an overlap is no improvement.
                        if (detect_at(a, b, c))
                            for (int m = 0; m < n_indep_k; ++m) trans_m[m][ax] -= best_shift;
                    }
                }
            }

            if (do_contract_k &&
                (lattice == TRICLINIC || lattice == MONOCLINIC || lattice == ORTHORHOMBIC)) {
                for (int stage = 1; stage <= 3; ++stage) {
                    // Coarse-to-fine: shrink in big steps, halving on overshoot
                    // down to 1 Å.
                    RealType step = RealType(4);
                    while (step >= RealType(1)) {
                        RealType sorted[3] = {a, b, c};
                        for (int p = 0; p < 3; ++p)
                            for (int q = p + 1; q < 3; ++q)
                                if (sorted[q] > sorted[p]) {
                                    RealType t = sorted[p]; sorted[p] = sorted[q]; sorted[q] = t;
                                }
                        const RealType target = sorted[stage - 1];
                        const int axis = (a == target) ? 0 : ((b == target) ? 1 : 2);
                        RealType na = a, nb = b, nc = c;
                        if (axis == 0) na -= step; else if (axis == 1) nb -= step; else nc -= step;
                        if (na <= 1 || nb <= 1 || nc <= 1 || detect_at(na, nb, nc)) {
                            step *= RealType(0.5);   // overshoot: refine
                        } else {
                            a = na; b = nb; c = nc;  // accept; keep shrinking
                        }
                    }
                }
            }

            cp_out(idx, 0) = a * ANGSTROM_TO_BOHR;
            cp_out(idx, 1) = b * ANGSTROM_TO_BOHR;
            cp_out(idx, 2) = c * ANGSTROM_TO_BOHR;
            cp_out(idx, 3) = alpha * 180.0 / M_PI;
            cp_out(idx, 4) = beta * 180.0 / M_PI;
            cp_out(idx, 5) = gamma * 180.0 / M_PI;
            // Fold the Cartesian COM nudge into the fractional translation for
            // the final cell.
            RealType cellf[3][3], cellf_inv[3][3];
            gpu_build_cell(a, b, c, ca, cb, cg, sg_val, sqrt_x, cellf, cellf_inv);
            for (int m = 0; m < n_indep_k; ++m) {
                RealType df[3] = {0, 0, 0};
                for (int d = 0; d < 3; ++d)
                    for (int k = 0; k < 3; ++k)
                        df[d] += cellf_inv[d][k] * nudge_comp[m][k];
                tr_out(idx, m, 0) = trans_m[m][0] + df[0];
                tr_out(idx, m, 1) = trans_m[m][1] + df[1];
                tr_out(idx, m, 2) = trans_m[m][2] + df[2];
                rv_out(idx, m, 0) = aa_m[m][0];
                rv_out(idx, m, 1) = aa_m[m][1];
                rv_out(idx, m, 2) = aa_m[m][2];
            }
            vf_out(idx) = 1;
        });

    Kokkos::fence("batch_generate_cc");

    // Counted on device.
    int n_valid = 0;
    Kokkos::parallel_reduce("count_valid",
        Kokkos::RangePolicy<ExecSpace>(0, n_seeds),
        KOKKOS_LAMBDA(int i, int& sum) { sum += valid_flags_out(i); },
        n_valid);
    return n_valid;
}

int generate_and_compact_on_device(
    int space_group, int seed_start, int n_try,
    const View2D<RealType>& body_positions_ang,
    const View1D<RealType>& cov_radii,
    const View1D<RealType>& vdw_radii,
    int n_sites, double tvp, double collision_buffer,
    CollisionMethod collision_method,
    View2D<RealType> cell_params_out,
    View3D<RealType> translations_out,
    View3D<RealType> rot_vectors_out,
    View1D<int> seeds_out,
    int max_output,
    int n_indep,
    const View1D<int>& mol_n_sites,
    const View1D<int>& mol_site_offsets,
    const View1D<RealType>& mol_bounding_radius,
    int rank, int nranks,
    int nudge_budget, bool contract,
    int* raw_valid_out,
    SamplingPrior* prior_out) {

    using ExecSpace = Kokkos::DefaultExecutionSpace;

    // Step 1: Generate into temporary scattered Views
    View2D<RealType> gen_cp("gen_cp", n_try, 6);
    View3D<RealType> gen_tr("gen_tr", n_try, n_indep, 3);
    View3D<RealType> gen_rv("gen_rv", n_try, n_indep, 3);
    View1D<int> gen_vf("gen_vf", n_try);

    int raw_valid = batch_generate_structures(
        space_group, seed_start, n_try,
        body_positions_ang, cov_radii, vdw_radii,
        n_sites, tvp, collision_buffer,
        gen_cp, gen_tr, gen_rv, gen_vf,
        collision_method,
        n_indep, mol_n_sites, mol_site_offsets,
        mol_bounding_radius,
        rank, nranks, nudge_budget, contract, prior_out);
    // batch_generate_structures already fences

    // The uncapped valid count, so the caller can size the next over-generation.
    if (raw_valid_out) *raw_valid_out = raw_valid;

    if (raw_valid == 0) return 0;

    // Step 2: Exclusive prefix sum on valid flags
    View1D<int> prefix("prefix", n_try);
    Kokkos::parallel_scan("compact_scan",
        Kokkos::RangePolicy<ExecSpace>(0, n_try),
        KOKKOS_LAMBDA(int i, int& partial_sum, bool is_final) {
            if (is_final) prefix(i) = partial_sum;
            partial_sum += gen_vf(i);
        });
    ExecSpace().fence();

    // Step 3: batch_generate_structures already counted these and fenced.
    const int n_output = std::min(raw_valid, max_output);
    if (n_output == 0) return 0;

    // Step 4: Scatter-gather valid entries into output Views
    auto cp_out = cell_params_out;
    auto tr_out = translations_out;
    auto rv_out = rot_vectors_out;
    auto sd_out = seeds_out;

    int n_indep_k = n_indep;
    (void)rank; (void)nranks;   // seed is now a rank-independent global index
    Kokkos::parallel_for("compact_gather",
        Kokkos::RangePolicy<ExecSpace>(0, n_try),
        KOKKOS_LAMBDA(int i) {
            if (!gen_vf(i)) return;
            int dst = prefix(i);
            if (dst >= n_output) return;
            for (int j = 0; j < 6; ++j) cp_out(dst, j) = gen_cp(i, j);
            for (int m = 0; m < n_indep_k; ++m) {
                for (int d = 0; d < 3; ++d) {
                    tr_out(dst, m, d) = gen_tr(i, m, d);
                    rv_out(dst, m, d) = gen_rv(i, m, d);
                }
            }
            sd_out(dst) = seed_start + i;   // matches the generator's contiguous seed
        });
    ExecSpace().fence();

    return n_output;
}

int QuasiRandomGenerator::generate(const GeneratorRequest& req, DofBatch& out,
                                   int* raw_valid, SamplingPrior* prior_out) const {
    // Seeds are claimed per space group by the caller, so rank/nranks are fixed.
    return generate_and_compact_on_device(
        req.space_group, req.seed_start, req.n_try,
        body_positions_ang, cov_radii, vdw_radii,
        n_sites, tvp, collision_buffer, collision_method,
        out.cell_params, out.translations, out.rot_vectors, out.seeds,
        out.capacity,
        out.n_indep, mol_n_sites, mol_site_offsets, mol_bounding_radius,
        0, 1, nudge_budget, contract, raw_valid, prior_out);
}

}  // namespace klasp
