#pragma once

#include "klasp/multipole.hpp"   // nhermsum

#include <utility>
#include <vector>
// Shared test utilities.

#include <Kokkos_Core.hpp>
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/sgdata.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace klasp::testing {

// ---------------------------------------------------------------------------
// Comparing an analytic derivative against a finite difference
// ---------------------------------------------------------------------------
//
// A component passes if it is within kFdRelTol relatively or kFdAbsTol
// absolutely; the absolute floor handles components that are zero by symmetry.
// kFdAbsTol = 1e-9 sits well above the difference quotient's noise (~1e-14) and
// below the optimiser's 1e-8 convergence threshold.
inline constexpr double kFdRelTol = 1e-6;
inline constexpr double kFdAbsTol = 1e-9;

// True when `analytic` and `fd` agree to within either tolerance.
inline bool fd_agrees(double analytic, double fd,
                      double rtol = kFdRelTol, double atol = kFdAbsTol) {
    const double diff = std::abs(analytic - fd);
    if (diff <= atol) return true;
    const double scale = std::max(std::abs(analytic), std::abs(fd));
    return scale > 0.0 && diff <= rtol * scale;
}


// Two charged sites per molecule in a P-1 cell: enough for the Buckingham and
// electrostatic terms to both contribute a non-trivial cell gradient.
struct Fixture {
    MoleculeData mol;
    Workspace ws;
    BatchWorkspace bws;
    BatchOptimizeSettings settings;
    int n_dof = 0;
};

// P2_1/c: four operations on one independent molecule, so the asymmetric-unit
// prefactor n_symops / (2 n_indep) is 2 rather than 1 as in the P-1 fixture.
// Monoclinic, so alpha = gamma = 90 is required of the caller.
inline Fixture make_fixture_p21c(double a_A, double b_A, double c_A, double beta);
inline void finish_fixture(Fixture& f);

inline Fixture make_fixture(double a_A, double b_A, double c_A,
                     double alpha, double beta, double gamma) {
    Fixture f;
    const int n_indep = 1, n_sites = 2, n_symops = 2;
    f.mol = allocate_molecule_data(n_indep, n_sites, n_symops, /*max_multipole_rank=*/1);
    f.ws  = allocate_workspace(8, 30000, 50000, n_indep * n_sites);

    f.mol.cell_params[0] = a_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[1] = b_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[2] = c_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[3] = alpha;
    f.mol.cell_params[4] = beta;
    f.mol.cell_params[5] = gamma;

    auto h_bp = Kokkos::create_mirror_view(f.mol.body_positions);
    auto h_bc = Kokkos::create_mirror_view(f.mol.body_charges);
    auto h_bt_types = Kokkos::create_mirror_view(f.mol.body_types);
    h_bp(0, 0) = -0.6 * ANGSTROM_TO_BOHR; h_bp(0, 1) = 0.0; h_bp(0, 2) = 0.0;
    h_bp(1, 0) =  0.6 * ANGSTROM_TO_BOHR; h_bp(1, 1) = 0.1 * ANGSTROM_TO_BOHR; h_bp(1, 2) = 0.0;
    h_bc(0) =  0.35; h_bc(1) = -0.35;
    h_bt_types(0) = 0; h_bt_types(1) = 0;
    Kokkos::deep_copy(f.mol.body_positions, h_bp);
    Kokkos::deep_copy(f.mol.body_charges, h_bc);
    Kokkos::deep_copy(f.mol.body_types, h_bt_types);

    // Body-frame multipoles: component 0 is the charge, 1..3 the dipole. Dipoles
    // are off-axis so every strain component sees a torque.
    auto h_mp = Kokkos::create_mirror_view(f.mol.body_multipoles);
    Kokkos::deep_copy(h_mp, RealType(0));
    h_mp(0, 0) =  0.35; h_mp(0, 1) =  0.12; h_mp(0, 2) = -0.08; h_mp(0, 3) = 0.05;
    h_mp(1, 0) = -0.35; h_mp(1, 1) = -0.09; h_mp(1, 2) =  0.11; h_mp(1, 3) = 0.07;
    Kokkos::deep_copy(f.mol.body_multipoles, h_mp);

    // P-1: identity + inversion.
    auto h_sr = Kokkos::create_mirror_view(f.mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(f.mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(f.mol.sym_indep_index);
    for (int c = 0; c < 9; ++c) { h_sr(0, c) = 0; h_sr(1, c) = 0; }
    h_sr(0,0) =  1; h_sr(0,4) =  1; h_sr(0,8) =  1;
    h_sr(1,0) = -1; h_sr(1,4) = -1; h_sr(1,8) = -1;
    for (int d = 0; d < 3; ++d) { h_st(0, d) = 0; h_st(1, d) = 0; }
    h_si(0) = 0; h_si(1) = 0;
    Kokkos::deep_copy(f.mol.sym_rotations, h_sr);
    Kokkos::deep_copy(f.mol.sym_translations, h_st);
    Kokkos::deep_copy(f.mol.sym_indep_index, h_si);

    f.mol.n_indep = n_indep;
    f.mol.n_symops = n_symops;
    f.mol.n_sites_per_mol = n_sites;
    f.ws.n_types = 1;
    f.ws.buck_cutoff = 8.0 * ANGSTROM_TO_BOHR;

    f.bws = allocate_batch_workspace(1, f.ws, f.mol);
    finish_fixture(f);
    return f;
}

// Everything after the batch workspace exists: potentials, an off-centre
// placement, cell setup and the six strain DOFs.
inline void finish_fixture(Fixture& f) {
    const int n_indep = f.mol.n_indep;

    // Buckingham repulsion + dispersion, roughly W99 magnitudes for a light
    // atom pair (Hartree, Bohr): the minimum sits near 3.5 A.
    {
        auto hA = Kokkos::create_mirror_view(f.bws.tables.buck_A);
        auto hB = Kokkos::create_mirror_view(f.bws.tables.buck_B);
        auto hC = Kokkos::create_mirror_view(f.bws.tables.buck_C);
        auto hR = Kokkos::create_mirror_view(f.bws.tables.buck_Rmin);
        hA(0,0) = 60.0; hB(0,0) = 1.9; hC(0,0) = 40.0; hR(0,0) = 0;
        Kokkos::deep_copy(f.bws.tables.buck_A, hA);
        Kokkos::deep_copy(f.bws.tables.buck_B, hB);
        Kokkos::deep_copy(f.bws.tables.buck_C, hC);
        Kokkos::deep_copy(f.bws.tables.buck_Rmin, hR);
    }

    // Off-centre molecule so no gradient component is zero by symmetry.
    auto h_bt = Kokkos::create_mirror_view(f.bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(f.bws.dofs.rot_vectors);
    h_bt(0,0,0) = 0.11; h_bt(0,0,1) = 0.07; h_bt(0,0,2) = 0.13;
    h_br(0,0,0) = 0.21; h_br(0,0,1) = 0.14; h_br(0,0,2) = 0.09;
    Kokkos::deep_copy(f.bws.dofs.translations, h_bt);
    Kokkos::deep_copy(f.bws.dofs.rot_vectors, h_br);

    auto h_cp = Kokkos::create_mirror_view(f.bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = f.mol.cell_params[i];
    Kokkos::deep_copy(f.bws.geom.cell_params, h_cp);
    batch_setup_cells(f.bws.tables, f.bws.geom);

    // All six strain components active, reference cell = current cell.
    // allocate_batch_workspace leaves these zero-extent until cell DOFs are used.
    f.bws.dofs.cell_ref    = decltype(f.bws.dofs.cell_ref)("cell_ref", 1, 3, 3);
    f.bws.dofs.strain      = decltype(f.bws.dofs.strain)("strain", 1, 6);
    f.bws.dofs.d_g_strain  = decltype(f.bws.dofs.d_g_strain)("d_g_strain", 1, 6);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));   // identity basis, 6 DOFs
    Kokkos::deep_copy(f.bws.dofs.cell_ref, f.bws.geom.cell);
    Kokkos::deep_copy(f.bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(f.bws.dofs.d_g_strain, RealType(0));

    f.n_dof = n_indep * 6 + 6;
    f.settings.n_iterations = 1;
    f.settings.max_step = 0.05;
    f.settings.lbfgs_memory = 5;
    f.settings.model.taper_on = 0;
    f.settings.model.taper_off = 0;
    f.settings.model.wolf = true;      // Wolf DSF: no reciprocal sum, order-independent
    f.settings.energy_only = false;
}

// Energy at the current strain, rebuilding every cell-derived quantity the way
// the optimiser does between steps.
template <int RANK>
RealType energy_at_strain(Fixture& f, const RealType eps[6]) {
    auto h_s = Kokkos::create_mirror_view(f.bws.dofs.strain);
    for (int i = 0; i < 6; ++i) h_s(0, i) = eps[i];
    Kokkos::deep_copy(f.bws.dofs.strain, h_s);
    batch_update_cell_from_strain(make_force_field(f.bws.tables, f.settings.model),
                                  f.bws.geom, f.bws.dofs);
    batch_expand_asymmetric_unit<RANK>(f.bws.tables, f.bws.geom, f.bws.dofs);
    Kokkos::fence();
    batch_energy_gradient_asymunit<RANK>(f.bws.tables, f.bws.geom, f.bws.dofs, f.settings, 0);
    Kokkos::fence();
    auto hE = Kokkos::create_mirror_view(f.bws.geom.energies);
    Kokkos::deep_copy(hE, f.bws.geom.energies);
    return hE(0);
}


// A host copy that is a real copy. create_mirror_view (and _and_copy) return the
// same view on a host build; create_mirror always allocates.
template <typename V>
auto host_snapshot(const V& v) {
    auto h = Kokkos::create_mirror(Kokkos::HostSpace(), v);
    Kokkos::deep_copy(h, v);
    return h;
}

// Central-difference check of the rigid-body gradient the optimiser consumes:
// every translation and rotation DOF of every independent molecule, against
// d_gt/d_gr from the same kernel. Returns the worst relative disagreement.
template <int RANK>
double worst_dof_gradient_mismatch(BatchWorkspace& bws,
                                   const BatchOptimizeSettings& settings,
                                   int b, double h, bool verbose = true) {
    auto energy = [&]() -> double {
        batch_expand_asymmetric_unit<RANK>(bws.tables, bws.geom, bws.dofs);
        Kokkos::fence();
        batch_energy_gradient_asymunit<RANK>(bws.tables, bws.geom, bws.dofs, settings, 0);
        Kokkos::fence();
        return host_snapshot(bws.geom.energies)(b);
    };

    const auto t0 = host_snapshot(bws.dofs.translations);
    const auto r0 = host_snapshot(bws.dofs.rot_vectors);
    energy();
    const auto g_t = host_snapshot(bws.dofs.d_gt);
    const auto g_r = host_snapshot(bws.dofs.d_gr);

    auto restore = [&]() {
        Kokkos::deep_copy(bws.dofs.translations, t0);
        Kokkos::deep_copy(bws.dofs.rot_vectors, r0);
    };
    // Displace one DOF of one molecule, leaving everything else at the original.
    auto displace = [&](bool rotation, int k, int d, double delta) {
        restore();
        auto& view = rotation ? bws.dofs.rot_vectors : bws.dofs.translations;
        auto hv = host_snapshot(view);
        hv(b, k, d) += static_cast<RealType>(delta);
        Kokkos::deep_copy(view, hv);
    };

    double worst = 0.0;
    for (int k = 0; k < bws.tables.n_indep; ++k)
        for (int pass = 0; pass < 2; ++pass) {
            const bool rotation = (pass == 1);
            for (int d = 0; d < 3; ++d) {
                displace(rotation, k, d,  h); const double Ep = energy();
                displace(rotation, k, d, -h); const double Em = energy();
                const double fd = (Ep - Em) / (2.0 * h);
                const double an = rotation ? g_r(b, k, d) : g_t(b, k, d);
                // Components that are zero by symmetry carry only round-off on both
                // sides, so they are skipped.
                const double mag = std::max(std::abs(an), std::abs(fd));
                const double rel = mag < 1e-9 ? 0.0 : std::abs(an - fd) / mag;
                if (verbose)
                    std::printf("    %s%d_%d: analytic=% .8e  fd=% .8e  rel=%.2e\n",
                                rotation ? "rot" : "trans", k, d, an, fd, rel);
                worst = std::max(worst, rel);
            }
        }
    restore();
    return worst;
}

// P-1 like make_fixture, but with n_sites atoms per molecule, enough
// contributions per force component for accumulation order to matter.
inline Fixture make_fixture_wide(int n_sites, double a_A, double b_A, double c_A,
                                 double alpha, double beta, double gamma) {
    Fixture f;
    const int n_indep = 1, n_symops = 2;
    f.mol = allocate_molecule_data(n_indep, n_sites, n_symops, /*max_multipole_rank=*/1);
    f.ws  = allocate_workspace(8, 30000, 50000, n_indep * n_sites);

    f.mol.cell_params[0] = a_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[1] = b_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[2] = c_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[3] = alpha;
    f.mol.cell_params[4] = beta;
    f.mol.cell_params[5] = gamma;

    // Sites on a loose helix so no two are equivalent and nothing cancels by
    // symmetry, with alternating charges so the molecule stays neutral.
    auto h_bp = Kokkos::create_mirror_view(f.mol.body_positions);
    auto h_bc = Kokkos::create_mirror_view(f.mol.body_charges);
    auto h_bt = Kokkos::create_mirror_view(f.mol.body_types);
    auto h_mp = Kokkos::create_mirror_view(f.mol.body_multipoles);
    Kokkos::deep_copy(h_mp, RealType(0));
    for (int i = 0; i < n_sites; ++i) {
        const double t = 0.7 * i;
        h_bp(i, 0) = 1.35 * std::cos(t) * ANGSTROM_TO_BOHR;
        h_bp(i, 1) = 1.35 * std::sin(t) * ANGSTROM_TO_BOHR;
        h_bp(i, 2) = (0.45 * i - 0.225 * (n_sites - 1)) * ANGSTROM_TO_BOHR;
        const double q = (i % 2 ? -0.31 : 0.31) + 0.004 * i;
        h_bc(i) = q;
        h_bt(i) = 0;
        h_mp(i, 0) = q;
        h_mp(i, 1) =  0.11 - 0.003 * i;
        h_mp(i, 2) = -0.07 + 0.005 * i;
        h_mp(i, 3) =  0.06 + 0.002 * i;
    }
    // Neutralise: a charged cell makes the Ewald sum meaningless.
    double qsum = 0;
    for (int i = 0; i < n_sites; ++i) qsum += h_bc(i);
    for (int i = 0; i < n_sites; ++i) {
        h_bc(i) -= qsum / n_sites;
        h_mp(i, 0) = h_bc(i);
    }
    Kokkos::deep_copy(f.mol.body_positions, h_bp);
    Kokkos::deep_copy(f.mol.body_charges, h_bc);
    Kokkos::deep_copy(f.mol.body_types, h_bt);
    Kokkos::deep_copy(f.mol.body_multipoles, h_mp);

    // P-1: identity + inversion, as make_fixture.
    auto h_sr = Kokkos::create_mirror_view(f.mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(f.mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(f.mol.sym_indep_index);
    for (int c = 0; c < 9; ++c) { h_sr(0, c) = 0; h_sr(1, c) = 0; }
    h_sr(0,0) =  1; h_sr(0,4) =  1; h_sr(0,8) =  1;
    h_sr(1,0) = -1; h_sr(1,4) = -1; h_sr(1,8) = -1;
    for (int d = 0; d < 3; ++d) { h_st(0, d) = 0; h_st(1, d) = 0; }
    h_si(0) = 0; h_si(1) = 0;
    Kokkos::deep_copy(f.mol.sym_rotations, h_sr);
    Kokkos::deep_copy(f.mol.sym_translations, h_st);
    Kokkos::deep_copy(f.mol.sym_indep_index, h_si);

    f.mol.n_indep = n_indep;
    f.mol.n_symops = n_symops;
    f.mol.n_sites_per_mol = n_sites;
    f.ws.n_types = 1;
    f.ws.buck_cutoff = 8.0 * ANGSTROM_TO_BOHR;

    f.bws = allocate_batch_workspace(1, f.ws, f.mol);
    finish_fixture(f);
    return f;
}

// Z' > 1: several independent molecules, each with symmetry copies, where a
// symmetry multiplicity error becomes visible.
//
// Layout follows io.cpp: operations interleaved molecule-minor, identity first,
// so sym_indep_index(oi * n_indep + m) == m and body data for molecule k site si
// lives at k * n_sites + si. `sym_map`, when given, replaces the interleaving:
// entry i is the independent molecule operation i copies, so molecules can have
// different numbers of copies (special positions).
inline Fixture make_fixture_zprime(int n_indep, int n_sites,
                                   double a_A, double b_A, double c_A, double beta,
                                   const std::vector<int>& sym_map = {},
                                   int max_rank = 1) {
    Fixture f;
    // P2_1/c point group: identity, two-fold, inversion, glide.
    const double ops[4][9] = {
        { 1, 0, 0,  0, 1, 0,  0, 0, 1},
        {-1, 0, 0,  0, 1, 0,  0, 0,-1},
        {-1, 0, 0,  0,-1, 0,  0, 0,-1},
        { 1, 0, 0,  0,-1, 0,  0, 0, 1},
    };
    const int n_ops = 4;
    const int n_symops = sym_map.empty() ? n_ops * n_indep
                                         : static_cast<int>(sym_map.size());

    f.mol = allocate_molecule_data(n_indep, n_sites, n_symops, max_rank);
    f.ws  = allocate_workspace(64, 30000, 50000, n_indep * n_sites, max_rank);

    f.mol.cell_params[0] = a_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[1] = b_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[2] = c_A * ANGSTROM_TO_BOHR;
    f.mol.cell_params[3] = 90.0;
    f.mol.cell_params[4] = beta;
    f.mol.cell_params[5] = 90.0;

    // Each molecule gets its own geometry.
    auto h_bp = Kokkos::create_mirror_view(f.mol.body_positions);
    auto h_bc = Kokkos::create_mirror_view(f.mol.body_charges);
    auto h_bt = Kokkos::create_mirror_view(f.mol.body_types);
    auto h_mp = Kokkos::create_mirror_view(f.mol.body_multipoles);
    Kokkos::deep_copy(h_mp, RealType(0));
    for (int k = 0; k < n_indep; ++k)
        for (int si = 0; si < n_sites; ++si) {
            const int idx = k * n_sites + si;
            const double t = 0.6 * si + 0.37 * k;
            h_bp(idx, 0) = (1.25 + 0.1 * k) * std::cos(t) * ANGSTROM_TO_BOHR;
            h_bp(idx, 1) = (1.25 + 0.1 * k) * std::sin(t) * ANGSTROM_TO_BOHR;
            h_bp(idx, 2) = (0.42 * si - 0.21 * (n_sites - 1)) * ANGSTROM_TO_BOHR;
            const double q = (si % 2 ? -0.29 : 0.29) + 0.01 * k;
            h_bc(idx) = q;
            h_bt(idx) = 0;
            h_mp(idx, 1) =  0.10 - 0.004 * si + 0.01 * k;
            h_mp(idx, 2) = -0.06 + 0.003 * si;
            h_mp(idx, 3) =  0.05 + 0.002 * si - 0.01 * k;
            // Ranks 2 and up, so the higher-order multipole path sees non-zeros.
            for (int c = 4; c < nhermsum(max_rank); ++c)
                h_mp(idx, c) = 0.03 * std::cos(0.9 * c + 0.4 * si + 0.2 * k);
        }
    // Neutralise each molecule separately: a charged cell makes Ewald meaningless.
    for (int k = 0; k < n_indep; ++k) {
        double qsum = 0;
        for (int si = 0; si < n_sites; ++si) qsum += h_bc(k * n_sites + si);
        for (int si = 0; si < n_sites; ++si) {
            const int idx = k * n_sites + si;
            h_bc(idx) -= qsum / n_sites;
            h_mp(idx, 0) = h_bc(idx);
        }
    }
    Kokkos::deep_copy(f.mol.body_positions, h_bp);
    Kokkos::deep_copy(f.mol.body_charges, h_bc);
    Kokkos::deep_copy(f.mol.body_types, h_bt);
    Kokkos::deep_copy(f.mol.body_multipoles, h_mp);

    auto h_sr = Kokkos::create_mirror_view(f.mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(f.mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(f.mol.sym_indep_index);
    for (int op = 0; op < n_symops; ++op) {
        const int m = sym_map.empty() ? (op % n_indep) : sym_map[op];
        const int oi = sym_map.empty() ? (op / n_indep) : (op / n_indep);
        for (int c = 0; c < 9; ++c) h_sr(op, c) = RealType(ops[oi % n_ops][c]);
        for (int d = 0; d < 3; ++d) h_st(op, d) = 0;
        h_si(op) = m;                     // molecule-minor, identity first
    }
    Kokkos::deep_copy(f.mol.sym_rotations, h_sr);
    Kokkos::deep_copy(f.mol.sym_translations, h_st);
    Kokkos::deep_copy(f.mol.sym_indep_index, h_si);

    f.mol.n_indep = n_indep;
    f.mol.n_symops = n_symops;
    f.mol.n_sites_per_mol = n_sites;
    f.ws.n_types = 1;
    f.ws.buck_cutoff = 8.0 * ANGSTROM_TO_BOHR;

    f.bws = allocate_batch_workspace(1, f.ws, f.mol);

    {
        auto hA = Kokkos::create_mirror_view(f.bws.tables.buck_A);
        auto hB = Kokkos::create_mirror_view(f.bws.tables.buck_B);
        auto hC = Kokkos::create_mirror_view(f.bws.tables.buck_C);
        auto hR = Kokkos::create_mirror_view(f.bws.tables.buck_Rmin);
        hA(0,0) = 60.0; hB(0,0) = 1.9; hC(0,0) = 40.0; hR(0,0) = 0;
        Kokkos::deep_copy(f.bws.tables.buck_A, hA);
        Kokkos::deep_copy(f.bws.tables.buck_B, hB);
        Kokkos::deep_copy(f.bws.tables.buck_C, hC);
        Kokkos::deep_copy(f.bws.tables.buck_Rmin, hR);
    }

    // Every molecule placed and oriented differently; finish_fixture only sets
    // molecule 0.
    auto h_bt2 = Kokkos::create_mirror_view(f.bws.dofs.translations);
    auto h_br  = Kokkos::create_mirror_view(f.bws.dofs.rot_vectors);
    for (int k = 0; k < n_indep; ++k) {
        h_bt2(0,k,0) = 0.11 + 0.17 * k; h_bt2(0,k,1) = 0.07 + 0.23 * k; h_bt2(0,k,2) = 0.13 - 0.09 * k;
        h_br(0,k,0)  = 0.21 - 0.13 * k; h_br(0,k,1)  = 0.14 + 0.19 * k; h_br(0,k,2)  = 0.09 + 0.11 * k;
    }
    Kokkos::deep_copy(f.bws.dofs.translations, h_bt2);
    Kokkos::deep_copy(f.bws.dofs.rot_vectors, h_br);

    auto h_cp = Kokkos::create_mirror_view(f.bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = f.mol.cell_params[i];
    Kokkos::deep_copy(f.bws.geom.cell_params, h_cp);
    batch_setup_cells(f.bws.tables, f.bws.geom);

    f.bws.dofs.cell_ref   = decltype(f.bws.dofs.cell_ref)("cell_ref", 1, 3, 3);
    f.bws.dofs.strain     = decltype(f.bws.dofs.strain)("strain", 1, 6);
    f.bws.dofs.d_g_strain = decltype(f.bws.dofs.d_g_strain)("d_g_strain", 1, 6);
    batch_set_strain_dofs(f.bws.geom, f.bws.dofs, static_cast<int>(TRICLINIC));
    Kokkos::deep_copy(f.bws.dofs.cell_ref, f.bws.geom.cell);
    Kokkos::deep_copy(f.bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(f.bws.dofs.d_g_strain, RealType(0));
    f.n_dof = n_indep * 6 + 6;
    return f;
}

inline Fixture make_fixture_p21c(double a_A, double b_A, double c_A, double beta) {
    Fixture f = make_fixture(a_A, b_A, c_A, 90.0, beta, 90.0);

    // Rebuild with four operations: allocate_molecule_data sizes the symop arrays.
    const int n_indep = 1, n_sites = 2, n_symops = 4;
    MoleculeData mol = allocate_molecule_data(n_indep, n_sites, n_symops,
                                              /*max_multipole_rank=*/1);
    for (int i = 0; i < 6; ++i) mol.cell_params[i] = f.mol.cell_params[i];
    Kokkos::deep_copy(mol.body_positions,  f.mol.body_positions);
    Kokkos::deep_copy(mol.body_charges,    f.mol.body_charges);
    Kokkos::deep_copy(mol.body_types,      f.mol.body_types);
    Kokkos::deep_copy(mol.body_multipoles, f.mol.body_multipoles);

    // P2_1/c:  (x,y,z)  (-x, y+1/2, -z+1/2)  (-x,-y,-z)  (x, -y+1/2, z+1/2)
    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    Kokkos::deep_copy(h_sr, RealType(0));
    Kokkos::deep_copy(h_st, RealType(0));
    const int diag[4][3] = {{1,1,1}, {-1,1,-1}, {-1,-1,-1}, {1,-1,1}};
    for (int op = 0; op < 4; ++op) {
        h_sr(op, 0) = RealType(diag[op][0]);
        h_sr(op, 4) = RealType(diag[op][1]);
        h_sr(op, 8) = RealType(diag[op][2]);
        h_si(op) = 0;
    }
    h_st(1, 1) = RealType(0.5); h_st(1, 2) = RealType(0.5);
    h_st(3, 1) = RealType(0.5); h_st(3, 2) = RealType(0.5);
    Kokkos::deep_copy(mol.sym_rotations,    h_sr);
    Kokkos::deep_copy(mol.sym_translations, h_st);
    Kokkos::deep_copy(mol.sym_indep_index,  h_si);

    mol.n_indep = n_indep;
    mol.n_symops = n_symops;
    mol.n_sites_per_mol = n_sites;

    f.mol = mol;
    f.bws = allocate_batch_workspace(1, f.ws, f.mol);
    finish_fixture(f);
    return f;
}

// ---------------------------------------------------------------------------
// One structure through the production kernel
// ---------------------------------------------------------------------------
//
// The production path with n_batch = 1, as batch_evaluate_energy does, built by
// hand to expose the EWALD_* term selector and the rigid-body gradient in
// dofs.d_gt.
struct SingleSlot {
    BatchWorkspace bws;
    BatchOptimizeSettings settings;
    int n_indep = 0;
};

inline SingleSlot make_single_slot(const MoleculeData& mol, const Workspace& ws,
                                   RealType ewald_accuracy = RealType(1e-8)) {
    SingleSlot s;
    s.bws = allocate_batch_workspace(1, ws, mol);
    s.bws.tables.ewald_accuracy = ewald_accuracy;
    s.n_indep = mol.n_indep;

    auto h_cp = Kokkos::create_mirror_view(s.bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = mol.cell_params[i];
    Kokkos::deep_copy(s.bws.geom.cell_params, h_cp);
    batch_setup_cells(s.bws.tables, s.bws.geom);

    auto h_bt = Kokkos::create_mirror_view(s.bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(s.bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror(Kokkos::HostSpace(), mol.translations);
    auto h_mr = Kokkos::create_mirror(Kokkos::HostSpace(), mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_bt(0, k, d) = h_mt(k, d);
            h_br(0, k, d) = h_mr(k, d);
        }
    Kokkos::deep_copy(s.bws.dofs.translations, h_bt);
    Kokkos::deep_copy(s.bws.dofs.rot_vectors, h_br);

    // The basis defaults: Tang-Toennies damping at its default range, and no
    // taper window (the kernel ramps the last 2 A down to the cutoff).
    s.settings.model.buck_damping = TangToenniesDamping{};
    s.settings.model.taper_on = 0;
    s.settings.model.taper_off = 0;
    return s;
}

// Overwrite one independent molecule's fractional translation.
inline void set_translation(SingleSlot& s, int k, RealType x, RealType y, RealType z) {
    auto h_bt = Kokkos::create_mirror_view(s.bws.dofs.translations);
    Kokkos::deep_copy(h_bt, s.bws.dofs.translations);
    h_bt(0, k, 0) = x; h_bt(0, k, 1) = y; h_bt(0, k, 2) = z;
    Kokkos::deep_copy(s.bws.dofs.translations, h_bt);
}

// Expand the asymmetric unit and run the energy+gradient kernel once. The
// energy is the return value; dofs.d_gt / dofs.d_gr hold the rigid-body
// gradient of exactly that energy.
template <int RANK>
RealType single_point(SingleSlot& s) {
    batch_expand_asymmetric_unit<RANK>(s.bws.tables, s.bws.geom, s.bws.dofs);
    Kokkos::fence();
    batch_energy_gradient_asymunit<RANK>(s.bws.tables, s.bws.geom, s.bws.dofs,
                                         s.settings, 0);
    Kokkos::fence();
    auto hE = Kokkos::create_mirror(Kokkos::HostSpace(), s.bws.geom.energies);
    Kokkos::deep_copy(hE, s.bws.geom.energies);
    return hE(0);
}

// Energy with one subset of the model's terms switched on, leaving the
// selector as it was found.
template <int RANK>
RealType single_point_terms(SingleSlot& s, uint32_t components) {
    const uint32_t saved = s.settings.model.ewald_components;
    s.settings.model.ewald_components = components;
    const RealType e = single_point<RANK>(s);
    s.settings.model.ewald_components = saved;
    return e;
}

// Rock salt, eight ions, one independent single-atom molecule per ion. Not one
// eight-site molecule: the pair enumeration drops intramolecular zero-shift
// pairs, which would remove Na-Cl contacts from the Madelung sum. Each ion's
// d_gt is its Cartesian force scaled by the cell length.
// `buckingham` adds a soft Na-Cl pair potential; without it the model is pure
// electrostatics.
inline std::pair<MoleculeData, Workspace> make_nacl_ions(
        RealType a = RealType(5.6402) * ANGSTROM_TO_BOHR,
        bool buckingham = false, int mp_rank = 0) {
    const int n = 8;
    auto mol = allocate_molecule_data(n, 1, n, mp_rank);
    auto ws  = allocate_workspace(n, 30000, 50000, 2, mp_rank);

    mol.cell_params[0] = a; mol.cell_params[1] = a; mol.cell_params[2] = a;
    mol.cell_params[3] = 90.0; mol.cell_params[4] = 90.0; mol.cell_params[5] = 90.0;

    // Na at (0,0,0), (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2);
    // Cl at (1/2,0,0), (0,1/2,0), (0,0,1/2), (1/2,1/2,1/2).
    const RealType h = RealType(0.5);
    const RealType frac[8][3] = {
        {0, 0, 0}, {h, h, 0}, {h, 0, h}, {0, h, h},
        {h, 0, 0}, {0, h, 0}, {0, 0, h}, {h, h, h},
    };

    auto h_t  = Kokkos::create_mirror_view(mol.translations);
    auto h_q  = Kokkos::create_mirror_view(mol.body_charges);
    auto h_ty = Kokkos::create_mirror_view(mol.body_types);
    auto h_mp = Kokkos::create_mirror_view(mol.body_multipoles);
    Kokkos::deep_copy(h_mp, RealType(0));
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) h_t(i, d) = frac[i][d];
        h_q(i)  = (i < 4) ? RealType(1) : RealType(-1);
        h_ty(i) = (i < 4) ? 0 : 1;          // Na = 0, Cl = 1
        if (mp_rank > 0) h_mp(i, 0) = h_q(i);
    }
    Kokkos::deep_copy(mol.translations, h_t);
    Kokkos::deep_copy(mol.body_charges, h_q);
    Kokkos::deep_copy(mol.body_types, h_ty);
    Kokkos::deep_copy(mol.body_multipoles, h_mp);
    Kokkos::deep_copy(mol.rot_vectors, RealType(0));
    Kokkos::deep_copy(mol.body_positions, RealType(0));

    // P1 with Z' = 8: identity for every copy, copy k belongs to molecule k, so
    // sym_indep_index is the identity map.
    auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
    auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
    Kokkos::deep_copy(h_sr, RealType(0));
    for (int op = 0; op < n; ++op) {
        h_sr(op, 0) = 1; h_sr(op, 4) = 1; h_sr(op, 8) = 1;
        h_si(op) = op;
    }
    Kokkos::deep_copy(mol.sym_rotations, h_sr);
    Kokkos::deep_copy(mol.sym_translations, RealType(0));
    Kokkos::deep_copy(mol.sym_indep_index, h_si);

    mol.n_indep = n;
    mol.n_symops = n;
    mol.n_sites_per_mol = 1;
    mol.max_multipole_rank = mp_rank;

    ws.n_types = 2;
    ws.buck_cutoff = 12.0 * ANGSTROM_TO_BOHR;
    if (buckingham) {
        auto hA = Kokkos::create_mirror_view(ws.buck_A);
        auto hB = Kokkos::create_mirror_view(ws.buck_B);
        auto hC = Kokkos::create_mirror_view(ws.buck_C);
        Kokkos::deep_copy(hA, RealType(0));
        Kokkos::deep_copy(hB, RealType(0));
        Kokkos::deep_copy(hC, RealType(0));
        const RealType A = 1000.0 / AU_TO_KJ_PER_MOL;
        const RealType B = 3.0 * BOHR_TO_ANGSTROM;
        const RealType C = 10.0 * std::pow(ANGSTROM_TO_BOHR, 6) / AU_TO_KJ_PER_MOL;
        hA(0,1) = A; hA(1,0) = A;
        hB(0,1) = B; hB(1,0) = B;
        hC(0,1) = C; hC(1,0) = C;
        Kokkos::deep_copy(ws.buck_A, hA);
        Kokkos::deep_copy(ws.buck_B, hB);
        Kokkos::deep_copy(ws.buck_C, hC);
    }
    return {mol, ws};
}

}  // namespace klasp::testing
