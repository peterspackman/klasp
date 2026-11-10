#include "klasp/cmd_fdcheck.hpp"

#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/cell.hpp"
#include "klasp/constants.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/ewald.hpp"
#include "klasp/io.hpp"
#include "klasp/molecule.hpp"
#include "klasp/multipole.hpp"
#include "klasp/taper.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/types.hpp"

#include <fmt/base.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace klasp {

int cmd_fdcheck(const FdcheckOptions& o) {
    const std::string& path = o.input;
    const bool no_taper = o.no_taper;
    const double abs_tol = o.abs_tol;
    const double rel_tol = o.rel_tol;
    auto ws = allocate_workspace(512, 27*27*27, 50000, 32, 4);
    MoleculeData mol{};
    auto config = load_structure_json(path, mol, ws);
    apply_ewald_accuracy_override(o.ewald_accuracy, ws);
    if (no_taper) { config.taper_on = 0.0; config.taper_off = 0.0; }

    // Batch path with n_batch = 1 for both analytic and FD evaluations.
    const int n_batch = 1;
    auto bws = allocate_batch_workspace(n_batch, ws, mol);

    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_mt = Kokkos::create_mirror_view(mol.translations);
    auto h_mr = Kokkos::create_mirror_view(mol.rot_vectors);
    Kokkos::deep_copy(h_mt, mol.translations);
    Kokkos::deep_copy(h_mr, mol.rot_vectors);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_bt(0, k, d) = h_mt(k, d);
            h_br(0, k, d) = h_mr(k, d);
        }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);

    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);
    for (int i = 0; i < 6; ++i) h_cp(0, i) = mol.cell_params[i];
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);

    // Allocate strain views if needed, to test the strain gradient.
    if (bws.dofs.cell_ref.extent(0) == 0) {
        bws.dofs.cell_ref = decltype(bws.dofs.cell_ref)("cell_ref", n_batch, 3, 3);
        bws.dofs.strain = decltype(bws.dofs.strain)("strain", n_batch, 6);
        bws.dofs.d_g_strain = decltype(bws.dofs.d_g_strain)("d_g_strain", n_batch, 6);
    }
    int sg_num = config.space_group > 0 ? config.space_group : 1;
    auto& sg_cell = get_space_group(sg_num);
    batch_set_strain_dofs(bws.geom, bws.dofs, static_cast<int>(sg_cell.lattice));
    Kokkos::deep_copy(bws.dofs.cell_ref, bws.geom.cell);
    Kokkos::deep_copy(bws.dofs.strain, RealType(0));
    Kokkos::deep_copy(bws.dofs.d_g_strain, RealType(0));

    const int n_mol_dof = mol.n_indep * 6;
    const int n_cell_dof = bws.dofs.n_cell_dof;
    const int n_dof = n_mol_dof + n_cell_dof;

    BatchOptimizeSettings bsettings;
    bsettings.n_iterations = 1;
    bsettings.max_step = 0.05;
    bsettings.lbfgs_memory = 5;
    bsettings.model.taper_on = config.taper_on;
    bsettings.model.taper_off = config.taper_off;
    bsettings.model.buck_damping = config.buck_damping;
    bsettings.pressure_gpa = 0.0;     // fdcheck compares against lattice E, not enthalpy
    bsettings.energy_only = true;     // just compute energy + gradient, no step
    bsettings.model.wolf = false;           // Ewald (correct long-range)
    // Precompute Wolf self-energy shift
    {
        auto h_kappa_w = Kokkos::create_mirror_view(bws.geom.kappa);
        Kokkos::deep_copy(h_kappa_w, bws.geom.kappa);
        RealType kw = h_kappa_w(0);
        RealType sw = ewald_split_width(bws.tables.ewald_accuracy);
        RealType rcw = (bsettings.model.wolf_cutoff > 0) ? bsettings.model.wolf_cutoff : sw / kw;
        bsettings.model.wolf_self_shift = compute_wolf_constants(kw, rcw).self_shift;
    }
    const ForceField ff = make_force_field(bws.tables, bsettings.model);

    auto lbfgs = allocate_batch_lbfgs_state(n_batch, n_dof, 5);

    const RealType eps = 5e-4;  // balance between truncation and roundoff

    fmt::print("\nFinite difference gradient check (batch kernel, eps={:.0e}):\n", eps);
    fmt::print("  n_atoms={}, n_indep={}, n_symops={}, n_cell_dof={}\n",
               bws.geom.n_atoms, mol.n_indep, mol.n_symops, n_cell_dof);
    fmt::print("\n  {:<14s} {:>12s} {:>12s} {:>12s} {:>8s}\n",
               "DOF", "analytical", "FD", "diff", "status");

    RealType max_err = 0;
    int n_fail = 0;
    // Pass if within atol absolutely or rtol relatively.
    const RealType atol = static_cast<RealType>(abs_tol);
    const RealType rtol = static_cast<RealType>(rel_tol);

    // --max-order only ever lowers the basis's rank.
    const int fd_max_rank = o.max_order >= 0
                                ? std::min(o.max_order, config.max_rank)
                                : config.max_rank;
    auto expand_for_fd = [&]() {
        if (fd_max_rank >= 3)      batch_expand_asymmetric_unit<4>(bws.tables, bws.geom, bws.dofs);
        else if (fd_max_rank >= 1) batch_expand_asymmetric_unit<2>(bws.tables, bws.geom, bws.dofs);
        else                       batch_expand_asymmetric_unit<0>(bws.tables, bws.geom, bws.dofs);
        Kokkos::fence();
    };
    auto eval_grad_for_fd = [&](const BatchOptimizeSettings& s) {
        if (fd_max_rank >= 3)      batch_energy_gradient_asymunit<4>(bws.tables, bws.geom, bws.dofs, s, 0);
        else if (fd_max_rank >= 1) batch_energy_gradient_asymunit<2>(bws.tables, bws.geom, bws.dofs, s, 0);
        else                       batch_energy_gradient_asymunit<0>(bws.tables, bws.geom, bws.dofs, s, 0);
        Kokkos::fence();
    };
    auto eval_energy = [&]() -> RealType {
        expand_for_fd();
        BatchOptimizeSettings eo = bsettings;
        eo.energy_only = true;
        eval_grad_for_fd(eo);
        auto hE = Kokkos::create_mirror_view(bws.geom.energies);
        Kokkos::deep_copy(hE, bws.geom.energies);
        return hE(0);
    };

    auto h_bt_orig = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.translations);
    auto h_br_orig = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.rot_vectors);
    auto h_strain_orig = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.strain);
    Kokkos::deep_copy(h_bt_orig, bws.dofs.translations);
    Kokkos::deep_copy(h_br_orig, bws.dofs.rot_vectors);
    Kokkos::deep_copy(h_strain_orig, bws.dofs.strain);

    expand_for_fd();
    eval_grad_for_fd(bsettings);

    auto h_E = Kokkos::create_mirror_view(bws.geom.energies);
    Kokkos::deep_copy(h_E, bws.geom.energies);
    RealType E0 = h_E(0);

    auto h_dgt = Kokkos::create_mirror_view(bws.dofs.d_gt);
    auto h_dgr = Kokkos::create_mirror_view(bws.dofs.d_gr);
    auto h_dgs_grad = Kokkos::create_mirror_view(bws.dofs.d_g_strain);
    Kokkos::deep_copy(h_dgt, bws.dofs.d_gt);
    Kokkos::deep_copy(h_dgr, bws.dofs.d_gr);
    Kokkos::deep_copy(h_dgs_grad, bws.dofs.d_g_strain);
    std::vector<RealType> anal_grad(n_dof);
    int gi = 0;
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d)
            anal_grad[gi++] = h_dgt(0, k, d);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d)
            anal_grad[gi++] = h_dgr(0, k, d);
    auto h_basis = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       bws.dofs.cell_dof_basis);
    for (int s = 0; s < n_cell_dof; ++s) {
        RealType g = 0;
        for (int v = 0; v < 6; ++v) g += h_basis(0, s, v) * h_dgs_grad(0, v);
        anal_grad[gi++] = g;
    }

    fmt::print("  E = {:.8f} Hartree ({:.4f} kJ/mol)\n", E0, E0 * AU_TO_KJ_PER_MOL);

    // E0 must match eval_energy at the same point.
    RealType E0_check = eval_energy();
    fmt::print("  E0={:.10f} E0_check={:.10f} diff={:.2e}\n", E0, E0_check, E0 - E0_check);

    for (int dof = 0; dof < n_dof; ++dof) {
        RealType anal = anal_grad[dof];

        Kokkos::deep_copy(bws.dofs.translations, h_bt_orig);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br_orig);
        Kokkos::deep_copy(bws.dofs.strain, h_strain_orig);
        Kokkos::deep_copy(bws.geom.cell, bws.dofs.cell_ref);

        auto h_bt2 = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.translations);
        auto h_br2 = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.rot_vectors);
        auto h_s2 = Kokkos::create_mirror(Kokkos::HostSpace(), bws.dofs.strain);
        Kokkos::deep_copy(h_bt2, bws.dofs.translations);
        Kokkos::deep_copy(h_br2, bws.dofs.rot_vectors);
        Kokkos::deep_copy(h_s2, bws.dofs.strain);

        if (dof < mol.n_indep * 3) {
            int k = dof / 3, d = dof % 3;
            h_bt2(0, k, d) += eps;
        } else if (dof < n_mol_dof) {
            int ri = dof - mol.n_indep * 3;
            int k = ri / 3, d = ri % 3;
            h_br2(0, k, d) += eps;
        } else {
            int si = dof - n_mol_dof;
            for (int v = 0; v < 6; ++v) h_s2(0, v) += eps * h_basis(0, si, v);
        }
        Kokkos::deep_copy(bws.dofs.translations, h_bt2);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br2);
        Kokkos::deep_copy(bws.dofs.strain, h_s2);
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);
        RealType Ep = eval_energy();

        Kokkos::deep_copy(bws.dofs.translations, h_bt_orig);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br_orig);
        Kokkos::deep_copy(bws.dofs.strain, h_strain_orig);
        Kokkos::deep_copy(bws.geom.cell, bws.dofs.cell_ref);

        Kokkos::deep_copy(h_bt2, h_bt_orig);
        Kokkos::deep_copy(h_br2, h_br_orig);
        Kokkos::deep_copy(h_s2, h_strain_orig);

        if (dof < mol.n_indep * 3) {
            int k = dof / 3, d = dof % 3;
            h_bt2(0, k, d) -= eps;
        } else if (dof < n_mol_dof) {
            int ri = dof - mol.n_indep * 3;
            int k = ri / 3, d = ri % 3;
            h_br2(0, k, d) -= eps;
        } else {
            int si = dof - n_mol_dof;
            for (int v = 0; v < 6; ++v) h_s2(0, v) -= eps * h_basis(0, si, v);
        }
        Kokkos::deep_copy(bws.dofs.translations, h_bt2);
        Kokkos::deep_copy(bws.dofs.rot_vectors, h_br2);
        Kokkos::deep_copy(bws.dofs.strain, h_s2);
        batch_update_cell_from_strain(ff, bws.geom, bws.dofs);
        RealType Em = eval_energy();

        RealType fd = (Ep - Em) / (2.0 * eps);
        RealType diff = std::abs(anal - fd);
        if (diff > max_err) max_err = diff;
        bool ok = diff <= atol + rtol * std::max(std::abs(anal), std::abs(fd));

        const char* label;
        int idx_in_group, d_comp;
        if (dof < mol.n_indep * 3) {
            label = "trans";
            idx_in_group = dof / 3;
            d_comp = dof % 3;
        } else if (dof < n_mol_dof) {
            label = "rot";
            idx_in_group = (dof - mol.n_indep * 3) / 3;
            d_comp = (dof - mol.n_indep * 3) % 3;
        } else {
            label = "strain";
            idx_in_group = dof - n_mol_dof;
            d_comp = -1;
        }

        if (d_comp >= 0)
            fmt::print("  {}{}_{:<6d} {:12.6f} {:12.6f} {:12.6f} {:>8s}\n",
                       label, idx_in_group, d_comp, anal, fd, anal - fd, ok ? "OK" : "FAIL");
        else
            fmt::print("  {}[{:<2d}]     {:12.6f} {:12.6f} {:12.6f} {:>8s}\n",
                       label, idx_in_group, anal, fd, anal - fd, ok ? "OK" : "FAIL");
        if (!ok) ++n_fail;
    }

    fmt::print("\n  Max |diff| = {:.6f} Hartree\n", max_err);
    fmt::print("  {} / {} DOFs pass (|diff| <= {:.1e} + {:.1e} * |grad|)\n",
               n_dof - n_fail, n_dof, atol, rtol);
    if (n_fail)
        fmt::print(stderr,
                   "\nfdcheck FAILED: {} of {} DOFs disagree with finite differences.\n",
                   n_fail, n_dof);

    Kokkos::deep_copy(bws.dofs.translations, h_bt_orig);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br_orig);
    Kokkos::deep_copy(bws.dofs.strain, h_strain_orig);
    return n_fail;
}

}  // namespace klasp
