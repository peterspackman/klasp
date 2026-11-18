// Construction of the symmetry-adapted strain DOF basis; see
// include/klasp/strain_basis.hpp.

#include "klasp/strain_basis.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_stream.hpp"
#include "klasp/log.hpp"
#include "klasp/sgdata.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace klasp {

PointGroup point_group_of(const RealType* rotations, int n) {
    using Rot = std::array<int, 9>;
    PointGroup pg;
    auto add = [&](const Rot& w) {
        if (std::find(pg.rotations.begin(), pg.rotations.end(), w) != pg.rotations.end())
            return false;
        if (pg.rotations.size() >= 48)
            throw std::runtime_error(
                "the symmetry operations generate more than 48 rotations, so they are "
                "not a crystallographic point group");
        pg.rotations.push_back(w);
        return true;
    };

    add(Rot{1, 0, 0, 0, 1, 0, 0, 0, 1});
    for (int k = 0; k < n; ++k) {
        Rot w;
        for (int i = 0; i < 9; ++i) {
            const double x = static_cast<double>(rotations[k*9 + i]);
            const double r = std::round(x);
            if (!(std::abs(x - r) < 1e-6))
                throw std::runtime_error(fmt::format(
                    "symmetry operation {} has a non-integer rotation part ({} at {},{}); "
                    "fractional rotations are integer matrices in any lattice basis",
                    k, x, i / 3, i % 3));
            w[i] = static_cast<int>(r);
        }
        add(w);
    }

    for (bool grew = true; grew;) {
        grew = false;
        const size_t m = pg.rotations.size();
        for (size_t a = 0; a < m; ++a)
            for (size_t b = 0; b < m; ++b) {
                Rot c{};
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        for (int l = 0; l < 3; ++l)
                            c[i*3 + j] += pg.rotations[a][i*3 + l] * pg.rotations[b][l*3 + j];
                grew |= add(c);
            }
    }

    // dim of the fixed space of Sym^2: (1/|G|) sum_g (tr(g)^2 + tr(g^2)) / 2.
    long sum = 0;
    for (const Rot& w : pg.rotations) {
        const long tr = w[0] + w[4] + w[8];
        long tr2 = 0;
        for (int i = 0; i < 3; ++i)
            for (int l = 0; l < 3; ++l) tr2 += w[i*3 + l] * w[l*3 + i];
        sum += (tr * tr + tr2) / 2;
    }
    const long order = static_cast<long>(pg.rotations.size());
    if (sum % order != 0)
        throw std::runtime_error(fmt::format(
            "the {} rotations do not form a group (character sum {} is not a multiple "
            "of the order)", order, sum));
    pg.strain_dim = static_cast<int>(sum / order);
    return pg;
}

int symmetric_strain_basis(const PointGroup& pg, const RealType cell[9], bool diag_only,
                           RealType basis[36]) {
    std::vector<RealType> w(pg.rotations.size() * 9);
    for (size_t k = 0; k < pg.rotations.size(); ++k)
        for (int i = 0; i < 9; ++i) w[k*9 + i] = static_cast<RealType>(pg.rotations[k][i]);
    RealType orth_err = 0;
    const int n = build_symmetric_strain_basis(cell, w.data(), static_cast<int>(pg.rotations.size()),
                                               pg.strain_dim, diag_only, basis, &orth_err);
    if (orth_err > RealType(1e-4))
        logging::warn("warning: the cell's metric does not fit its symmetry operations "
                      "(max |R^T R - I| = {:.2e}); the strain basis is approximate\n",
                      static_cast<double>(orth_err));
    return n;
}

// ---------------------------------------------------------------------------

template <typename Space>
PointGroup point_group_of(const SymmetryOpsT<Space>& ops) {
    auto h_rot = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ops.sym_rotations);
    return point_group_of(h_rot.data(), ops.n_symops);
}

namespace {

template <typename Space>
void ensure_basis_view(int n_slots, RigidDofStateT<Space>& dofs) {
    if ((int)dofs.cell_dof_basis.extent(0) < n_slots)
        dofs.cell_dof_basis = SView3D<RealType, Space>("cell_dof_basis", n_slots, 6, 6);
}

// Rows for cells(i), i in [begin, end), into rows(i - begin) when given. All
// must have the same count, which is returned.
template <class Cells, class Rows>
int build_rows(const PointGroup& pg, const Cells& cells, int begin, int end, bool fix_angles,
               const char* what, Rows* rows) {
    int n_dof = -1;
    for (int b = begin; b < end; ++b) {
        RealType cell[9], basis[36];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) cell[r*3 + c] = cells(b, r, c);
        const int n = symmetric_strain_basis(pg, cell, fix_angles, basis);
        if (n < 0)
            throw std::runtime_error(fmt::format(
                "{} {}: no strain basis of dimension {} fits its cell and its {} symmetry "
                "rotations", what, b, pg.strain_dim, pg.rotations.size()));
        if (n_dof >= 0 && n != n_dof)
            throw std::runtime_error(fmt::format(
                "{} {} has {} strain degrees of freedom where those before it have {}; a "
                "batch shares one count", what, b, n, n_dof));
        n_dof = n;
        if (rows)
            for (int s = 0; s < 6; ++s)
                for (int v = 0; v < 6; ++v) (*rows)(b - begin, s, v) = basis[s*6 + v];
    }
    return n_dof < 0 ? 0 : n_dof;
}

// What a refill needs to rebuild a slot's rows.
template <typename Space>
void set_refill_tables(const PointGroup& pg, bool fix_angles, int n_dof,
                       RigidDofStateT<Space>& dofs) {
    const int n_rot = static_cast<int>(pg.rotations.size());
    dofs.cell_dof_rot = SView2D<RealType, Space>("cell_dof_rot", n_rot, 9);
    auto h_rot = Kokkos::create_mirror_view(dofs.cell_dof_rot);
    for (int k = 0; k < n_rot; ++k)
        for (int i = 0; i < 9; ++i) h_rot(k, i) = static_cast<RealType>(pg.rotations[k][i]);
    Kokkos::deep_copy(dofs.cell_dof_rot, h_rot);
    dofs.cell_dof_n_rot = n_rot;
    dofs.cell_dof_dim = pg.strain_dim;
    dofs.cell_dof_diag_only = fix_angles;
    dofs.n_cell_dof = n_dof;
}

}  // namespace

template <typename Space>
int batch_set_strain_dofs(const SymmetryOpsT<Space>& ops, const BatchGeometryT<Space>& geom,
                          RigidDofStateT<Space>& dofs, bool fix_angles) {
    const int n_batch = geom.n_batch;
    if ((int)dofs.cell_ref.extent(0) < n_batch)
        throw std::logic_error(fmt::format(
            "batch_set_strain_dofs needs cell_ref for all {} structures; it holds {}",
            n_batch, dofs.cell_ref.extent(0)));
    ensure_basis_view(n_batch, dofs);

    const PointGroup pg = point_group_of(ops);
    auto h_cell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dofs.cell_ref);
    auto h_basis = Kokkos::create_mirror_view(dofs.cell_dof_basis);
    Kokkos::deep_copy(h_basis, RealType(0));
    const int n_dof = build_rows(pg, h_cell, 0, n_batch, fix_angles, "structure", &h_basis);
    Kokkos::deep_copy(dofs.cell_dof_basis, h_basis);
    set_refill_tables(pg, fix_angles, n_dof, dofs);
    return n_dof;
}

template <typename Space>
int batch_set_strain_dofs(const BatchGeometryT<Space>& geom, RigidDofStateT<Space>& dofs,
                          const RealType basis[36], int n) {
    ensure_basis_view(geom.n_batch, dofs);
    auto h_basis = Kokkos::create_mirror_view(dofs.cell_dof_basis);
    for (int b = 0; b < geom.n_batch; ++b)
        for (int s = 0; s < 6; ++s)
            for (int v = 0; v < 6; ++v) h_basis(b, s, v) = s < n ? basis[s*6 + v] : RealType(0);
    Kokkos::deep_copy(dofs.cell_dof_basis, h_basis);
    dofs.cell_dof_rot = SView2D<RealType, Space>();
    dofs.cell_dof_n_rot = 0;
    dofs.cell_dof_dim = n;
    dofs.cell_dof_diag_only = false;
    dofs.n_cell_dof = n;
    return n;
}

template <typename Space>
int stream_set_strain_dofs(const SymmetryOpsT<Space>& ops, const StreamPoolT<Space>& pool,
                           RigidDofStateT<Space>& dofs, int n_slots, bool fix_angles) {
    ensure_basis_view(n_slots, dofs);
    Kokkos::deep_copy(dofs.cell_dof_basis, RealType(0));
    const PointGroup pg = point_group_of(ops);
    auto h_cell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pool.cells.cell);
    using NoRows = Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>;
    const int n_dof = build_rows(pg, h_cell, pool.consumed, pool.filled, fix_angles,
                                 "pool entry", static_cast<NoRows*>(nullptr));
    set_refill_tables(pg, fix_angles, n_dof, dofs);
    return n_dof;
}

void warn_strain_refill_failures(int n_failed) {
    if (n_failed > 0)
        logging::warn("warning: {} refilled structure{} had no strain basis that fits "
                      "{} cell and operations; {} cell{} held fixed\n", n_failed,
                      n_failed == 1 ? "" : "s", n_failed == 1 ? "its" : "their",
                      n_failed == 1 ? "its" : "their", n_failed == 1 ? " was" : "s were");
}

void log_strain_dofs(const PointGroup& ops_pg, int space_group, int n_cell_dof) {
    logging::info("  cell: {} strain degree{} of freedom from {} symmetry rotation{}\n",
                  n_cell_dof, n_cell_dof == 1 ? "" : "s",
                  ops_pg.rotations.size(), ops_pg.rotations.size() == 1 ? "" : "s");
    if (space_group < 1 || space_group > 230) return;
    const auto& sg = get_space_group(space_group);
    std::vector<RealType> w;
    for (int k = 0; k < sg.n_symops; ++k)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) w.push_back(static_cast<RealType>(sg.symops[k].rot[r][c]));
    const PointGroup sg_pg = point_group_of(w.data(), sg.n_symops);
    if (sg_pg.rotations.size() != ops_pg.rotations.size() ||
        sg_pg.strain_dim != ops_pg.strain_dim)
        logging::warn("warning: space_group {} has {} rotations and {} strain directions, "
                      "but the structure's operations have {} and {}; the cell relaxes "
                      "under the operations\n",
                      space_group, sg_pg.rotations.size(), sg_pg.strain_dim,
                      ops_pg.rotations.size(), ops_pg.strain_dim);
}

template PointGroup point_group_of<ExecSpace>(const SymmetryOps&);
template int batch_set_strain_dofs<ExecSpace>(const SymmetryOps&, const BatchGeometry&,
                                              RigidDofState&, bool);
template int batch_set_strain_dofs<ExecSpace>(const BatchGeometry&, RigidDofState&,
                                              const RealType*, int);
template int stream_set_strain_dofs<ExecSpace>(const SymmetryOps&, const StreamPool&,
                                               RigidDofState&, int, bool);

}  // namespace klasp
