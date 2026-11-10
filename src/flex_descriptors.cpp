#include "klasp/flex_descriptors.hpp"

#include "klasp/batch.hpp"
#include "klasp/dedup.hpp"   // AMD_K
#include "klasp/cell.hpp"
#include "klasp/element.hpp"
#include "klasp/constants.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace klasp {

namespace {

// Interplanar spacing normal to axis `ax`: volume over the area of the face the
// other two vectors span, as batch_setup_cells uses to size the image search.
double axis_spacing(const double cell[3][3], int ax, double vol) {
    const int j = (ax + 1) % 3, k = (ax + 2) % 3;
    const double cx = cell[1][j]*cell[2][k] - cell[2][j]*cell[1][k];
    const double cy = cell[2][j]*cell[0][k] - cell[0][j]*cell[2][k];
    const double cz = cell[0][j]*cell[1][k] - cell[1][j]*cell[0][k];
    const double cn = std::max(1e-10, std::sqrt(cx*cx + cy*cy + cz*cz));
    return vol / cn;
}

}  // namespace

FlexDescriptors compute_flex_descriptors(const std::vector<CrystalGeometry>& geoms,
                                         bool want_pdd, double cutoff_bohr, int k,
                                         int fold_rows) {
    FlexDescriptors out;
    if (geoms.empty()) return out;

    const int n_batch = static_cast<int>(geoms.size());
    const int n_atoms = static_cast<int>(geoms[0].positions.size());
    if (n_atoms < 2) return out;
    for (const auto& g : geoms)
        if (static_cast<int>(g.positions.size()) != n_atoms)
            throw std::runtime_error(
                "compute_flex_descriptors: structures differ in atom count");

    // n_indep = 1 and n_sites_per_mol = n_atoms make every atom a descriptor row;
    // ghost_type = -1 because the PET geometry has no padding.
    BatchWorkspace bws;
    bws.geom.n_batch = n_batch;
    bws.geom.n_atoms = n_atoms;
    bws.tables.n_indep = 1;
    bws.tables.n_sites_per_mol = n_atoms;
    bws.tables.ghost_type = -1;
    bws.geom.positions = SView3D<RealType, ExecSpace>("flex_pos", n_batch, n_atoms, 3);
    bws.geom.cell      = SView3D<RealType, ExecSpace>("flex_cell", n_batch, 3, 3);
    bws.geom.cell_inv  = SView3D<RealType, ExecSpace>("flex_cell_inv", n_batch, 3, 3);
    bws.geom.max_shift = SView2D<int, ExecSpace>("flex_mshift", n_batch, 3);

    auto h_pos   = Kokkos::create_mirror_view(bws.geom.positions);
    auto h_cell  = Kokkos::create_mirror_view(bws.geom.cell);
    auto h_cinv  = Kokkos::create_mirror_view(bws.geom.cell_inv);
    auto h_shift = Kokkos::create_mirror_view(bws.geom.max_shift);
    std::vector<double> reach_used(n_batch, 0.0);
    std::vector<std::array<double, 3>> spacing(n_batch);

    for (int b = 0; b < n_batch; ++b) {
        const auto& g = geoms[b];
        double inv[3][3] = {};
        mat3_inv(g.cell, inv);  // zeros on a degenerate cell

        // Wrap into the home cell: the image range below assumes every atom is
        // inside it, and relaxed structures are not re-wrapped.
        for (int i = 0; i < n_atoms; ++i) {
            double f[3];
            for (int r = 0; r < 3; ++r)
                f[r] = inv[r][0]*g.positions[i][0] + inv[r][1]*g.positions[i][1]
                     + inv[r][2]*g.positions[i][2];
            for (int r = 0; r < 3; ++r) f[r] -= std::floor(f[r]);
            for (int r = 0; r < 3; ++r)
                h_pos(b, i, r) = static_cast<RealType>(
                    g.cell[r][0]*f[0] + g.cell[r][1]*f[1] + g.cell[r][2]*f[2]);
        }

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                h_cell(b, r, c) = static_cast<RealType>(g.cell[r][c]);
                h_cinv(b, r, c) = static_cast<RealType>(inv[r][c]);
            }

        const double det =
            g.cell[0][0]*(g.cell[1][1]*g.cell[2][2] - g.cell[1][2]*g.cell[2][1]) -
            g.cell[0][1]*(g.cell[1][0]*g.cell[2][2] - g.cell[1][2]*g.cell[2][0]) +
            g.cell[0][2]*(g.cell[1][0]*g.cell[2][1] - g.cell[1][1]*g.cell[2][0]);
        const double vol = std::fabs(det);
        // Radius whose sphere holds ~3x k atoms at this structure's density.
        double reach = cutoff_bohr;
        if (reach <= 0.0) {
            const double per_atom = vol / std::max(1, n_atoms);
            reach = 1.5 * std::cbrt(3.0 * k * per_atom / (4.0 * M_PI));
        }
        reach_used[b] = reach;
        for (int ax = 0; ax < 3; ++ax) {
            const double d = axis_spacing(g.cell, ax, vol);
            spacing[b][ax] = d;
            h_shift(b, ax) = static_cast<int>(std::ceil(reach / std::max(1e-10, d))) + 1;
        }
    }
    Kokkos::deep_copy(bws.geom.positions, h_pos);
    Kokkos::deep_copy(bws.geom.cell, h_cell);
    Kokkos::deep_copy(bws.geom.cell_inv, h_cinv);
    Kokkos::deep_copy(bws.geom.max_shift, h_shift);

    // The reach comes from the mean density and can be too short for a clustered
    // structure, so grow it until the furthest neighbour used is well inside.
    AMDResult amd = compute_amd(bws.tables, bws.geom, k, want_pdd);
    for (int attempt = 0; attempt < 4; ++attempt) {
        double furthest_A = 0;                       // AMD is in Angstrom
        for (const auto& v : amd.vectors)
            if (!v.empty()) furthest_A = std::max(furthest_A, v.back());
        const double furthest = furthest_A * ANGSTROM_TO_BOHR;
        bool enough = true;
        for (int b = 0; b < n_batch && enough; ++b)
            if (reach_used[b] < 1.5 * furthest) enough = false;
        if (enough) break;

        for (int b = 0; b < n_batch; ++b) {
            reach_used[b] = std::max(reach_used[b], 1.5 * furthest) * 1.3;
            for (int ax = 0; ax < 3; ++ax)
                h_shift(b, ax) = static_cast<int>(
                    std::ceil(reach_used[b] / std::max(1e-10, spacing[b][ax]))) + 1;
        }
        Kokkos::deep_copy(bws.geom.max_shift, h_shift);
        amd = compute_amd(bws.tables, bws.geom, k, want_pdd);
    }
    out.amd = amd.vectors;
    out.pdd = amd.pdd;
    out.n_asym = amd.n_asym;
    if (want_pdd && fold_rows > 0) {
        if (n_atoms % fold_rows != 0) {
            out.pdd.clear();
        } else {
            const int copies = n_atoms / fold_rows;
            for (auto& rows : out.pdd) {
                std::vector<double> folded(static_cast<std::size_t>(fold_rows) * k, 0.0);
                for (int i = 0; i < n_atoms; ++i) {
                    const double* src = rows.data() + static_cast<std::size_t>(i) * k;
                    double* dst = folded.data() + static_cast<std::size_t>(i % fold_rows) * k;
                    for (int j = 0; j < k; ++j) dst[j] += src[j];
                }
                for (double& v : folded) v /= copies;
                rows = std::move(folded);
            }
            out.n_asym = fold_rows;
        }
    }

    // Scattering factors from the structure's own atomic numbers: the compacted
    // PET geometry does not follow the workspace's padded element order.
    std::vector<std::string> elements;
    elements.reserve(n_atoms);
    for (int i = 0; i < n_atoms; ++i)
        elements.push_back(element_symbol(geoms[0].atomic_numbers[i]));
    const PowderFormFactors ff = build_powder_form_factors(elements);
    const PowderResult pw = compute_powder_pattern(bws.geom, ff);
    out.powder.resize(pw.patterns.size());
    for (std::size_t b = 0; b < pw.patterns.size(); ++b) {
        out.powder[b].resize(pw.patterns[b].size());
        for (std::size_t i = 0; i < pw.patterns[b].size(); ++i)
            out.powder[b][i] = static_cast<float>(pw.patterns[b][i]);
    }
    return out;
}

}  // namespace klasp
