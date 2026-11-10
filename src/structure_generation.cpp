// Crystal structures from the CSP generator, in memory. See
// klasp/structure_generation.hpp.

#include "klasp/structure_generation.hpp"

#include "klasp/constants.hpp"
#include "klasp/csp.hpp"
#include "klasp/element.hpp"
#include "klasp/ff_structure.hpp"
#include "klasp/fft_generator.hpp"
#include "klasp/generator.hpp"
#include "klasp/io.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace klasp {

using json = nlohmann::json;

namespace {

// What both generators read about the molecules. Must be built exactly as
// csp_run and SGWorker build it, or a seed no longer names the same structure.
struct GeneratorInputs {
    View2D<RealType> body_positions_ang;
    View1D<RealType> cov_radii;
    View1D<RealType> vdw_radii;
    View1D<int> mol_n_sites;
    View1D<int> mol_site_offsets;
    View1D<RealType> mol_bounding_radius;
};

GeneratorInputs generator_inputs(const BasisConfig& basis) {
    const int n_mols = basis.n_mols;
    const int n_sites = basis.n_sites;
    const int total = n_mols * n_sites;

    GeneratorInputs in;
    in.body_positions_ang = View2D<RealType>("gen_body_positions", total, 3);
    in.cov_radii = View1D<RealType>("gen_cov_radii", total);
    in.vdw_radii = View1D<RealType>("gen_vdw_radii", total);
    in.mol_n_sites = View1D<int>("gen_mol_n_sites", n_mols);
    in.mol_site_offsets = View1D<int>("gen_mol_site_offsets", n_mols);
    in.mol_bounding_radius = View1D<RealType>("gen_mol_radius", n_mols);

    auto h_bp = Kokkos::create_mirror_view(in.body_positions_ang);
    auto h_cr = Kokkos::create_mirror_view(in.cov_radii);
    auto h_vr = Kokkos::create_mirror_view(in.vdw_radii);
    auto h_mns = Kokkos::create_mirror_view(in.mol_n_sites);
    auto h_mso = Kokkos::create_mirror_view(in.mol_site_offsets);
    auto h_mR = Kokkos::create_mirror_view(in.mol_bounding_radius);

    std::vector<double> cov(total, 0.0), vdw(total, 0.0);
    for (int m = 0; m < n_mols; ++m) {
        const int ns = n_mols > 1 ? basis.mol_n_sites[m] : n_sites;
        for (int i = 0; i < ns; ++i) {
            const int idx = m * n_sites + i;
            const auto& el = element(basis.elements[idx]);
            cov[idx] = std::max(0.68, static_cast<double>(el.cov_radius));
            vdw[idx] = el.vdw_radius;
        }
    }
    for (int i = 0; i < total; ++i) {
        for (int d = 0; d < 3; ++d) h_bp(i, d) = basis.body_positions[i][d] * BOHR_TO_ANGSTROM;
        h_cr(i) = cov[i];
        h_vr(i) = vdw[i];
    }
    for (int m = 0; m < n_mols; ++m) {
        h_mns(m) = n_mols > 1 ? basis.mol_n_sites[m] : n_sites;
        h_mso(m) = m * n_sites;
        const int ns_m = h_mns(m);
        const int off = h_mso(m);
        double cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < ns_m; ++i) {
            cx += h_bp(off + i, 0);
            cy += h_bp(off + i, 1);
            cz += h_bp(off + i, 2);
        }
        cx /= ns_m; cy /= ns_m; cz /= ns_m;
        double R_max = 0;
        for (int i = 0; i < ns_m; ++i) {
            const double dx = h_bp(off + i, 0) - cx;
            const double dy = h_bp(off + i, 1) - cy;
            const double dz = h_bp(off + i, 2) - cz;
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz) + h_vr(off + i);
            R_max = std::max(R_max, r);
        }
        h_mR(m) = static_cast<RealType>(R_max);
    }
    Kokkos::deep_copy(in.body_positions_ang, h_bp);
    Kokkos::deep_copy(in.cov_radii, h_cr);
    Kokkos::deep_copy(in.vdw_radii, h_vr);
    Kokkos::deep_copy(in.mol_n_sites, h_mns);
    Kokkos::deep_copy(in.mol_site_offsets, h_mso);
    Kokkos::deep_copy(in.mol_bounding_radius, h_mR);
    return in;
}

struct Drawn {
    int seed = 0;
    std::array<double, 6> cell{};   // Bohr, degrees
    std::vector<double> translations;
    std::vector<double> rotations;
};

}  // namespace

std::vector<GeneratedStructure> generate_structures(const std::vector<json>& basis_docs,
                                                    const GenerateOptions& o) {
    if (basis_docs.empty())
        throw std::invalid_argument("generating structures needs a basis");
    if (o.space_group < 1 || o.space_group > 230)
        throw std::invalid_argument(
            fmt::format("space group {} is not one of 1 to 230", o.space_group));
    if (o.count < 1)
        throw std::invalid_argument(fmt::format("count must be at least 1; got {}", o.count));
    if (o.seed < 0 || o.max_seeds < 0)
        throw std::invalid_argument("seed and max_seeds cannot be negative");
    const GeneratorConfig& g = o.generator;
    if (g.collision == CollisionMethod::RIGID_PRESS)
        throw std::invalid_argument(
            "rigid_press relaxes generated structures rather than generating them; "
            "generate with direct or fft_pack and relax what comes back");
    const bool fft = g.collision == CollisionMethod::FFT_PACK;
    if (fft && basis_docs.size() > 1)
        throw std::invalid_argument(
            "fft_pack places a single molecule type; a co-crystal needs the direct generator");

    Workspace ws = allocate_workspace(512, 27 * 27 * 27, 50000, 32, 4);
    const BasisConfig basis = basis_docs.size() > 1
                                  ? load_basis_documents(basis_docs, ws)
                                  : load_basis_document(basis_docs.front(), ws);
    const GeneratorInputs in = generator_inputs(basis);
    const int n_indep = basis.n_mols;

    // One seed yields at most one structure, or up to k for fft_pack, where a
    // seed is a trial owning a block of k identities.
    const int per_seed = fft ? std::max(1, g.fft_k_per_trial) : 1;
    const int max_seeds = o.max_seeds > 0 ? o.max_seeds : std::max(1000, 100 * o.count);

    // The fft_pack search starts at the trial owning o.seed and drops the block's
    // structures before it.
    std::vector<Drawn> kept;
    int next = o.seed / per_seed;
    int tried = 0;
    while (static_cast<int>(kept.size()) < o.count && tried < max_seeds) {
        const int wanted = o.count - static_cast<int>(kept.size());
        const int n_try = std::min(max_seeds - tried,
                                   std::max(16, fft ? wanted / per_seed + 1 : 2 * wanted));
        const int capacity = n_try * per_seed;

        DofBatch batch;
        batch.cell_params = View2D<RealType>("gen_cell_params", capacity, 6);
        batch.translations = View3D<RealType>("gen_translations", capacity, n_indep, 3);
        batch.rot_vectors = View3D<RealType>("gen_rotations", capacity, n_indep, 3);
        batch.seeds = View1D<int>("gen_seeds", capacity);
        batch.capacity = capacity;
        batch.n_indep = n_indep;
        const GeneratorRequest request{o.space_group, next, n_try};

        int n = 0;
        if (fft) {
            FftGenerator gen;
            gen.body_positions_ang = in.body_positions_ang;
            gen.cov_radii = in.cov_radii;
            gen.vdw_radii = in.vdw_radii;
            gen.n_sites = basis.n_sites;
            gen.config.tvp = g.tvp;
            gen.config.collision_buffer = 0.5;
            gen.config.k_per_trial = g.fft_k_per_trial;
            gen.config.grid_spacing = g.fft_grid_spacing;
            gen.config.collision_tol = g.fft_collision_tol;
            gen.config.max_aspect = g.fft_max_aspect;
            n = gen.generate(request, batch);
        } else {
            QuasiRandomGenerator gen;
            gen.body_positions_ang = in.body_positions_ang;
            gen.cov_radii = in.cov_radii;
            gen.vdw_radii = in.vdw_radii;
            gen.n_sites = basis.n_sites;
            gen.tvp = g.tvp;
            gen.collision_buffer = 0.5;
            gen.collision_method = g.collision;
            gen.mol_n_sites = in.mol_n_sites;
            gen.mol_site_offsets = in.mol_site_offsets;
            gen.mol_bounding_radius = in.mol_bounding_radius;
            gen.nudge_budget = g.nudge_budget;
            gen.contract = g.contract;
            n = gen.generate(request, batch);
        }
        next += n_try;
        tried += n_try;

        const auto h_cp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.cell_params);
        const auto h_t = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.translations);
        const auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.rot_vectors);
        const auto h_s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.seeds);
        std::vector<Drawn> block;
        block.reserve(n);
        for (int i = 0; i < n; ++i) {
            Drawn d;
            d.seed = h_s(i);
            for (int k = 0; k < 6; ++k) d.cell[k] = h_cp(i, k);
            for (int k = 0; k < n_indep; ++k)
                for (int c = 0; c < 3; ++c) {
                    d.translations.push_back(h_t(i, k, c));
                    d.rotations.push_back(h_r(i, k, c));
                }
            block.push_back(std::move(d));
        }
        std::sort(block.begin(), block.end(),
                  [](const Drawn& a, const Drawn& b) { return a.seed < b.seed; });
        for (auto& d : block) {
            if (static_cast<int>(kept.size()) == o.count) break;
            if (d.seed < o.seed) continue;
            kept.push_back(std::move(d));
        }
    }

    std::vector<GeneratedStructure> out;
    out.reserve(kept.size());
    for (const Drawn& d : kept) {
        json doc = build_crystal_json(basis, o.space_group, d.cell.data(), d.translations.data(),
                                      d.rotations.data(), n_indep);
        doc["title"] = fmt::format("seed {} in space group {}", d.seed, o.space_group);
        json structure = normalize_structure_json(doc);
        structure["seed"] = d.seed;
        out.push_back({d.seed, std::move(structure)});
    }
    return out;
}

}  // namespace klasp
