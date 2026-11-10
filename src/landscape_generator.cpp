#include "klasp/landscape_generator.hpp"
#include "klasp/db_tool.hpp"

#include <algorithm>
#include <tuple>

namespace klasp {

LandscapeGenerator load_landscape_generator(const std::string& db_path,
                                            const LandscapeGeneratorOptions& o) {
    LandscapeGenerator g;
    // filter_sg 0: read every space group once; requests select later.
    g.rows = read_landscape(db_path, o.stage, /*filter_sg=*/0, o.max_energy,
                            o.unique_only, o.exclude_catastrophes,
                            o.catastrophe_threshold, o.exclude_warned,
                            /*only_missing_stage=*/0);
    // Re-sort by (space_group, seed) so a seed window is contiguous.
    std::sort(g.rows.begin(), g.rows.end(),
              [](const StructureResult& a, const StructureResult& b) {
                  return std::tie(a.space_group, a.seed) <
                         std::tie(b.space_group, b.seed);
              });
    return g;
}

int LandscapeGenerator::seed_end(int space_group) const {
    int end = 0;
    for (const auto& r : rows)
        if (r.space_group == space_group) end = std::max(end, r.seed + 1);
    return end;
}

int LandscapeGenerator::generate(const GeneratorRequest& req, DofBatch& out) const {
    if (out.capacity <= 0 || req.n_try <= 0) return 0;
    const int seed_last = req.seed_start + req.n_try;   // half-open

    auto h_cp = Kokkos::create_mirror_view(out.cell_params);
    auto h_tr = Kokkos::create_mirror_view(out.translations);
    auto h_rv = Kokkos::create_mirror_view(out.rot_vectors);
    auto h_sd = Kokkos::create_mirror_view(out.seeds);

    int n = 0;
    for (const auto& r : rows) {
        if (n >= out.capacity) break;
        if (r.space_group != req.space_group) continue;
        if (r.seed < req.seed_start || r.seed >= seed_last) continue;
        if (r.n_mols != out.n_indep) continue;
        for (int j = 0; j < 6; ++j)
            h_cp(n, j) = static_cast<RealType>(r.cell_params[j]);
        for (int m = 0; m < out.n_indep; ++m) {
            for (int d = 0; d < 3; ++d) {
                h_tr(n, m, d) = static_cast<RealType>(r.translation[m * 3 + d]);
                h_rv(n, m, d) = static_cast<RealType>(r.rotation[m * 3 + d]);
            }
        }
        h_sd(n) = r.seed;   // the stored identity, never a fresh one
        ++n;
    }
    if (n == 0) return 0;

    // Copy only the rows written; the rest of the mirror is undefined.
    const auto rowsn = std::make_pair(0, n);
    Kokkos::deep_copy(Kokkos::subview(out.cell_params, rowsn, Kokkos::ALL),
                      Kokkos::subview(h_cp, rowsn, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(out.translations, rowsn, Kokkos::ALL, Kokkos::ALL),
                      Kokkos::subview(h_tr, rowsn, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(out.rot_vectors, rowsn, Kokkos::ALL, Kokkos::ALL),
                      Kokkos::subview(h_rv, rowsn, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(out.seeds, rowsn),
                      Kokkos::subview(h_sd, rowsn));
    return n;
}

}  // namespace klasp
