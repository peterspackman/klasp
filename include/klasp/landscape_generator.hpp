#pragma once
// A generator that reads starting structures from an existing landscape database
// instead of drawing them from the prior. A request's window is a window of seeds,
// [seed_start, seed_start + n_try) within a space group, and rows keep their own
// seeds. Host-side.

#include "klasp/dedup.hpp"       // StructureResult
#include "klasp/generator.hpp"

#include <string>
#include <vector>

namespace klasp {

// Which database rows form the landscape. Defaults match `extract`: highest stage,
// catastrophes and warned structures excluded, all energies and duplicates kept.
struct LandscapeGeneratorOptions {
    int    stage = 0;                        // 0 = highest stage present
    double max_energy = 1e30;
    bool   unique_only = false;
    bool   exclude_catastrophes = true;
    double catastrophe_threshold = -1000.0;
    bool   exclude_warned = true;
};

struct LandscapeGenerator {
    // Sorted by (space_group, seed), so a seed window is a contiguous run.
    std::vector<StructureResult> rows;

    // Rows whose Z' does not match out.n_indep are skipped.
    int generate(const GeneratorRequest& req, DofBatch& out) const;

    // One past the largest seed held for a space group, so {sg, 0, seed_end(sg)}
    // covers it all. Zero when the space group is absent.
    int seed_end(int space_group) const;
};
static_assert(StructureGenerator<LandscapeGenerator>);

// Read a landscape into a generator. Throws as read_landscape does.
LandscapeGenerator load_landscape_generator(
    const std::string& db_path,
    const LandscapeGeneratorOptions& opts = LandscapeGeneratorOptions{});

}  // namespace klasp
