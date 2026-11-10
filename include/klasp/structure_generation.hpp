#pragma once

// The CSP generator on its own: a basis and a space group in, structure
// documents (asymmetric unit plus space-group operations) out, nothing written.
// A seed gives the same structure as in `klasp csp`.

#include "klasp/config.hpp"

#include <nlohmann/json.hpp>

#include <vector>

namespace klasp {

struct GenerateOptions {
    int space_group = 1;
    int count = 1;          // valid structures to return
    int seed = 0;           // the first structure seed returned, if it is valid
    // Seeds to try before returning fewer than `count` (for fft_pack, trials of
    // fft_k_per_trial seeds each). 0 means 100 per structure, at least 1000.
    int max_seeds = 0;
    // RIGID_PRESS is refused; FFT_PACK places a single molecule type.
    GeneratorConfig generator;
};

struct GeneratedStructure {
    int seed = 0;
    nlohmann::json structure;   // basis and crystal, with "symmetry", "space_group" and "seed"
};

// One document per structure, in seed order. Fewer than opts.count only when
// opts.max_seeds ran out first. `basis` holds one CSP basis document
// ({"basis": {...}}) per molecule type, in the order a co-crystal run lists them.
std::vector<GeneratedStructure> generate_structures(const std::vector<nlohmann::json>& basis,
                                                    const GenerateOptions& opts);

}  // namespace klasp
