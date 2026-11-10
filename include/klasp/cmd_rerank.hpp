#pragma once

// Re-rank the top of a finished landscape with the PET model, writing stage-3
// rows. Declared only when the build has the ML backend.

#include <string>

namespace klasp {

struct RerankOptions {
    std::string db_path;
    std::string basis_path;         // optional: the DB usually embeds it

    // Short alias (xs, mols), canonical name, or path prefix; model_json and
    // model_weights override it.
    std::string model = "xs";
    std::string model_json;
    std::string model_weights;

    // The N lowest by force-field energy; 0 = all.
    int    top = 100;
    bool   symmetry = true;         // relax in the symmetry-adapted DOFs
    double force_tol = 1e-3;
    double stress_tol = 5.0e-6;
    int    max_iter = 300;
    // Source stage; 0 takes the highest present.
    int    from_stage = 2;
    double coarse_factor = 0.0;     // >0 runs a coarse pass first
    double min_contact = 0.7;       // reject candidates closer than this
    // Relax only rows without a stage-3 energy.
    bool   top_up = false;
    // KLASP_PET_NODEVICE still forces the host path.
    bool   device = true;
};

struct RerankResult {
    bool ok = false;
    int candidates = 0;    // structures admitted to the ML relaxation
    int relaxed = 0;       // stage-3 rows written
    int rejected = 0;      // candidates the clash screen turned away
};

RerankResult cmd_rerank(const RerankOptions& o);

}  // namespace klasp
