#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

#include "klasp/csp.hpp"  // MAX_INDEP
#include "klasp/powder.hpp"  // POWDER_N_BINS

namespace klasp {

// Fixed-size result message sent from workers to rank 0 consumer.
static constexpr int AMD_K = 100;

// The pipeline stage a row describes. Stored on every structure row and part of
// the dedup key, so rows from two stages never compare. The numbers are a
// database contract.
enum Stage : int {
    STAGE_GENERATED = 0,  // as generated, before any relaxation
    STAGE_CHARGES   = 1,  // charge-only relaxation; the last stage when no multipoles run
    STAGE_MULTIPOLE = 2,  // full multipole relaxation
    STAGE_FINAL     = 3,  // final refinement, explicit non-rigid geometry (currently PET-MAD)
};

struct StructureResult {
    int seed;
    int space_group;
    int stage;
    int n_mols = 1;         // number of independent molecules
    double energy;
    double density;
    double time_ms;         // processing time for this structure
    double cell_params[6];
    double translation[MAX_INDEP * 3];  // fractional COM per molecule (flat)
    double rotation[MAX_INDEP * 3];     // angle-axis per molecule (flat)
    double amd[AMD_K];
    float  powder[POWDER_N_BINS];  // area-normalized X-ray powder pattern (float to fit in MPI message)
    // Per-structure convergence diagnostics (sentinels unless stored). grad_* are
    // the final gradient max/RMS over DOFs (a.u.); iters_to_converge is the
    // iteration it first met grad_tol (-1 = never).
    float grad_max = -1.0f;
    float grad_rms = -1.0f;
    int   iters_to_converge = -1;
    // Trust-region exit state (see BatchDiagnostics). final_delta < 0 when the
    // optimiser was not the trust region.
    float final_delta = -1.0f;
    int   n_restarts = 0;
    // Stage-1 (charge-only) convergence for the same structure.
    float s1_grad_max = -1.0f;
    int   s1_iters_to_converge = -1;
    // StructureWarning bits derived from the stored geometry; 0 is normal. See
    // klasp/warnings.hpp.
    int   warnings = 0;
    // Largest vacuum gap over low-index lattice planes, Angstrom; -1 when not
    // computed. See klasp/vacuum.hpp.
    double vacuum_gap = -1.0;
};

// Online structure deduplication config. The pipeline is sequential: energy
// bin, density bin, AMD L-inf, aligned PDD, powder cosine, powder CDTW. Each
// stage is skipped by setting its threshold/flag appropriately.
//
// Defaults follow mol-cspy's `cdtw_cos` method (CDTW decides, earlier stages
// prefilter), with thresholds calibrated for klasp's 5-50 deg, 1024-bin patterns.
struct DedupConfig {
    double energy_bin = 5.0;           // kJ/mol
    double density_bin = 0.05;         // g/cm³
    double amd_tol = 1.0;              // Å, L∞, coarse prefilter. <=0 disables.

    bool   use_powder = true;          // enable cosine + (optional) CDTW
    double powder_cos_thresh = 0.80;   // prefilter: duplicate candidate if cos_sim >= thresh
    double powder_cdtw_thresh = 10.0;  // decides: duplicate if min warp dist <= thresh; <=0 disables
    int    powder_cdtw_band = 10;      // Sakoe-Chiba band range (iterative r=0..band-1)

    // Omit the space group from the bin keys, clustering across all SGs as
    // cspy-db does. Offline prune sets this; the online CSP path does not.
    bool   cross_sg = false;

    // Aligned-PDD stage (after AMD, before powder): RMS over every entry of the
    // per-asym-atom k-NN distance matrix, rows in fixed atom order. <=0 disables.
    // pdd_rows = n_asym, pdd_k = neighbours per row (= AMD_K); both fixed per run.
    double pdd_tol = 0.0;   // Å
    int    pdd_rows = 0;
    int    pdd_k = 0;
    // One per molecule of the asymmetric unit: its first row and the row
    // relabellings its own symmetry allows, identity included (built by
    // pdd_symmetry_blocks). Each block takes its closest relabelling. Empty
    // compares rows in order.
    struct PddBlock {
        int first_row = 0;
        std::vector<std::vector<int>> perms;
    };
    std::vector<PddBlock> pdd_blocks;
};

// Online structure deduplication using energy/density binning, AMD
// (L∞/Chebyshev on the AMD_K vector), and optional powder-pattern
// refinement (cosine + Sakoe-Chiba CDTW). Thread-safe.
class StructureDedup {
public:
    struct Entry {
        int seed;
        int space_group;
        int stage = 0;
        double energy;
        double density;
        double amd[AMD_K];
        float  powder[POWDER_N_BINS];
        // Group norms of `powder`, filled by add_entry, for
        // powder_cosine_can_reach.
        PowderNorms powder_norms{};
        std::vector<double> pdd;   // n_asym*k, empty if PDD not stored
    };

    StructureDedup() = default;
    explicit StructureDedup(const DedupConfig& cfg) : cfg_(cfg) {}

    // Mutex makes the dedup non-assignable, so reconfigure in place.
    void configure(const DedupConfig& cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg_ = cfg;
    }

    // Returned by value under the lock; configure() may run concurrently.
    DedupConfig config() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cfg_;
    }

    // What classify() decided about a structure. A basin always keeps its
    // minimum-energy member as representative, independent of arrival order.
    // The displaced structure is identified by its full key (seed, SG, stage).
    struct Verdict {
        int duplicate_of = -1;   // representative's seed, or -1 if r IS the representative
        // The representative's space group. Seeds are per-SG, so with cross_sg
        // the seed alone does not identify the row.
        int duplicate_of_sg = -1;
        int demoted_seed = -1;   // structure r displaced, or -1 if it displaced nothing
        int demoted_sg = -1;
        int demoted_stage = -1;

        bool is_unique() const { return duplicate_of < 0; }
        // r took over an existing basin. The unique count is unchanged; the caller
        // must flip the demoted row in persistent storage.
        bool displaced() const { return demoted_seed >= 0; }
    };

    // Classify r against everything seen so far, adding it to the index if it is
    // a representative. Thread-safe. `pdd` (length n_asym*k) may be null when the
    // PDD is not stored; the aligned-PDD stage is then skipped.
    Verdict classify(const StructureResult& r,
                     const double* pdd = nullptr, int pdd_len = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (const Match m = find_match(r, pdd)) {
            const Entry& e = (*m.bin)[m.at];
            if (r.energy < e.energy) {
                // r is the better representative: evict the old member and file r
                // under its own bin. Net unique count is unchanged.
                Verdict v;
                v.demoted_seed  = e.seed;
                v.demoted_sg    = e.space_group;
                v.demoted_stage = e.stage;
                m.bin->erase(m.bin->begin() + m.at);
                add_entry(r, pdd, pdd_len);   // may rehash: m is dead after this
                return v;
            }
            Verdict v;
            v.duplicate_of = e.seed;
            v.duplicate_of_sg = e.space_group;
            return v;
        }
        add_entry(r, pdd, pdd_len);
        ++n_unique_;
        return Verdict{};
    }

    // Seed of the matching representative, or -1 if r is one. Never displaces;
    // callers must feed structures in ascending energy order (offline prune, tests).
    int find_duplicate_or_add(const StructureResult& r,
                              const double* pdd = nullptr, int pdd_len = 0) {
        return classify(r, pdd, pdd_len).duplicate_of;
    }

    // Has this basin been seen at all? Files r and returns false if not.
    // Unlike classify(), a lower-energy arrival does not displace the entry: this
    // answers whether a basin is covered, not which member represents it.
    bool seen_basin(const StructureResult& r,
                    const double* pdd = nullptr, int pdd_len = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (find_match(r, pdd)) return true;
        add_entry(r, pdd, pdd_len);
        ++n_unique_;
        return false;
    }

    // File r as a known representative without classifying it (resume path).
    // Supply only rows the database marks unique, in ascending energy order, as
    // load_landscape returns them.
    void seed_existing(const StructureResult& r,
                       const double* pdd = nullptr, int pdd_len = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        add_entry(r, pdd, pdd_len);
        ++n_unique_;
    }

    // True iff r and existing pass every enabled dedup stage. `rn` and `renv` are
    // r's powder group norms and CDTW envelope; null computes the norms here and
    // skips the envelope bound.
    bool matches(const StructureResult& r, const Entry& existing,
                 const double* pdd = nullptr,
                 const PowderNorms* rn = nullptr,
                 const PowderEnvelope* renv = nullptr) const {
        if (r.stage != existing.stage) return false;
        if (cfg_.amd_tol > 0) {
            if (amd_linf(r.amd, existing.amd) >= cfg_.amd_tol) return false;
        }
        if (cfg_.pdd_tol > 0 && pdd && !existing.pdd.empty() &&
            cfg_.pdd_rows > 0 && cfg_.pdd_k > 0) {
            if (!amd_can_reach_pdd(r.amd, existing.amd, cfg_.pdd_tol)) return false;
            if (!pdd_within(pdd, existing.pdd.data()))
                return false;
        }
        if (cfg_.use_powder) {
            PowderNorms local;
            if (!rn) { local = powder_group_norms(r.powder); rn = &local; }
            // Never rejects a pair the cosine would accept.
            if (!powder_cosine_can_reach(*rn, existing.powder_norms,
                                         cfg_.powder_cos_thresh))
                return false;
            const double cos_sim = powder_cosine_similarity(
                r.powder, existing.powder, POWDER_N_BINS);
            if (cos_sim < cfg_.powder_cos_thresh) return false;
            if (cfg_.powder_cdtw_thresh > 0) {
                if (renv && !powder_cdtw_can_reach(existing.powder, *renv, POWDER_N_BINS,
                                                   static_cast<float>(cfg_.powder_cdtw_thresh)))
                    return false;
                if (!powder_cdtw_within(r.powder, existing.powder,
                                        POWDER_N_BINS,
                                        cfg_.powder_cdtw_band,
                                        static_cast<float>(cfg_.powder_cdtw_thresh)))
                    return false;
            }
        }
        return true;
    }

    int n_unique() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return n_unique_;
    }

private:
    // Where a matching entry lives, so a caller can read it and (for classify)
    // erase it. Null bin means no match. Invalidated by add_entry.
    struct Match {
        std::vector<Entry>* bin = nullptr;
        size_t at = 0;
        explicit operator bool() const { return bin != nullptr; }
    };

    // The bin sweep shared by classify() and seen_basin(). Assumes the mutex is
    // held. Searches +/-1 bins because a basin can straddle a bin edge.
    Match find_match(const StructureResult& r, const double* pdd) {
        const int ebin = static_cast<int>(std::floor(r.energy / cfg_.energy_bin));
        const int dbin = static_cast<int>(std::floor(r.density / cfg_.density_bin));
        const PowderNorms rn =
            cfg_.use_powder ? powder_group_norms(r.powder) : PowderNorms{};
        const bool use_env = cfg_.use_powder && cfg_.powder_cdtw_thresh > 0;
        const PowderEnvelope renv = use_env
            ? powder_envelope(r.powder, std::max(0, cfg_.powder_cdtw_band - 1))
            : PowderEnvelope{};
        for (int de = -1; de <= 1; ++de) {
            for (int dd = -1; dd <= 1; ++dd) {
                auto it = bins_.find(make_key(r.space_group, r.stage, ebin + de, dbin + dd));
                if (it == bins_.end()) continue;
                auto& vec = it->second;
                for (size_t i = 0; i < vec.size(); ++i)
                    if (matches(r, vec[i], pdd, &rn, use_env ? &renv : nullptr))
                        return Match{&vec, i};
            }
        }
        return Match{};
    }

    // Build an index Entry from a result and file it under its own
    // (energy, density) bin. Assumes the mutex is held; does not touch
    // n_unique_ (callers manage the unique count).
    void add_entry(const StructureResult& r, const double* pdd, int pdd_len) {
        const int ebin = static_cast<int>(std::floor(r.energy / cfg_.energy_bin));
        const int dbin = static_cast<int>(std::floor(r.density / cfg_.density_bin));
        uint64_t key = make_key(r.space_group, r.stage, ebin, dbin);
        Entry e;
        e.seed = r.seed;
        e.space_group = r.space_group;
        e.stage = r.stage;
        e.energy = r.energy;
        e.density = r.density;
        std::memcpy(e.amd, r.amd, sizeof(e.amd));
        std::memcpy(e.powder, r.powder, sizeof(e.powder));
        e.powder_norms = powder_group_norms(e.powder);
        if (pdd && pdd_len > 0) e.pdd.assign(pdd, pdd + pdd_len);
        bins_[key].push_back(std::move(e));
    }

    // Chebyshev (L∞) distance, following Widdowson & Kurlin's
    // `average-minimum-distance` library default.
    static double amd_linf(const double* a, const double* b) {
        double m = 0;
        for (int i = 0; i < AMD_K; ++i) {
            double d = std::fabs(a[i] - b[i]);
            if (d > m) m = d;
        }
        return m;
    }

    // Is the PDD RMS distance below the tolerance? Stops early once the sum of
    // squares passes tol^2 * n. Each block takes its own closest relabelling.
    bool pdd_within(const double* a, const double* b) const {
        const int rows = cfg_.pdd_rows, k = cfg_.pdd_k;
        const double limit = cfg_.pdd_tol * cfg_.pdd_tol * static_cast<double>(rows) * k;
        const auto row_sq = [k](const double* x, const double* y) {
            double s = 0;
            for (int j = 0; j < k; ++j) {
                const double d = x[j] - y[j];
                s += d * d;
            }
            return s;
        };
        double total = 0;
        if (cfg_.pdd_blocks.empty()) {
            for (int i = 0; i < rows; ++i) {
                total += row_sq(a + static_cast<size_t>(i) * k, b + static_cast<size_t>(i) * k);
                if (total >= limit) return false;
            }
            return true;
        }
        for (const DedupConfig::PddBlock& blk : cfg_.pdd_blocks) {
            double best = limit - total;   // a block past this already fails
            for (const std::vector<int>& perm : blk.perms) {
                double s = 0;
                for (int r = 0; r < static_cast<int>(perm.size()) && s < best; ++r)
                    s += row_sq(a + static_cast<size_t>(blk.first_row + r) * k,
                                b + static_cast<size_t>(blk.first_row + perm[r]) * k);
                if (s < best) best = s;
            }
            total += best;
            if (total >= limit) return false;
        }
        return true;
    }

    // The AMD is the PDD's column mean, so by Jensen its RMS difference is at
    // most the PDD's. The slack covers single-precision AMD.
    static bool amd_can_reach_pdd(const double* a, const double* b, double tol) {
        constexpr double kSlack = 1e-5;   // Angstrom
        const double limit = (tol + kSlack) * (tol + kSlack) * AMD_K;
        double s = 0;
        for (int j = 0; j < AMD_K; ++j) {
            const double d = a[j] - b[j];
            s += d * d;
        }
        return s < limit;
    }

    // Stage is part of the key because energies at different stages are on
    // different scales. One byte each for SG (<=230) and stage.
    uint64_t make_key(int sg, int stage, int ebin, int dbin) const {
        const uint64_t sg_part = cfg_.cross_sg ? 0 : static_cast<uint64_t>(sg & 0xFF);
        return (static_cast<uint64_t>(stage & 0xFF) << 56)
             | (sg_part << 48)
             | (static_cast<uint64_t>(ebin & 0xFFFFFF) << 24)
             | static_cast<uint64_t>(dbin & 0xFFFFFF);
    }

    DedupConfig cfg_;
    int n_unique_ = 0;
    std::unordered_map<uint64_t, std::vector<Entry>> bins_;
    mutable std::mutex mutex_;
};

}  // namespace klasp
