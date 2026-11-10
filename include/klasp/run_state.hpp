#pragma once
// Reading an existing CSP database back into memory. Host-only and Kokkos-free.
//   load_landscape - the structures with their descriptors.
//   probe_run      - a per-space-group summary (rows, uniques, highest seed),
//                    needed before the work counters are constructed on resume.

#include "klasp/dedup.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <map>
#include <vector>

namespace klasp {

// Which rows to read, and which descriptors to bring with them.
struct LandscapeFilter {
    // 0 = MAX(stage), resolved per file. Prefer an explicit value from
    // stage_histogram(): a partial top-K stage is also the maximum.
    int stage = 0;
    int space_group = 0;                  // 0 = every space group
    double max_energy = 1e30;
    double catastrophe_threshold = 0.0;
    bool exclude_catastrophes = false;
    bool unique_only = false;
    bool exclude_warned = false;          // drop rows flagged by warnings.hpp
    bool want_amd = true;
    bool want_powder = true;
    bool want_pdd = false;
};

// Row counts per stage across every file.
std::map<int, int> stage_histogram(const std::vector<std::string>& db_paths);

// Which stage a reader chose, and why. A pure function of the stage histogram.
struct StageChoice {
    int stage = 0;          // the stage to read
    int rows = 0;           // rows at that stage
    int fullest_stage = 0;  // the stage with the most rows
    int fullest_rows = 0;
    bool requested = false; // the caller named a stage rather than taking the default
    bool unknown = false;   // the named stage has no rows at all

    // Fraction of the fullest stage's rows this one has; 1.0 if they are the same.
    double coverage() const {
        return fullest_rows > 0 ? double(rows) / double(fullest_rows) : 1.0;
    }
    // An unrequested stage covering materially less of the landscape than another
    // (e.g. a top-K rerank, as opposed to a few dropped structures).
    bool silently_partial(double min_coverage = 0.90) const {
        return !requested && !unknown && coverage() < min_coverage;
    }
    bool slightly_short() const { return !unknown && rows < fullest_rows; }
};

// Choose the stage to read from a histogram. Never throws or prints.
StageChoice choose_stage(const std::map<int, int>& histogram, int requested);

// A landscape read back from disk. `rows` is sorted by ascending energy; callers
// depend on this so the first row in each basin is its representative.
struct Landscape {
    std::vector<StructureResult> rows;
    // Parallel to rows. A descriptor may be absent; callers must not compare against
    // missing ones.
    std::vector<std::uint8_t> has_amd;
    std::vector<std::uint8_t> has_powder;
    std::vector<std::vector<double>> pdd;   // empty entry = no PDD stored
    // Counted only for descriptors the filter requested.
    int n_missing_amd = 0;
    int n_missing_powder = 0;
    int n_missing_pdd = 0;
};

// Read one or more databases. The first file supplying a given (seed, space_group) wins.
Landscape load_landscape(const std::vector<std::string>& db_paths,
                         const LandscapeFilter& filter);

inline Landscape load_landscape(const std::string& db_path,
                                const LandscapeFilter& filter) {
    return load_landscape(std::vector<std::string>{db_path}, filter);
}

// One row of a streamed landscape. `pdd` is empty when none was stored; has_*
// flags mean present and well-formed, not requested.
struct LandscapeRow {
    StructureResult r{};
    bool has_amd = false;
    bool has_powder = false;
    std::vector<double> pdd;
};

using LandscapeRowFn = std::function<void(const LandscapeRow&)>;

// Stream one database row by row, in load_landscape's order (and tie-break),
// without holding the whole landscape. `fn` must not keep the reference: the row
// is reused.
void for_each_landscape_row(const std::string& db_path,
                            const LandscapeFilter& filter,
                            const LandscapeRowFn& fn);

struct SGProgress {
    int space_group = 0;
    int rows = 0;              // rows at the primary stage, i.e. structures
    int unique_rows = 0;
    // Highest seed recorded for this space group, or -1. A resume continues from
    // max_seed + 1.
    long long max_seed = -1;
};

// State of a run that already wrote into this output directory.
struct RunState {
    bool exists = false;             // a database was there at all
    int schema_version = 0;          // 0 = written before versioning
    std::string config_hash;         // empty if the database predates fingerprinting
    std::string config_json;
    std::vector<SGProgress> per_sg;  // one entry per requested space group, in order
    int total_rows = 0;

    const SGProgress* find(int sg) const;
};

// Read the summary. Every space group in `space_groups` gets an entry, zeroed if
// absent. A missing or unreadable file returns exists = false; never throws.
RunState probe_run(const std::string& db_path,
                   const std::vector<int>& space_groups,
                   int primary_stage);

// Schema version this binary writes; open() refuses anything newer.
// 3: descriptor table has a stage column.
// 4: duplicate_of_sg; displacement re-points the whole basin.
// 5: structure.vacuum_gap.
inline constexpr int KLASP_DB_SCHEMA_VERSION = 5;

// Non-cryptographic fingerprint of the settings two landscapes must share to be
// merged or resumed: everything that changes which structures exist, their
// energies, or dedup decisions. Excludes performance settings, the target and
// the space-group list.
std::string config_fingerprint(const std::string& canonical_json);

// Dotted paths of fields where two fingerprinted documents differ; empty if equal.
std::vector<std::string> config_differences(const std::string& a_json,
                                            const std::string& b_json);

}  // namespace klasp
