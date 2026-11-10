#include "klasp/run_state.hpp"
#include "klasp/csp.hpp"          // MAX_INDEP
#include "klasp/descriptors.hpp"  // AMD_K
#include "klasp/powder.hpp"       // POWDER_N_BINS
#include "klasp/db_open.hpp"
#include "klasp/db_schema.hpp"

#include <sqlite_modern_cpp.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <unordered_map>

namespace klasp {

namespace {

// (seed, space_group) key: sg in the low bits, exact for seeds below 2^40.
inline std::uint64_t row_key(int seed, int sg) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)) << 16)
         | static_cast<std::uint64_t>(sg & 0xFFFF);
}


// Fill a StructureResult from the `structure` columns both readers select. For
// Z' > 1 the rigid-body DOFs come from the `dofs` JSON blob.
inline void fill_row(StructureResult& r, int seed, int sg, int stage,
                     double energy, double density,
                     double a, double b, double c, double al, double be, double ga,
                     double tx, double ty, double tz, double rx, double ry, double rz,
                     int n_mols, const std::string* dofs_json) {
    r.seed = seed; r.space_group = sg; r.stage = stage;
    r.energy = energy; r.density = density;
    r.n_mols = std::max(1, n_mols);
    r.cell_params[0]=a;  r.cell_params[1]=b;  r.cell_params[2]=c;
    r.cell_params[3]=al; r.cell_params[4]=be; r.cell_params[5]=ga;
    r.translation[0]=tx; r.translation[1]=ty; r.translation[2]=tz;
    r.rotation[0]=rx;    r.rotation[1]=ry;    r.rotation[2]=rz;
    if (r.n_mols > 1 && dofs_json) {
        auto jd = nlohmann::json::parse(*dofs_json);
        auto& jt = jd["t"];
        auto& jr = jd["r"];
        for (int i = 0; i < r.n_mols * 3 && i < MAX_INDEP * 3; ++i) {
            r.translation[i] = jt[i].get<double>();
            r.rotation[i]    = jr[i].get<double>();
        }
    }
}

// Descriptor blob decoders. Each returns false and leaves the destination
// untouched when the blob is absent or the wrong length, so truncation reads as
// missing.
inline bool decode_amd_blob(const void* data, std::size_t bytes, double* amd) {
    if (bytes != AMD_K * sizeof(double)) return false;
    std::memcpy(amd, data, bytes);
    return true;
}

inline bool decode_powder_blob(const void* data, std::size_t bytes, float* powder) {
    return decode_powder(data, bytes, powder, POWDER_N_BINS);
}

// [int32 n_rows][int32 k][n_rows*k float64]; see insert_pdd.
inline bool decode_pdd_blob(const void* data, std::size_t bytes,
                            std::vector<double>& out) {
    if (bytes < 2 * sizeof(std::int32_t)) return false;
    std::int32_t hdr[2];
    std::memcpy(hdr, data, sizeof(hdr));
    const std::size_t n =
        static_cast<std::size_t>(std::max(0, hdr[0])) * std::max(0, hdr[1]);
    if (bytes != sizeof(hdr) + n * sizeof(double)) return false;
    out.resize(n);
    if (n > 0)
        std::memcpy(out.data(), static_cast<const char*>(data) + sizeof(hdr),
                    n * sizeof(double));
    return true;
}

// Columns both readers select from `structure`, in fill_row's order.
inline constexpr const char* kStructureColumns =
    "seed, space_group, stage, energy, density,"
    " a, b, c, alpha, beta, gamma, tx, ty, tz, rx, ry, rz, n_mols, dofs";

// WHERE clause shared by both readers, minus the stage test. Column names stay
// unqualified so they resolve in the streaming reader's outer FROM.
inline std::string filter_clause(sqlite::database& db, const LandscapeFilter& filter) {
    std::string q;
    if (filter.space_group > 0)
        q += fmt::format(" AND space_group = {}", filter.space_group);
    if (filter.max_energy < 1e29)
        q += fmt::format(" AND energy < {}", filter.max_energy);
    if (filter.exclude_catastrophes)
        q += fmt::format(" AND energy > {}", filter.catastrophe_threshold);
    if (filter.unique_only)
        q += " AND unique_structure = 1";
    q += warned_filter(db, filter.exclude_warned);
    return q;
}

}  // namespace

const SGProgress* RunState::find(int sg) const {
    for (const auto& p : per_sg)
        if (p.space_group == sg) return &p;
    return nullptr;
}

StageChoice choose_stage(const std::map<int, int>& hist, int requested) {
    StageChoice c;
    if (hist.empty()) { c.stage = requested; c.unknown = requested > 0; return c; }

    for (const auto& [stage, n] : hist)
        if (n > c.fullest_rows) { c.fullest_stage = stage; c.fullest_rows = n; }

    c.requested = requested > 0;
    c.stage = c.requested ? requested : hist.rbegin()->first;   // else highest present
    const auto it = hist.find(c.stage);
    if (it == hist.end()) { c.unknown = true; return c; }
    c.rows = it->second;
    return c;
}

std::map<int, int> stage_histogram(const std::vector<std::string>& db_paths) {
    std::map<int, int> hist;
    for (const auto& path : db_paths) {
        try {
            sqlite::database db = open_database_readonly(path);
            db << "SELECT stage, COUNT(*) FROM structure GROUP BY stage"
               >> [&](int stage, int n) { hist[stage] += n; };
        } catch (const std::exception&) {
        }
    }
    return hist;
}

void for_each_landscape_row(const std::string& db_path,
                            const LandscapeFilter& filter,
                            const LandscapeRowFn& fn) {
    sqlite::database db = open_database_readonly(db_path);

    const std::string stage_sel = filter.stage > 0
        ? std::to_string(filter.stage)
        : std::string("(SELECT MAX(stage) FROM structure)");

    // Descriptors come from correlated subqueries on descriptor's primary key.
    // ORDER BY must match load_landscape's sort, tie-break included.
    std::string query =
        std::string("SELECT ") + kStructureColumns +
        ", (SELECT value FROM descriptor WHERE descriptor.seed = structure.seed"
        "   AND descriptor.space_group = structure.space_group"
        "   AND descriptor.stage = structure.stage AND descriptor.name = 'amd')"
        ", (SELECT value FROM descriptor WHERE descriptor.seed = structure.seed"
        "   AND descriptor.space_group = structure.space_group"
        "   AND descriptor.stage = structure.stage AND descriptor.name = 'powder')"
        ", (SELECT value FROM descriptor WHERE descriptor.seed = structure.seed"
        "   AND descriptor.space_group = structure.space_group"
        "   AND descriptor.stage = structure.stage AND descriptor.name = 'pdd')"
        " FROM structure WHERE stage = " + stage_sel + filter_clause(db, filter) +
        " ORDER BY energy, space_group, seed";

    // One row reused for the whole stream.
    LandscapeRow row;
    db << query >> [&](int seed, int sg, int stage, double energy, double density,
                       double a, double b, double c, double al, double be, double ga,
                       double tx, double ty, double tz, double rx, double ry, double rz,
                       int n_mols, std::unique_ptr<std::string> dofs_json,
                       std::unique_ptr<std::vector<char>> amd_blob,
                       std::unique_ptr<std::vector<char>> powder_blob,
                       std::unique_ptr<std::vector<char>> pdd_blob) {
        row.r = StructureResult{};
        fill_row(row.r, seed, sg, stage, energy, density, a, b, c, al, be, ga,
                 tx, ty, tz, rx, ry, rz, n_mols, dofs_json.get());
        row.has_amd = amd_blob &&
            decode_amd_blob(amd_blob->data(), amd_blob->size(), row.r.amd);
        row.has_powder = powder_blob &&
            decode_powder_blob(powder_blob->data(), powder_blob->size(), row.r.powder);
        row.pdd.clear();
        if (pdd_blob)
            decode_pdd_blob(pdd_blob->data(), pdd_blob->size(), row.pdd);
        fn(row);
    };
}

Landscape load_landscape(const std::vector<std::string>& db_paths,
                         const LandscapeFilter& filter) {
    Landscape out;

    // Row index by (seed, space_group).
    std::unordered_map<std::uint64_t, int> index;

    const std::string stage_sel = filter.stage > 0
        ? std::to_string(filter.stage)
        : std::string("(SELECT MAX(stage) FROM structure)");

    // Pass 1: structures.
    for (const auto& path : db_paths) {
        sqlite::database db = open_database_readonly(path);
        const std::string query =
            std::string("SELECT ") + kStructureColumns +
            " FROM structure WHERE stage = " + stage_sel + filter_clause(db, filter);

        db << query >> [&](int seed, int sg, int stage, double energy, double density,
                           double a, double b, double c, double al, double be, double ga,
                           double tx, double ty, double tz, double rx, double ry, double rz,
                           int n_mols, std::unique_ptr<std::string> dofs_json) {
            const std::uint64_t k = row_key(seed, sg);
            if (index.count(k)) return;      // first file to supply a row wins
            StructureResult r{};
            fill_row(r, seed, sg, stage, energy, density, a, b, c, al, be, ga,
                     tx, ty, tz, rx, ry, rz, n_mols, dofs_json.get());
            index.emplace(k, static_cast<int>(out.rows.size()));
            out.rows.push_back(r);
        };
    }

    out.has_amd.assign(out.rows.size(), 0);
    out.has_powder.assign(out.rows.size(), 0);
    out.pdd.resize(out.rows.size());

    // Pass 2: descriptors, after the index covers every file.
    const bool need_desc = filter.want_amd || filter.want_powder || filter.want_pdd;
    if (need_desc) {
        for (const auto& path : db_paths) {
            sqlite::database db = open_database_readonly(path);
            auto place = [&](int seed, int sg) -> int {
                auto it = index.find(row_key(seed, sg));
                return it == index.end() ? -1 : it->second;
            };
            if (filter.want_amd) {
                db << ("SELECT seed, space_group, value FROM descriptor WHERE name = 'amd'"
                       " AND stage = " + stage_sel)
                   >> [&](int seed, int sg, std::vector<char> blob) {
                    const int i = place(seed, sg);
                    if (i < 0 || out.has_amd[i]) return;
                    if (decode_amd_blob(blob.data(), blob.size(), out.rows[i].amd))
                        out.has_amd[i] = 1;
                };
            }
            if (filter.want_powder) {
                db << ("SELECT seed, space_group, value FROM descriptor WHERE name = 'powder'"
                       " AND stage = " + stage_sel)
                   >> [&](int seed, int sg, std::vector<char> blob) {
                    const int i = place(seed, sg);
                    if (i < 0 || out.has_powder[i]) return;
                    if (decode_powder_blob(blob.data(), blob.size(), out.rows[i].powder))
                        out.has_powder[i] = 1;
                };
            }
            if (filter.want_pdd) {
                db << ("SELECT seed, space_group, value FROM descriptor WHERE name = 'pdd'"
                       " AND stage = " + stage_sel)
                   >> [&](int seed, int sg, std::vector<char> blob) {
                    const int i = place(seed, sg);
                    if (i < 0 || !out.pdd[i].empty()) return;
                    decode_pdd_blob(blob.data(), blob.size(), out.pdd[i]);
                };
            }
        }
    }

    // Count a descriptor as missing only when it was requested.
    for (std::size_t i = 0; i < out.rows.size(); ++i) {
        if (filter.want_amd    && !out.has_amd[i])    ++out.n_missing_amd;
        if (filter.want_powder && !out.has_powder[i]) ++out.n_missing_powder;
        if (filter.want_pdd    && out.pdd[i].empty()) ++out.n_missing_pdd;
    }

    // Ascending energy, carrying the parallel arrays along via an index permutation.
    std::vector<int> order(out.rows.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (out.rows[a].energy != out.rows[b].energy)
            return out.rows[a].energy < out.rows[b].energy;
        // Deterministic tie-break.
        if (out.rows[a].space_group != out.rows[b].space_group)
            return out.rows[a].space_group < out.rows[b].space_group;
        return out.rows[a].seed < out.rows[b].seed;
    });

    // Apply the permutation in place (cycle-following needs dest[perm[i]] = src[i],
    // so `order` is inverted first) to avoid a second copy of the landscape.
    std::vector<int> inv(order.size());
    for (std::size_t k = 0; k < order.size(); ++k) inv[order[k]] = static_cast<int>(k);
    for (std::size_t i = 0; i < inv.size(); ++i) {
        while (inv[i] != static_cast<int>(i)) {
            const int j = inv[i];
            std::swap(out.rows[i], out.rows[j]);
            std::swap(out.has_amd[i], out.has_amd[j]);
            std::swap(out.has_powder[i], out.has_powder[j]);
            std::swap(out.pdd[i], out.pdd[j]);
            std::swap(inv[i], inv[j]);
        }
    }
    return out;
}

std::string config_fingerprint(const std::string& canonical_json) {
    // FNV-1a, 64-bit.
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : canonical_json) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;
    }
    return fmt::format("{:016x}", h);
}

namespace {

// Flatten a JSON document to dotted leaf paths.
void flatten_json(const nlohmann::json& j, const std::string& prefix,
                  std::map<std::string, std::string>& out) {
    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it)
            flatten_json(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(), out);
    } else if (j.is_array()) {
        // Arrays compare whole.
        out[prefix] = j.dump();
    } else {
        out[prefix] = j.dump();
    }
}

}  // namespace

std::vector<std::string> config_differences(const std::string& a_json,
                                            const std::string& b_json) {
    std::vector<std::string> diffs;
    nlohmann::json a, b;
    try {
        a = nlohmann::json::parse(a_json);
        b = nlohmann::json::parse(b_json);
    } catch (const std::exception&) {
        diffs.push_back("<unreadable recorded config>");
        return diffs;
    }
    std::map<std::string, std::string> fa, fb;
    flatten_json(a, "", fa);
    flatten_json(b, "", fb);
    for (const auto& [k, v] : fa) {
        auto it = fb.find(k);
        if (it == fb.end()) diffs.push_back(k + " (removed)");
        else if (it->second != v)
            diffs.push_back(fmt::format("{}: {} -> {}", k, v, it->second));
    }
    for (const auto& [k, v] : fb) {
        (void)v;
        if (!fa.count(k)) diffs.push_back(k + " (added)");
    }
    return diffs;
}

RunState probe_run(const std::string& db_path,
                   const std::vector<int>& space_groups,
                   int primary_stage) {
    RunState st;
    st.per_sg.reserve(space_groups.size());
    for (int sg : space_groups) st.per_sg.push_back(SGProgress{sg, 0, 0, -1});

    std::error_code ec;
    if (!std::filesystem::exists(db_path, ec)) return st;

    try {
        sqlite::database db = open_database_readonly(db_path);
        // A file with no structure table is not a run; treat as absent.
        int has_table = 0;
        db << "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='structure'"
           >> has_table;
        if (has_table == 0) return st;

        st.exists = true;

        for (auto& p : st.per_sg) {
            db << "SELECT COUNT(*), SUM(unique_structure), MAX(seed) FROM structure "
                  "WHERE space_group = ? AND stage = ?"
               << p.space_group << primary_stage
               >> [&](int rows, std::unique_ptr<int> uniq, std::unique_ptr<long long> mx) {
                   p.rows = rows;
                   p.unique_rows = uniq ? *uniq : 0;
                   p.max_seed = mx ? *mx : -1;
               };
            st.total_rows += p.rows;
        }

        // The seed watermark is the maximum over every stage, so a resume never reissues
        // a seed any stage already used.
        for (auto& p : st.per_sg) {
            db << "SELECT MAX(seed) FROM structure WHERE space_group = ?"
               << p.space_group
               >> [&](std::unique_ptr<long long> mx) {
                   if (mx && *mx > p.max_seed) p.max_seed = *mx;
               };
        }

        auto meta = [&](const char* key) {
            std::string v;
            try { db << "SELECT value FROM meta WHERE key = ?" << key >> v; }
            catch (...) {}
            return v;
        };
        st.config_hash = meta("config_hash");
        st.config_json = meta("config_json");
        const std::string ver = meta("schema_version");
        st.schema_version = ver.empty() ? 0 : std::atoi(ver.c_str());
    } catch (const std::exception&) {
        // Unreadable, locked, or not SQLite. Reported as such; the caller decides.
        st.exists = false;
        st.schema_version = -1;
    }
    return st;
}

}  // namespace klasp
