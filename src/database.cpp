#include "klasp/log.hpp"
#include "klasp/database.hpp"
#include "klasp/powder.hpp"
#include "klasp/dedup.hpp"
#include <sqlite_modern_cpp.h>
#include <fmt/format.h>
#include <cstring>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace klasp {

struct CSPDatabase::Impl {
    std::unique_ptr<sqlite::database> db;
    int pending = 0;
    // Commit every N statements. Under WAL with synchronous=NORMAL a commit does
    // not fsync, so this only bounds the loss on a kill.
    static constexpr int BATCH_SIZE = 1000;

    // The write statements, prepared once. Binding to a slot after an execute
    // resets the statement and clears the old bindings.
    using Stmt = std::optional<sqlite::database_binder>;
    Stmt ins_structure, ins_descriptor, ins_flexible, demote_repoint, demote_row;

    // Bind and execute `slot`, preparing it on first use. On a throw the slot is
    // dropped so the next call starts clean; `used(true)` first so a failed
    // binder does not execute itself in its destructor.
    template <class Bind>
    void run(Stmt& slot, const char* sql, Bind&& bind) {
        if (!slot) slot.emplace(*db << sql);
        try {
            bind(*slot);
            slot->execute();
        } catch (...) {
            slot->used(true);
            slot.reset();
            throw;
        }
    }
};

CSPDatabase::CSPDatabase() : impl_(std::make_unique<Impl>()) {}
CSPDatabase::~CSPDatabase() { close(); }

void CSPDatabase::open(const std::string& path) {
    impl_->db = std::make_unique<sqlite::database>(path);
    auto& db = *impl_->db;

    // Wait on lock contention: an exception on the consumer thread is std::terminate.
    db << "PRAGMA busy_timeout=30000;";

    // WAL for concurrent reads. Parallel filesystems may not support it and SQLite
    // silently stays in rollback-journal mode, so read the mode back and report it.
    std::string journal_mode;
    db << "PRAGMA journal_mode=WAL;" >> journal_mode;
    if (journal_mode != "wal")
        logging::warn("\nWARNING: {} is in '{}' journal mode -- WAL was requested and the\n"
                   "filesystem refused it. Writes are slower and an interrupted run leaves a\n"
                   "hot journal rather than a write-ahead log.\n",
                   path, journal_mode);
    db << "PRAGMA synchronous=NORMAL;";
    // Larger page cache and less frequent automatic checkpoints than the defaults.
    db << "PRAGMA cache_size=-20000;";
    db << "PRAGMA wal_autocheckpoint=20000;";

    db << R"(
        CREATE TABLE IF NOT EXISTS structure (
            seed            INTEGER NOT NULL,
            space_group     INTEGER NOT NULL,
            stage           INTEGER NOT NULL,
            energy          REAL,
            density         REAL,
            converged       INTEGER NOT NULL DEFAULT 1,
            unique_structure INTEGER NOT NULL DEFAULT 1,
            duplicate_of    INTEGER DEFAULT NULL,
            n_mols          INTEGER NOT NULL DEFAULT 1,
            a               REAL NOT NULL,
            b               REAL NOT NULL,
            c               REAL NOT NULL,
            alpha           REAL NOT NULL,
            beta            REAL NOT NULL,
            gamma           REAL NOT NULL,
            tx              REAL NOT NULL,
            ty              REAL NOT NULL,
            tz              REAL NOT NULL,
            rx              REAL NOT NULL,
            ry              REAL NOT NULL,
            rz              REAL NOT NULL,
            dofs            TEXT DEFAULT NULL,
            grad_max        REAL DEFAULT NULL,
            grad_rms        REAL DEFAULT NULL,
            iters_to_converge INTEGER DEFAULT NULL,
            final_delta       REAL DEFAULT NULL,
            n_restarts        INTEGER DEFAULT NULL,
            s1_grad_max       REAL DEFAULT NULL,
            s1_iters_to_converge INTEGER DEFAULT NULL,
            warnings          INTEGER NOT NULL DEFAULT 0,
            duplicate_of_sg   INTEGER DEFAULT NULL,
            vacuum_gap        REAL DEFAULT NULL,
            PRIMARY KEY (seed, space_group, stage)
        );
    )";

    db << R"(
        CREATE TABLE IF NOT EXISTS meta (
            key     TEXT PRIMARY KEY,
            value   TEXT
        );
    )";

    db << R"(
        CREATE TABLE IF NOT EXISTS descriptor (
            seed            INTEGER NOT NULL,
            space_group     INTEGER NOT NULL,
            stage           INTEGER NOT NULL,
            name            TEXT NOT NULL,
            value           BLOB NOT NULL,
            PRIMARY KEY (seed, space_group, stage, name)
        );
    )";

    // Explicit-atom geometry for non-rigid structures (e.g. PET-relaxed stage 3),
    // one row per parent (seed, space_group, stage) structure row.
    //   atoms BLOB = [int32 Z * n_atoms][float64 frac * 3*n_atoms] (fractional
    //   coords w.r.t. the stored cell; a,b,c in Angstrom, angles in degrees).
    db << R"(
        CREATE TABLE IF NOT EXISTS flexible_structure (
            seed            INTEGER NOT NULL,
            space_group     INTEGER NOT NULL,
            stage           INTEGER NOT NULL,
            n_atoms         INTEGER NOT NULL,
            a               REAL NOT NULL,
            b               REAL NOT NULL,
            c               REAL NOT NULL,
            alpha           REAL NOT NULL,
            beta            REAL NOT NULL,
            gamma           REAL NOT NULL,
            atoms           BLOB NOT NULL,
            PRIMARY KEY (seed, space_group, stage)
        );
    )";

    // Migrate v2 -> v3: the descriptor table gains a stage column. Old rows are
    // copied to every non-zero stage of their structure.
    try {
        db << "SELECT stage FROM descriptor LIMIT 1;";
    } catch (const std::exception&) {
        db << "ALTER TABLE descriptor RENAME TO descriptor_v2;";
        db << R"(
            CREATE TABLE descriptor (
                seed            INTEGER NOT NULL,
                space_group     INTEGER NOT NULL,
                stage           INTEGER NOT NULL,
                name            TEXT NOT NULL,
                value           BLOB NOT NULL,
                PRIMARY KEY (seed, space_group, stage, name)
            );
        )";
        db << "INSERT OR REPLACE INTO descriptor (seed, space_group, stage, name, value) "
              "SELECT d.seed, d.space_group, s.stage, d.name, d.value "
              "FROM descriptor_v2 d JOIN structure s "
              "  ON s.seed = d.seed AND s.space_group = d.space_group "
              "WHERE s.stage > 0;";
        db << "DROP TABLE descriptor_v2;";
    }

    // Migrate v3 -> v4: duplicate_of gains a space group, backfilled with the
    // pointing row's own space group.
    try {
        db << "SELECT duplicate_of_sg FROM structure LIMIT 1;";
    } catch (const std::exception&) {
        db << "ALTER TABLE structure ADD COLUMN duplicate_of_sg INTEGER DEFAULT NULL;";
        db << "UPDATE structure SET duplicate_of_sg = space_group "
              "WHERE duplicate_of IS NOT NULL;";
        // Collapse duplicate_of chains so each reference names its representative.
        // One pass per hop, capped so a cycle cannot loop forever.
        for (int hop = 0; hop < 32; ++hop) {
            db << R"(
                UPDATE structure AS d
                SET duplicate_of = (SELECT t.duplicate_of FROM structure t
                                    WHERE t.seed = d.duplicate_of
                                      AND t.space_group = d.duplicate_of_sg
                                      AND t.stage = d.stage),
                    duplicate_of_sg = (SELECT t.duplicate_of_sg FROM structure t
                                       WHERE t.seed = d.duplicate_of
                                         AND t.space_group = d.duplicate_of_sg
                                         AND t.stage = d.stage)
                WHERE d.duplicate_of IS NOT NULL
                  AND EXISTS (SELECT 1 FROM structure t
                              WHERE t.seed = d.duplicate_of
                                AND t.space_group = d.duplicate_of_sg
                                AND t.stage = d.stage
                                AND t.unique_structure = 0
                                AND t.duplicate_of IS NOT NULL);
            )";
            int changed = 0;
            db << "SELECT changes();" >> changed;
            if (changed == 0) break;
        }
    }

    // Migrate v4 onwards: add any missing columns. ADD COLUMN cannot take a
    // non-constant default, so each is nullable or constant-defaulted;
    // vacuum_gap is NULL on rows written before it was measured.
    {
        std::set<std::string> have;
        db << "SELECT name FROM pragma_table_info('structure')"
           >> [&](std::string name) { have.insert(std::move(name)); };
        static constexpr std::pair<const char*, const char*> kColumns[] = {
            {"duplicate_of",         "INTEGER DEFAULT NULL"},
            {"n_mols",               "INTEGER NOT NULL DEFAULT 1"},
            {"dofs",                 "TEXT DEFAULT NULL"},
            {"grad_max",             "REAL DEFAULT NULL"},
            {"grad_rms",             "REAL DEFAULT NULL"},
            {"iters_to_converge",    "INTEGER DEFAULT NULL"},
            {"final_delta",          "REAL DEFAULT NULL"},
            {"s1_grad_max",          "REAL DEFAULT NULL"},
            {"s1_iters_to_converge", "INTEGER DEFAULT NULL"},
            {"n_restarts",           "INTEGER DEFAULT NULL"},
            {"warnings",             "INTEGER NOT NULL DEFAULT 0"},
            {"vacuum_gap",           "REAL DEFAULT NULL"},
        };
        for (const auto& [name, decl] : kColumns)
            if (!have.count(name))
                db << fmt::format("ALTER TABLE structure ADD COLUMN {} {};", name, decl);
    }

    db << "CREATE INDEX IF NOT EXISTS idx_sg_stage ON structure(space_group, stage);";

    // Partial index for demote_to_duplicate's search on duplicate_of; SQLite
    // infers duplicate_of=? implies IS NOT NULL and still uses it.
    db << "CREATE INDEX IF NOT EXISTS idx_dup ON structure"
          "(duplicate_of, duplicate_of_sg, stage) WHERE duplicate_of IS NOT NULL;";

    db << "BEGIN;";
}


void CSPDatabase::flush() {
    if (impl_->db && impl_->pending > 0) {
        *impl_->db << "COMMIT;";
        *impl_->db << "BEGIN;";
        impl_->pending = 0;
    }
}

// Fold the write-ahead log back into the database file, so csp.db is complete
// without its -wal. PASSIVE never blocks a writer.
void CSPDatabase::checkpoint(bool truncate) {
    if (!impl_->db) return;
    flush();
    try {
        *impl_->db << (truncate ? "PRAGMA wal_checkpoint(TRUNCATE);"
                                : "PRAGMA wal_checkpoint(PASSIVE);");
    } catch (const std::exception&) {
        // A checkpoint is an optimisation; the data is committed either way.
    }
}

void CSPDatabase::set_meta(const std::string& key, const std::string& value) {
    *impl_->db << "INSERT OR REPLACE INTO meta VALUES (?,?)" << key << value;
    // Counted so the next flush commits it.
    ++impl_->pending;
}

void CSPDatabase::insert_result(const void* result_ptr, bool is_unique, int duplicate_of,
                                int duplicate_of_sg) {
    auto& r = *static_cast<const StructureResult*>(result_ptr);
    std::optional<int> dup_opt = duplicate_of >= 0
        ? std::optional<int>(duplicate_of) : std::nullopt;
    std::optional<int> dup_sg_opt = duplicate_of >= 0
        ? std::optional<int>(duplicate_of_sg >= 0 ? duplicate_of_sg : r.space_group)
        : std::nullopt;

    // Multi-molecule DOFs go in the dofs column as JSON.
    std::optional<std::string> dofs_json;
    if (r.n_mols > 1) {
        std::string s = fmt::format("{{\"n_mols\":{},\"t\":[", r.n_mols);
        for (int i = 0; i < r.n_mols * 3; ++i) {
            if (i > 0) s += ",";
            s += fmt::format("{:.15g}", r.translation[i]);
        }
        s += "],\"r\":[";
        for (int i = 0; i < r.n_mols * 3; ++i) {
            if (i > 0) s += ",";
            s += fmt::format("{:.15g}", r.rotation[i]);
        }
        s += "]}";
        dofs_json = std::move(s);
    }

    // Convergence diagnostics: grad_max < 0 means not recorded (stored as NULL),
    // in which case `converged` stays 1.
    const bool have_diag = r.grad_max >= 0.0f;
    std::optional<double> gmax_opt = have_diag ? std::optional<double>(r.grad_max) : std::nullopt;
    std::optional<double> grms_opt = have_diag ? std::optional<double>(r.grad_rms) : std::nullopt;
    std::optional<int> iters_opt = have_diag ? std::optional<int>(r.iters_to_converge) : std::nullopt;
    const bool have_s1 = r.s1_grad_max >= 0.0f;
    std::optional<double> s1g_opt = have_s1 ? std::optional<double>(r.s1_grad_max) : std::nullopt;
    std::optional<int> s1i_opt = have_s1 ? std::optional<int>(r.s1_iters_to_converge) : std::nullopt;
    const bool have_tr = r.final_delta >= 0.0f;
    std::optional<double> delta_opt = have_tr ? std::optional<double>(r.final_delta) : std::nullopt;
    std::optional<int> restarts_opt = have_tr ? std::optional<int>(r.n_restarts) : std::nullopt;
    const int converged = have_diag ? (r.iters_to_converge >= 0 ? 1 : 0) : 1;

    const std::optional<double> vac_opt =
        r.vacuum_gap >= 0.0 ? std::optional<double>(r.vacuum_gap) : std::nullopt;
    impl_->run(impl_->ins_structure,
        "INSERT OR REPLACE INTO structure "
        "(seed, space_group, stage, energy, density, converged, "
        "unique_structure, duplicate_of, n_mols, "
        "a, b, c, alpha, beta, gamma, tx, ty, tz, rx, ry, rz, dofs, "
        "grad_max, grad_rms, iters_to_converge, final_delta, n_restarts, "
        "s1_grad_max, s1_iters_to_converge, warnings, duplicate_of_sg, vacuum_gap) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [&](sqlite::database_binder& st) {
            st << r.seed << r.space_group << r.stage
               << r.energy << r.density << converged << (is_unique ? 1 : 0)
               << dup_opt << r.n_mols
               << r.cell_params[0] << r.cell_params[1] << r.cell_params[2]
               << r.cell_params[3] << r.cell_params[4] << r.cell_params[5]
               << r.translation[0] << r.translation[1] << r.translation[2]
               << r.rotation[0] << r.rotation[1] << r.rotation[2]
               << dofs_json
               << gmax_opt << grms_opt << iters_opt << delta_opt << restarts_opt
               << s1g_opt << s1i_opt << r.warnings << dup_sg_opt << vac_opt;
        });
    if (++impl_->pending >= Impl::BATCH_SIZE) flush();
}

void CSPDatabase::demote_to_duplicate(int old_seed, int space_group, int stage,
                                      int new_seed, int new_sg) {
    // Re-point duplicates of the displaced row before demoting it, so
    // duplicate_of always names a representative.
    impl_->run(impl_->demote_repoint,
        "UPDATE structure SET duplicate_of=?, duplicate_of_sg=? "
        "WHERE duplicate_of=? AND duplicate_of_sg=? AND stage=?",
        [&](sqlite::database_binder& st) {
            st << new_seed << new_sg << old_seed << space_group << stage;
        });
    impl_->run(impl_->demote_row,
        "UPDATE structure SET unique_structure=0, duplicate_of=?, duplicate_of_sg=? "
        "WHERE seed=? AND space_group=? AND stage=?",
        [&](sqlite::database_binder& st) {
            st << new_seed << new_sg << old_seed << space_group << stage;
        });
    impl_->pending += 2;
    if (impl_->pending >= Impl::BATCH_SIZE) flush();
}

void CSPDatabase::set_verdict(int seed, int space_group, int stage, int duplicate_of,
                              int duplicate_of_sg) {
    const std::optional<int> dup = duplicate_of >= 0 ? std::optional<int>(duplicate_of)
                                                     : std::nullopt;
    const std::optional<int> dup_sg = duplicate_of >= 0
        ? std::optional<int>(duplicate_of_sg >= 0 ? duplicate_of_sg : space_group)
        : std::nullopt;
    *impl_->db << "UPDATE structure SET unique_structure=?, duplicate_of=?, duplicate_of_sg=? "
                  "WHERE seed=? AND space_group=? AND stage=?"
               << (duplicate_of < 0 ? 1 : 0) << dup << dup_sg << seed << space_group << stage;
    if (++impl_->pending >= Impl::BATCH_SIZE) flush();
}

std::vector<char> CSPDatabase::encode_flexible_atoms(const FlexibleStructure& fs) {
    const int n = fs.n_atoms;
    std::vector<char> blob(static_cast<std::size_t>(n) * sizeof(std::int32_t)
                           + 3ull * n * sizeof(double));
    const std::vector<std::int32_t> z32(fs.Z.begin(), fs.Z.begin() + n);
    std::memcpy(blob.data(), z32.data(), n * sizeof(std::int32_t));
    std::memcpy(blob.data() + n * sizeof(std::int32_t), fs.frac.data(),
                3ull * n * sizeof(double));
    return blob;
}

bool CSPDatabase::decode_flexible_atoms(int n_atoms, const void* data, std::size_t bytes,
                                        FlexibleStructure& out) {
    const std::size_t want = static_cast<std::size_t>(n_atoms) * sizeof(std::int32_t)
                           + 3ull * n_atoms * sizeof(double);
    if (n_atoms <= 0 || bytes != want) return false;
    const char* p = static_cast<const char*>(data);
    std::vector<std::int32_t> z32(n_atoms);
    std::memcpy(z32.data(), p, n_atoms * sizeof(std::int32_t));
    out.n_atoms = n_atoms;
    out.Z.assign(z32.begin(), z32.end());
    out.frac.resize(3ull * n_atoms);
    std::memcpy(out.frac.data(), p + n_atoms * sizeof(std::int32_t),
                3ull * n_atoms * sizeof(double));
    return true;
}

void CSPDatabase::insert_flexible(int seed, int space_group, int stage,
                                  const FlexibleStructure& fs) {
    const int n = fs.n_atoms;
    const std::vector<char> blob = encode_flexible_atoms(fs);
    impl_->run(impl_->ins_flexible,
        "INSERT OR REPLACE INTO flexible_structure VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [&](sqlite::database_binder& st) {
            st << seed << space_group << stage << n
               << fs.cell_params[0] << fs.cell_params[1] << fs.cell_params[2]
               << fs.cell_params[3] << fs.cell_params[4] << fs.cell_params[5]
               << blob;
        });
    if (++impl_->pending >= Impl::BATCH_SIZE) flush();
}

bool CSPDatabase::read_flexible(int seed, int space_group, int stage,
                                FlexibleStructure& out) {
    auto& db = *impl_->db;
    bool found = false;
    db << "SELECT n_atoms, a, b, c, alpha, beta, gamma, atoms FROM flexible_structure"
          " WHERE seed = ? AND space_group = ? AND stage = ?"
       << seed << space_group << stage
       >> [&](int n_atoms, double a, double b, double c, double al, double be,
              double ga, std::vector<char> blob) {
           if (!decode_flexible_atoms(n_atoms, blob.data(), blob.size(), out)) return;
           out.cell_params[0] = a;  out.cell_params[1] = b;  out.cell_params[2] = c;
           out.cell_params[3] = al; out.cell_params[4] = be; out.cell_params[5] = ga;
           found = true;
       };
    return found;
}

// One prepared statement for the three descriptor rows.
void CSPDatabase::insert_descriptor(int seed, int space_group, int stage,
                                    const char* name, const std::vector<char>& blob) {
    impl_->run(impl_->ins_descriptor,
        "INSERT OR REPLACE INTO descriptor VALUES (?,?,?,?,?)",
        [&](sqlite::database_binder& st) {
            st << seed << space_group << stage << name << blob;
        });
    if (++impl_->pending >= Impl::BATCH_SIZE) flush();
}

void CSPDatabase::insert_amd(int seed, int space_group, int stage,
                             const std::vector<double>& amd) {
    std::vector<char> blob(amd.size() * sizeof(double));
    std::memcpy(blob.data(), amd.data(), blob.size());
    insert_descriptor(seed, space_group, stage, "amd", blob);
}

void CSPDatabase::insert_powder(int seed, int space_group, int stage,
                                const float* pattern, int n_bins) {
    // Sparse or dense, lossless either way (see powder.hpp).
    const std::vector<char> blob = encode_powder(pattern, n_bins);
    insert_descriptor(seed, space_group, stage, "powder", blob);
}

void CSPDatabase::insert_pdd(int seed, int space_group, int stage, const double* pdd,
                             int n_rows, int k) {
    // Blob: [int32 n_rows][int32 k][n_rows*k float64 distances]
    const size_t n = static_cast<size_t>(n_rows) * k;
    std::vector<char> blob(2 * sizeof(std::int32_t) + n * sizeof(double));
    std::int32_t hdr[2] = {n_rows, k};
    std::memcpy(blob.data(), hdr, sizeof(hdr));
    std::memcpy(blob.data() + sizeof(hdr), pdd, n * sizeof(double));
    insert_descriptor(seed, space_group, stage, "pdd", blob);
}

std::string CSPDatabase::get_meta(const std::string& key) {
    std::string val;
    try {
        *impl_->db << "SELECT value FROM meta WHERE key = ?" << key >> val;
    } catch (...) {
        // Key not found
    }
    return val;
}

int CSPDatabase::count_valid(int space_group, int min_stage) {
    int count = 0;
    *impl_->db << "SELECT COUNT(*) FROM structure WHERE space_group = ? AND stage >= ?"
               << space_group << min_stage
               >> count;
    return count;
}

int CSPDatabase::count_at_stage(int space_group, int stage) {
    int count = 0;
    *impl_->db << "SELECT COUNT(*) FROM structure WHERE space_group = ? AND stage = ?"
               << space_group << stage
               >> count;
    return count;
}

int CSPDatabase::count_unique_at_stage(int space_group, int stage) {
    int count = 0;
    *impl_->db << "SELECT COUNT(*) FROM structure WHERE space_group = ? AND stage = ? "
                  "AND unique_structure = 1"
               << space_group << stage
               >> count;
    return count;
}

int CSPDatabase::count_unique_missing_stage(int from_stage, int missing_stage) {
    int count = 0;
    *impl_->db << "SELECT COUNT(*) FROM structure WHERE stage = ? "
                  "AND unique_structure = 1 AND NOT EXISTS ("
                  "  SELECT 1 FROM structure AS later WHERE later.stage = ? "
                  "  AND later.seed = structure.seed "
                  "  AND later.space_group = structure.space_group)"
               << from_stage << missing_stage
               >> count;
    return count;
}

void CSPDatabase::close() {
    if (impl_->db) {
        // Unconditional: open() issues a BEGIN, so there is always a transaction to close.
        try { *impl_->db << "COMMIT;"; } catch (const std::exception&) {}
        impl_->pending = 0;
        // Finalize statements before the connection, or sqlite3_close reports busy.
        for (Impl::Stmt* st : {&impl_->ins_structure, &impl_->ins_descriptor,
                               &impl_->ins_flexible, &impl_->demote_repoint,
                               &impl_->demote_row})
            st->reset();
        impl_->db.reset();
    }
}

}  // namespace klasp
