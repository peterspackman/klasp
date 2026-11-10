#pragma once
// Every connection outside CSPDatabase goes through these. Readers open
// read-only (no write lock, no empty file created); all connections wait out
// lock contention rather than failing.

#include <sqlite_modern_cpp.h>
#include <stdexcept>
#include <string>

namespace klasp {

inline constexpr int kDbBusyTimeoutMs = 30000;

inline sqlite::database open_database(const std::string& path, sqlite::OpenFlags flags) {
    sqlite::sqlite_config cfg;
    cfg.flags = flags;
    try {
        sqlite::database db(path, cfg);
        db << "PRAGMA busy_timeout=" + std::to_string(kDbBusyTimeoutMs) + ";";
        return db;
    } catch (const sqlite::sqlite_exception& e) {
        throw std::runtime_error("could not open " + path + ": " + e.what());
    }
}

inline sqlite::database open_database_readonly(const std::string& path) {
    return open_database(path, sqlite::OpenFlags::READONLY);
}

inline sqlite::database open_database_readwrite(const std::string& path) {
    return open_database(path, sqlite::OpenFlags::READWRITE);
}

// What to tell someone whose command gave up on a lock.
inline const char* const kDbLockedHint =
    "another process held a lock on the database for longer than 30 s. A CSP run "
    "or rerank that is still writing holds one; wait for it to finish or work on "
    "a copy. A database on a parallel filesystem (Lustre) should only be opened "
    "from one node at a time.";

}  // namespace klasp
