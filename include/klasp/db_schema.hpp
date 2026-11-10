#pragma once
// Schema probes for databases opened directly with sqlite rather than through
// CSPDatabase, which is the only thing that runs migrations.

#include <sqlite_modern_cpp.h>
#include <string>

namespace klasp {

inline bool db_has_column(sqlite::database& db, const std::string& table,
                          const std::string& column) {
    int n = 0;
    try {
        db << "SELECT COUNT(*) FROM pragma_table_info(?) WHERE name = ?"
           << table << column >> n;
    } catch (const std::exception&) { return false; }
    return n > 0;
}

// Clause dropping structures flagged by klasp/warnings.hpp. Empty when the
// caller wants them, or when the database predates the column.
inline std::string warned_filter(sqlite::database& db, bool exclude_warned) {
    if (!exclude_warned || !db_has_column(db, "structure", "warnings")) return "";
    return " AND warnings = 0";
}

}  // namespace klasp
