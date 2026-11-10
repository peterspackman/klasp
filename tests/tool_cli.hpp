#pragma once

// CLI11 setup shared by the klasp_pet_* developer tools. Kokkos::initialize
// reads --kokkos-* from argv after the parse, so those pass through; any other
// leftover is an error.

#include <CLI/CLI.hpp>

#include <cstdio>
#include <string>

namespace klasp::tools {

// Configures in place: CLI::App is neither copyable nor movable.
inline void tool_init(CLI::App& app) {
    app.allow_extras();
    app.footer("Kokkos arguments (--kokkos-*) are passed through to Kokkos::initialize.");
}

// Reject leftovers that are not Kokkos'. Returns 0 when everything left over is
// a --kokkos-* argument, or 2 (the tools' usage-error code) otherwise.
inline int tool_check_extras(const CLI::App& app) {
    for (const auto& extra : app.remaining(/*recurse=*/true)) {
        if (extra.rfind("--kokkos", 0) == 0) continue;
        std::fprintf(stderr, "Unrecognised option: %s\n", extra.c_str());
        return 2;
    }
    return 0;
}

}  // namespace klasp::tools

// Parse, then check the leftovers. Used at the top of a tool's main, before
// Kokkos::initialize. Expands to a `return` on a parse error or a stray option.
#define KLASP_TOOL_PARSE(app, argc, argv)                       \
    do {                                                        \
        try {                                                   \
            (app).parse((argc), (argv));                        \
        } catch (const CLI::ParseError& e) {                    \
            return (app).exit(e);                               \
        }                                                       \
        const int rc_ = klasp::tools::tool_check_extras(app);   \
        if (rc_ != 0) return rc_;                               \
    } while (0)
