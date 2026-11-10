// What klasp and klasp-energy share around their commands: the output flags, the
// parse, and running a command in one process.

#include "commands.hpp"

#include "cli_options.hpp"
#include "klasp/db_open.hpp"
#include "klasp/exit_code.hpp"
#include "klasp/log.hpp"

#include <Kokkos_Core.hpp>
#include <fmt/base.h>

#include <exception>

namespace klasp::cli {

void add_program_options(CLI::App& app, OutputOptions& options) {
    // Kokkos claims its own --kokkos-* arguments after the parse, so CLI11 must
    // tolerate extras; parse_command_line rejects every other unknown option.
    app.allow_extras();

    // How much a run says. Rank gating is the log sink's job, not each call site's.
    app.add_flag("-q,--quiet", options.quiet,
        "Only report problems: suppress the running commentary, keep warnings "
        "and errors.");
    app.add_flag("-v,--verbose", options.verbosity,
        "Say more. Repeat for per-slot and per-structure detail (-vv).");
    app.add_flag("--plain", options.plain,
        "No ANSI escapes and no in-place redraw, whatever stdout is. For a job "
        "whose output is forwarded to a log file, where escape codes would "
        "accumulate as literal bytes.");
}

void apply_output_options(const OutputOptions& options) {
    logging::set_plain(options.plain);
    logging::set_level(options.quiet             ? logging::Level::Warn
                       : options.verbosity >= 2 ? logging::Level::Debug
                       : options.verbosity == 1 ? logging::Level::Detail
                                                : logging::Level::Info);
}

std::optional<int> parse_command_line(CLI::App& app, int argc, char* argv[]) {
    // Not CLI11_PARSE: its error numbering would break "2 means usage error".
    // --help and --version still exit 0.
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        const int rc = app.exit(e);
        return rc == 0 ? +ExitCode::Ok : +ExitCode::Usage;
    }

    // An unknown option is an error, not something to ignore.
    bool kokkos_value_pending = false;
    for (const auto& extra : app.remaining(/*recurse=*/true)) {
        // "--kokkos-num-threads 4" spells its value as a separate argument. Only a
        // token that is not itself an option counts as that value, so a valueless
        // Kokkos flag cannot swallow a following typo.
        if (kokkos_value_pending) {
            kokkos_value_pending = false;
            if (extra.rfind("-", 0) != 0) continue;
        }
        if (extra.rfind("--kokkos", 0) == 0) {
            kokkos_value_pending = extra.find('=') == std::string::npos;
            continue;
        }
        fmt::print(stderr,
            "Unrecognised option: {}\n"
            "Only --kokkos-* arguments are passed through. If this option exists in a\n"
            "newer klasp, the build being run is older than the script calling it.\n",
            extra);
        return +ExitCode::Usage;
    }
    return std::nullopt;
}

int run_local(LocalCommand& command, const char* program, int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int status = +ExitCode::Ok;
    try {
        logging::info("{}\n", version_banner(program));
        status = command.run();
    } catch (const ConfigError& e) {
        fmt::print(stderr, "{}: {}\n", program, e.what());
        status = +ExitCode::Usage;
    } catch (const sqlite::errors::busy& e) {
        fmt::print(stderr, "{}: {}: {}\n", program, e.what(), kDbLockedHint);
        status = +ExitCode::Failed;
    } catch (const sqlite::errors::locked& e) {
        fmt::print(stderr, "{}: {}: {}\n", program, e.what(), kDbLockedHint);
        status = +ExitCode::Failed;
    } catch (const std::exception& e) {
        // cspy reads stderr and the non-zero exit status.
        fmt::print(stderr, "{}: {}\n", program, e.what());
        status = +ExitCode::Failed;
    }
    Kokkos::finalize();
    return status;
}

}  // namespace klasp::cli
