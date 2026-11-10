// klasp entry point. Only csp initialises MPI; the commands are defined in src/cli.

#include "commands.hpp"

#include "cli_options.hpp"
#include "klasp/config.hpp"
#include "klasp/csp_run.hpp"
#include "klasp/exit_code.hpp"
#include "klasp/log.hpp"
#include "klasp/orchestrator.hpp"

#include <CLI/CLI.hpp>
#include <Kokkos_Core.hpp>
#include <fmt/base.h>
#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace klasp;

namespace {

// This process's launcher rank, readable before MPI is up; 0 outside a launcher.
int launcher_rank() {
    for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"})
        if (const char* value = std::getenv(name)) return std::atoi(value);
    return 0;
}

int run_csp(const cli::CspCommand& csp, int argc, char* argv[]) {
    // Taken before MPI so the time budget covers setup too.
    const auto process_start = std::chrono::steady_clock::now();

    // Parsed before MPI so a refusal needs no launcher.
    RunConfig cfg;
    try {
        cfg = csp.config();
        const std::vector<std::string> problems = config_problems(cfg);
        if (!problems.empty()) throw ConfigError(problems);
        if (csp.dump_config()) {
            // No banner: the output is JSON for piping.
            fmt::print("{}\n", effective_config_json(cfg));
            return +ExitCode::Ok;
        }
        cfg = prepare_run_config(cfg);
    } catch (const ConfigError& e) {
        if (launcher_rank() == 0) fmt::print(stderr, "klasp: {}\n", e.what());
        return +ExitCode::Usage;
    } catch (const std::exception& e) {
        if (launcher_rank() == 0) fmt::print(stderr, "klasp: {}\n", e.what());
        return +ExitCode::Failed;
    }

    int mpi_thread_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_provided);

    // Rank 0 runs two MPI threads during csp, which requires MPI_THREAD_MULTIPLE.
    // Cray MPICH silently grants less unless MPICH_MAX_THREAD_SAFETY=multiple.
    if (mpi_thread_provided < MPI_THREAD_MULTIPLE) {
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        if (r == 0) {
            const char* got = mpi_thread_provided == MPI_THREAD_SINGLE     ? "SINGLE"
                            : mpi_thread_provided == MPI_THREAD_FUNNELED   ? "FUNNELED"
                            : mpi_thread_provided == MPI_THREAD_SERIALIZED ? "SERIALIZED"
                                                                           : "unknown";
            fmt::print(stderr,
                "\nklasp: this MPI granted MPI_THREAD_{} where MPI_THREAD_MULTIPLE was\n"
                "requested. Rank 0 drains results on a second thread, so continuing would\n"
                "corrupt results and hang, silently. On Cray MPICH set:\n"
                "    export MPICH_MAX_THREAD_SAFETY=multiple\n"
                "and re-run.\n", got);
            std::fflush(stderr);
        }
        MPI_Finalize();
        return +ExitCode::UnsupportedMpiThreading;
    }

    // Print an uncaught exception's message alone, then MPI_Abort so other ranks do
    // not deadlock in a collective.
    std::set_terminate([] {
        if (auto exc = std::current_exception()) {
            try {
                std::rethrow_exception(exc);
            } catch (const std::exception& e) {
                fmt::print(stderr, "\nklasp: fatal error: {}\n", e.what());
            } catch (...) {
                fmt::print(stderr, "\nklasp: fatal error: unknown exception\n");
            }
        } else {
            fmt::print(stderr, "\nklasp: terminate called with no active exception\n");
        }
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();  // not reached: MPI_Abort does not return
    });

    Kokkos::initialize(argc, argv);
    int exit_status = +ExitCode::Ok;
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // Narration from rank 0 only; warnings and errors from any rank.
        logging::set_rank(rank);
        logging::info("{}\n", version_banner("klasp"));
        try {
            CspRunOptions options;
            options.process_start = process_start;
            // The library does not install signal handlers itself; see CspRunOptions.
            options.install_signal_handlers = true;
            run_orchestrated_csp(cfg, options);
        } catch (const RunRefused& r) {
            // Every rank agreed to stop; exit through MPI_Finalize.
            exit_status = r.code;
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return exit_status;
}

}  // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"klasp - crystal structure prediction and lattice energy evaluation"};
    app.require_subcommand(1);
    app.set_version_flag("--version", version_banner("klasp"));
    cli::OutputOptions output;
    cli::add_program_options(app, output);
    cli::Commands commands(app);

    if (const auto status = cli::parse_command_line(app, argc, argv)) return *status;
    cli::apply_output_options(output);

    if (commands.csp.parsed()) return run_csp(commands.csp, argc, argv);
    return cli::run_local(*commands.parsed_local(), "klasp", argc, argv);
}
