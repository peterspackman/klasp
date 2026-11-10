// klasp-energy: the `energy` command in a binary that does not link MPI, so it can
// run inside another MPI job's allocation. Takes the same arguments as `klasp energy`.

#include "cli_options.hpp"
#include "commands.hpp"

#include <CLI/CLI.hpp>

int main(int argc, char* argv[]) {
    CLI::App app{"klasp-energy - evaluate or relax structures (no MPI)"};
    app.require_subcommand(1);
    app.set_version_flag("--version", klasp::version_banner("klasp-energy"));
    klasp::cli::OutputOptions output;
    klasp::cli::add_program_options(app, output);
    klasp::cli::EnergyCommand energy(app);

    if (const auto status = klasp::cli::parse_command_line(app, argc, argv)) return *status;
    klasp::cli::apply_output_options(output);
    return klasp::cli::run_local(energy, "klasp-energy", argc, argv);
}
