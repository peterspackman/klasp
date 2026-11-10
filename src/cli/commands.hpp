#pragma once

// klasp's subcommands, as the command line sees them. Each owns its options and
// after the parse either runs in this process or, for csp, hands back the
// configuration it describes. CLI11 binds to a command's own members, so
// commands cannot be copied or moved.

#include "klasp/cmd_energy.hpp"
#include "klasp/cmd_fdcheck.hpp"
#include "klasp/config.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/wyckoff.hpp"
#ifdef KLASP_HAS_PET
#include "klasp/cmd_rerank.hpp"
#endif

#include <CLI/CLI.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace klasp::cli {

// -q, -v and --plain: how much a program says, and whether it redraws in place.
struct OutputOptions {
    bool quiet = false;
    int verbosity = 0;
    bool plain = false;
};

// Registers -q/-v/--plain and lets --kokkos-* arguments through the parse. Call
// it before adding any command: a subcommand takes its handling of unrecognised
// arguments from its parent when it is created.
void add_program_options(CLI::App& app, OutputOptions& options);
void apply_output_options(const OutputOptions& options);

// Parse, then refuse anything left over that is not a --kokkos-* argument.
// Returns the exit status after --help, --version or a usage error, or nothing
// when the program should carry on.
std::optional<int> parse_command_line(CLI::App& app, int argc, char* argv[]);

// A command that runs in this process once Kokkos is up, without MPI.
class LocalCommand {
public:
    virtual ~LocalCommand() = default;
    LocalCommand(const LocalCommand&) = delete;
    LocalCommand& operator=(const LocalCommand&) = delete;

    bool parsed() const { return app_->parsed(); }

    // The exit status (klasp/exit_code.hpp). A failure throws.
    virtual int run() = 0;

protected:
    explicit LocalCommand(CLI::App* app) : app_(app) {}
    CLI::App* app_;
};

// Kokkos up, the command, Kokkos down. A ConfigError exits as a usage error and
// anything else as a failure, each explained on stderr under `program`'s name.
int run_local(LocalCommand& command, const char* program, int argc, char* argv[]);

class EnergyCommand : public LocalCommand {
public:
    explicit EnergyCommand(CLI::App& parent);
    int run() override;

private:
    EnergyOptions options_;
    std::vector<std::string> inputs_;
};

class FdcheckCommand : public LocalCommand {
public:
    explicit FdcheckCommand(CLI::App& parent);
    int run() override;

private:
    FdcheckOptions options_;
};

#ifdef KLASP_HAS_PET
class RerankCommand : public LocalCommand {
public:
    explicit RerankCommand(CLI::App& parent);
    int run() override;

private:
    RerankOptions options_;
};
#endif

class ExtractCommand : public LocalCommand {
public:
    explicit ExtractCommand(CLI::App& parent);
    int run() override;

private:
    ExtractOptions options_;
};

class PruneCommand : public LocalCommand {
public:
    explicit PruneCommand(CLI::App& parent);
    int run() override;

private:
    PruneOptions options_;
};

class CompackCommand : public LocalCommand {
public:
    explicit CompackCommand(CLI::App& parent);
    int run() override;

private:
    CompackRunOptions options_;
    std::vector<std::string> inputs_;
};

class CompactCommand : public LocalCommand {
public:
    explicit CompactCommand(CLI::App& parent);
    int run() override;

private:
    CompactOptions options_;
};

class SummaryCommand : public LocalCommand {
public:
    explicit SummaryCommand(CLI::App& parent);
    int run() override;

private:
    SummaryOptions options_;
};

class WyckoffCommand : public LocalCommand {
public:
    explicit WyckoffCommand(CLI::App& parent);
    int run() override;

private:
    WyckoffOptions options_;
};

class ConvergedCommand : public LocalCommand {
public:
    explicit ConvergedCommand(CLI::App& parent);
    int run() override;

private:
    ConvergedOptions options_;
};

// csp: the command line as one layer of a run's configuration.
class CspCommand {
public:
    explicit CspCommand(CLI::App& parent);
    CspCommand(const CspCommand&) = delete;
    CspCommand& operator=(const CspCommand&) = delete;

    bool parsed() const { return app_->parsed(); }
    bool dump_config() const { return dump_config_; }

    // The run this command line describes: the defaults, then --config, then each
    // --set, then the flags that were typed. Not checked for contradictions (see
    // config_problems). Throws ConfigError when a layer cannot apply.
    RunConfig config() const;

    // A flag that sets exactly one configuration key, and which one.
    struct Setting {
        const CLI::Option* option;
        std::string key;
    };
    const std::vector<Setting>& settings() const { return settings_; }

private:
    template <class Field>
    void when_typed(CLI::Option* option, Field field);
    template <class Field>
    CLI::Option* keyed(CLI::Option* option, const std::string& key, Field field,
                       const char* group);
    template <class Field>
    CLI::Option* setting(const std::string& names, const std::string& key, Field field,
                         const std::string& help, const char* group);
    template <class Field>
    CLI::Option* flag(const std::string& names, const std::string& key, Field field,
                      const std::string& help, const char* group);
    template <class Field>
    CLI::Option* toggle(const std::string& name, const std::string& key, Field field,
                        const std::string& help, const char* group);
    template <class Field>
    CLI::Option* choice(const std::string& name, const std::string& key, Field field,
                        const std::vector<std::string>& names, const std::string& help,
                        const char* group);

    CLI::App* app_;
    // What each flag holds; a field reaches the configuration only when its flag
    // was typed.
    RunConfig typed_;
    std::vector<std::function<void(RunConfig&)>> apply_;
    std::vector<Setting> settings_;

    std::vector<std::string> inputs_;
    std::string config_path_;
    std::vector<std::string> sets_;
    bool dump_config_ = false;
    std::string sg_spec_;
    std::string max_runtime_;
    double wolf_cutoff_ang_ = 0;
};

// Every subcommand, registered on `app` in the order --help lists them.
struct Commands {
    explicit Commands(CLI::App& app);

    CspCommand csp;
    EnergyCommand energy;
    FdcheckCommand fdcheck;
    ExtractCommand extract;
#ifdef KLASP_HAS_PET
    RerankCommand rerank;
#endif
    PruneCommand prune;
    CompackCommand compack;
    CompactCommand compact;
    SummaryCommand summary;
    ConvergedCommand converged;
    WyckoffCommand wyckoff;

    // The one-process command the parse chose, or nullptr when it chose csp.
    LocalCommand* parsed_local();
};

}  // namespace klasp::cli
