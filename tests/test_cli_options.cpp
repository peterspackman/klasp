// The command line: what each subcommand registers, and how csp's flags become a
// run's configuration. Commands are built as src/cli/commands.hpp builds them.
// Test names never begin with "-": Catch2 would parse the name as an option.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "cli_options.hpp"
#include "commands.hpp"
#include "klasp/config.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <sys/wait.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <vector>

using namespace klasp;

namespace {

// CLI11's vector overload consumes from the back, so hand it argc/argv instead.
// One App per parse: re-parsing an App without clear() is not supported.
void parse(CLI::App& app, const std::vector<std::string>& args) {
    std::vector<const char*> argv{"klasp"};
    for (const auto& a : args) argv.push_back(a.c_str());
    app.parse(static_cast<int>(argv.size()), argv.data());
}

DedupConfig registered_defaults(const char* sub, std::vector<std::string> opts = {}) {
    DedupConfig cfg{};
    CLI::App app;
    add_dedup_options(app.add_subcommand(sub, "x"), cfg);
    std::vector<std::string> args{sub};
    args.insert(args.end(), opts.begin(), opts.end());
    parse(app, args);
    return cfg;
}

// A subcommand's --help, from the registration the binaries use.
std::string subcommand_help(const char* sub) {
    CLI::App app{"klasp"};
    cli::Commands commands(app);
    return app.get_subcommand(sub)->help();
}

// The `[value]` CLI11 prints after an option name, which capture_default_str()
// fills in from the bound variable. Returns "" if the option is absent.
std::string help_default(const std::string& help, const std::string& option) {
    const std::size_t at = help.find("  " + option + " ");
    if (at == std::string::npos) return {};
    const std::size_t open = help.find('[', at);
    const std::size_t eol = help.find('\n', at);
    if (open == std::string::npos || open > eol) return {};
    const std::size_t close = help.find(']', open);
    if (close == std::string::npos || close > eol) return {};
    return help.substr(open + 1, close - open - 1);
}

// One option's entry in --help: its name line and the description under it, up to
// the next option.
std::string help_entry(const std::string& help, const std::string& option) {
    const std::size_t at = help.find("  " + option);
    if (at == std::string::npos) return {};
    const std::size_t next = help.find("\n  -", at + 1);
    return help.substr(at, next == std::string::npos ? std::string::npos : next - at);
}

// `klasp csp <args>` parsed as klasp parses it, before config_problems runs.
RunConfig parse_csp(const std::vector<std::string>& args) {
    CLI::App app{"klasp"};
    cli::Commands commands(app);
    std::vector<std::string> command_line{"csp"};
    command_line.insert(command_line.end(), args.begin(), args.end());
    parse(app, command_line);
    return commands.csp.config();
}

// The same, for a run on one basis file and one space-group spec.
RunConfig csp_config(const std::vector<std::string>& flags, const std::string& sg = "14") {
    std::vector<std::string> args{"basis.json", "--sg", sg};
    args.insert(args.end(), flags.begin(), flags.end());
    return parse_csp(args);
}

std::string write_config(const std::string& name, const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << text;
    return path.string();
}

std::vector<std::string> subcommands() {
    std::vector<std::string> subs{"energy", "fdcheck", "csp", "extract", "prune",
                                  "compact", "compack", "summary", "converged"};
#ifdef KLASP_HAS_PET
    subs.push_back("rerank");
#endif
    return subs;
}

}  // namespace

TEST_CASE("registering dedup options leaves DedupConfig's own defaults intact",
          "[cli]") {
    const DedupConfig pristine{};
    const DedupConfig cfg = registered_defaults("csp");

    // capture_default_str() only reads the bound value; default_val() would assign.
    REQUIRE(cfg.energy_bin == pristine.energy_bin);
    REQUIRE(cfg.density_bin == pristine.density_bin);
    REQUIRE(cfg.amd_tol == pristine.amd_tol);
    REQUIRE(cfg.use_powder == pristine.use_powder);
    REQUIRE(cfg.powder_cos_thresh == pristine.powder_cos_thresh);
    REQUIRE(cfg.powder_cdtw_thresh == pristine.powder_cdtw_thresh);
    REQUIRE(cfg.powder_cdtw_band == pristine.powder_cdtw_band);
    REQUIRE(cfg.pdd_tol == pristine.pdd_tol);
}

// A CSP flag applies only when typed, so its CLI default must equal the config
// tree's.
TEST_CASE("the CLI and the config tree agree on every default", "[cli][config]") {
    const std::string help = subcommand_help("csp");
    const RunConfig cfg{};

    // Option name -> the value the tree holds for it. Booleans have no printed
    // default to compare.
    const std::vector<std::pair<std::string, double>> numeric{
        {"--target",              double(cfg.control.target_per_sg)},
        {"--batch-size",          double(cfg.control.batch_size)},
        {"--tvp",                 cfg.generator.tvp},
        {"--gen-nudge",           double(cfg.generator.nudge_budget)},
        {"--rp-iterations",       double(cfg.generator.rp_iterations)},
        {"--rp-pressure",         cfg.generator.rp_pressure_gpa},
        {"--rp-max-strain-step",  cfg.generator.rp_max_strain_step},
        {"--iterations",          double(cfg.optimizer.n_iterations)},
        {"--iterations-stage1",   double(cfg.optimizer.n_iterations_stage1)},
        {"--stage1-pressure",     cfg.optimizer.stage1_pressure_gpa},
        {"--grad-tol",            cfg.optimizer.grad_tol},
        {"--stream-pool-mult",    double(cfg.optimizer.stream_pool_mult)},
        {"--refill-interval",     double(cfg.optimizer.refill_interval)},
        {"--max-iters-per-slot",  double(cfg.optimizer.max_iters_per_slot)},
        {"--mp-cutoff",           cfg.forcefield.mp_cutoff_ang},
        {"--wolf-cutoff",         cfg.forcefield.wolf_cutoff},
        {"--max-vacuum",          cfg.output.max_vacuum},
        {"--result-queue-mb",     double(cfg.control.result_queue_mb)},
        {"--amd-tol",             cfg.dedup.amd_tol},
        {"--energy-bin",          cfg.dedup.energy_bin},
        {"--density-bin",         cfg.dedup.density_bin},
        {"--powder-cos-thresh",   cfg.dedup.powder_cos_thresh},
        {"--powder-cdtw-thresh",  cfg.dedup.powder_cdtw_thresh},
        {"--powder-cdtw-band",    double(cfg.dedup.powder_cdtw_band)},
        {"--dedup-pdd",           cfg.dedup.pdd_tol},
    };

    for (const auto& [option, tree_value] : numeric) {
        const std::string shown = help_default(help, option);
        INFO("option " << option << " prints [" << shown << "]");
        REQUIRE_FALSE(shown.empty());
        REQUIRE_THAT(std::stod(shown), Catch::Matchers::WithinRel(tree_value, 1e-9)
                                           || Catch::Matchers::WithinAbs(tree_value, 1e-12));
    }
}

// For every flag that names a key: type it, read back the value at its key, set
// that value through --set instead, and require the same document. The typed
// value is the first of several candidate forms that parses and changes the
// configuration.
TEST_CASE("every csp flag that names a configuration key sets that key and nothing else",
          "[cli][config]") {
    CLI::App listing_app{"klasp"};
    cli::Commands listing(listing_app);
    const auto& settings = listing.csp.settings();
    REQUIRE(settings.size() >= 50);

    const nlohmann::json baseline = csp_config({});
    for (const auto& setting : settings) {
        const std::string name = "--" + setting.option->get_lnames().front();
        INFO(name << " (" << setting.key << ")");

        std::vector<std::vector<std::string>> attempts{
            {name}, {"--no-" + setting.option->get_lnames().front()}};
        for (const char* value : {"3", "7", "0.37", "mc", "rigid_press", "legacy"})
            attempts.push_back({name, value});

        std::optional<nlohmann::json> typed;
        for (const auto& args : attempts) {
            try {
                const nlohmann::json j = csp_config(args);
                if (j != baseline) {
                    typed = j;
                    break;
                }
            } catch (const CLI::ParseError&) {
                // Not a form this option takes; try the next.
            }
        }
        REQUIRE(typed.has_value());

        std::string pointer = "/" + setting.key;
        std::replace(pointer.begin(), pointer.end(), '.', '/');
        const nlohmann::json value = typed->at(nlohmann::json::json_pointer(pointer));
        const nlohmann::json via_set = csp_config({"--set", setting.key + "=" + value.dump()});
        CHECK(*typed == via_set);
    }
}

TEST_CASE("a typed flag wins over --set and --config, and an untyped one leaves them alone",
          "[cli][config]") {
    const std::string file = write_config("klasp-test-layers.json",
        R"({"optimizer": {"n_iterations": 250, "grad_tol": 1e-6, "mc_moves": 90}})");

    const RunConfig from_file = csp_config({"--config", file});
    CHECK(from_file.optimizer.n_iterations == 250);
    CHECK(from_file.optimizer.grad_tol == 1e-6);

    const RunConfig layered = csp_config({"--config", file, "--set", "optimizer.mc_moves=120",
                                          "--iterations", "300"});
    CHECK(layered.optimizer.n_iterations == 300);
    CHECK(layered.optimizer.mc_moves == 120);
    CHECK(layered.optimizer.grad_tol == 1e-6);
}

TEST_CASE("lockstep stage 2 takes stage 1 with it", "[cli][config]") {
    const RunConfig cfg = csp_config({"--lockstep-stage2"});
    CHECK_FALSE(cfg.optimizer.stream);
    CHECK_FALSE(cfg.optimizer.stream_stage1);
}

TEST_CASE("a typed target replaces the per-group targets a preset carries", "[cli][config]") {
    const RunConfig preset = csp_config({}, "fine10");
    CHECK_FALSE(preset.control.targets.empty());

    const RunConfig typed = csp_config({"--target", "5"}, "fine10");
    CHECK(typed.control.targets.empty());
    CHECK(typed.control.target_per_sg == 5);
}

TEST_CASE("a config file can name a run's inputs and space groups on its own",
          "[cli][config]") {
    const std::string file = write_config("klasp-test-inputs.json",
        R"({"control": {"basis_paths": ["urea.json", "acetic_acid.json"],
                        "space_groups": [2, 14]}})");

    const RunConfig from_file = parse_csp({"--config", file});
    CHECK(from_file.control.basis_paths ==
          std::vector<std::string>{"urea.json", "acetic_acid.json"});
    CHECK(from_file.control.space_groups == std::vector<int>{2, 14});

    const RunConfig overridden = parse_csp({"benzene.json", "--config", file, "--sg", "19"});
    CHECK(overridden.control.basis_paths == std::vector<std::string>{"benzene.json"});
    CHECK(overridden.control.space_groups == std::vector<int>{19});
}

TEST_CASE("the configuration a run writes reads back as the same run", "[cli][config]") {
    RunConfig written;
    written.control.basis_paths = {"acetic_acid.json"};
    written.control.space_groups = {2, 14};
    written.control.targets = {{14, 500}};
    written.optimizer.grad_tol = 1e-7;
    written.generator.collision = CollisionMethod::RIGID_PRESS;
    const std::string file =
        write_config("klasp-test-rerun.json", effective_config_json(written));

    CHECK(nlohmann::json(parse_csp({"--config", file})) == nlohmann::json(written));
}

#ifdef KLASP_HAS_PET
TEST_CASE("a model named on the command line replaces the pair a config file recorded",
          "[cli][config]") {
    const std::string file = write_config("klasp-test-model.json",
        R"({"pet": {"model_json": "/runs/a/pet-mad-xs.json",
                    "model_weights": "/runs/a/pet-mad-xs.safetensors"}})");

    const RunConfig kept = csp_config({"--config", file});
    CHECK(kept.pet.model_json == "/runs/a/pet-mad-xs.json");

    const RunConfig renamed = csp_config({"--config", file, "--model", "pbe0"});
    CHECK(renamed.pet.model == "pbe0");
    CHECK(renamed.pet.model_json.empty());
    CHECK(renamed.pet.model_weights.empty());
}
#endif

TEST_CASE("csp and prune ship the same dedup defaults", "[cli]") {
    const std::string csp = subcommand_help("csp");
    const std::string prune = subcommand_help("prune");

    for (const char* opt : {"--amd-tol", "--energy-bin", "--density-bin",
                            "--powder-cos-thresh", "--powder-cdtw-thresh",
                            "--powder-cdtw-band"}) {
        const std::string a = help_default(csp, opt);
        const std::string b = help_default(prune, opt);
        INFO("option " << opt << ": csp=[" << a << "] prune=[" << b << "]");
        REQUIRE_FALSE(a.empty());        // registered on both
        REQUIRE_FALSE(b.empty());
        REQUIRE(a == b);
    }

    // Both must still be DedupConfig's own initialisers.
    const DedupConfig pristine{};
    REQUIRE(help_default(csp, "--amd-tol") == fmt::format("{}", pristine.amd_tol));
    REQUIRE(help_default(csp, "--energy-bin") == fmt::format("{}", pristine.energy_bin));
    REQUIRE(help_default(csp, "--density-bin") == fmt::format("{}", pristine.density_bin));
    REQUIRE(help_default(csp, "--powder-cdtw-band") ==
            fmt::format("{}", pristine.powder_cdtw_band));

    REQUIRE(csp.find("--no-powder-dedup") != std::string::npos);
    REQUIRE(prune.find("--no-powder-dedup") != std::string::npos);
}

// cross_sg defaults differ: csp is per-space-group, prune clusters across them.
// add_switch appends the real default to the description.
TEST_CASE("the cross-SG default is opposite for csp and prune, and says so",
          "[cli]") {
    // Read from the --cross-sg entry alone: other switches are on by default too.
    const std::string csp = help_entry(subcommand_help("csp"), "--cross-sg");
    const std::string prune = help_entry(subcommand_help("prune"), "--cross-sg");

    REQUIRE_FALSE(csp.empty());
    REQUIRE_FALSE(prune.empty());
    CHECK(csp.find("[off by default]") != std::string::npos);
    CHECK(prune.find("[on by default]") != std::string::npos);
    CHECK(csp.find("[on by default]") == std::string::npos);
    CHECK(prune.find("[off by default]") == std::string::npos);
}

TEST_CASE("every dedup tolerance is reachable by its documented name", "[cli]") {
    const DedupConfig cfg = registered_defaults("prune", {
        "--amd-tol", "0.1",
        "--energy-bin", "50",
        "--density-bin", "0.5",
        "--powder-cos-thresh", "0.9",
        "--powder-cdtw-thresh", "2",
        "--powder-cdtw-band", "3",
        "--no-powder-dedup"});

    REQUIRE(cfg.amd_tol == 0.1);
    REQUIRE(cfg.energy_bin == 50.0);
    REQUIRE(cfg.density_bin == 0.5);
    REQUIRE(cfg.powder_cos_thresh == 0.9);
    REQUIRE(cfg.powder_cdtw_thresh == 2.0);
    REQUIRE(cfg.powder_cdtw_band == 3);
    REQUIRE_FALSE(cfg.use_powder);   // the "!" prefix negates
}

TEST_CASE("the CSP optimizer defaults have one definition", "[cli]") {
    const RunConfig cfg{};
    REQUIRE(cfg.optimizer.grad_tol == 1e-8);
    REQUIRE(cfg.optimizer.n_iterations == 150);
    REQUIRE(cfg.optimizer.n_iterations_stage1 == 100);
}

namespace {

RowFilter parse_filter(std::vector<std::string> opts = {}) {
    RowFilter filter{};
    CLI::App app;
    add_structure_filter_options(app.add_subcommand("extract", "x"), filter);
    std::vector<std::string> args{"extract"};
    args.insert(args.end(), opts.begin(), opts.end());
    parse(app, args);
    return filter;
}

}  // namespace

TEST_CASE("the keep flags keep what they name", "[cli]") {
    const RowFilter off = parse_filter();
    REQUIRE_FALSE(off.keep_catastrophes);
    REQUIRE_FALSE(off.keep_warned);

    REQUIRE(parse_filter({"--keep-catastrophes"}).keep_catastrophes);
    REQUIRE(parse_filter({"--keep-warned"}).keep_warned);

    const RowFilter one = parse_filter({"--keep-warned"});
    REQUIRE_FALSE(one.keep_catastrophes);
}

TEST_CASE("the catastrophe threshold keeps its registered default", "[cli]") {
    const RowFilter pristine{};
    REQUIRE(parse_filter().catastrophe_threshold == pristine.catastrophe_threshold);
    REQUIRE(parse_filter({"--catastrophe-threshold", "-70"}).catastrophe_threshold == -70.0);
}

TEST_CASE("a keep flag survives alongside other filter options", "[cli]") {
    const RowFilter o = parse_filter({"--keep-warned", "--catastrophe-threshold=-70"});
    REQUIRE(o.keep_warned);
    REQUIRE(o.catastrophe_threshold == -70.0);
    REQUIRE_FALSE(o.keep_catastrophes);
}

TEST_CASE("wall-clock durations parse in the units they look like", "[cli][runtime]") {
    REQUIRE(parse_duration_seconds("30s") == 30.0);
    REQUIRE(parse_duration_seconds("90m") == 5400.0);
    REQUIRE(parse_duration_seconds("7.5h") == 27000.0);
    REQUIRE(parse_duration_seconds("2H") == 7200.0);
    // A bare number is minutes.
    REQUIRE(parse_duration_seconds("450") == 27000.0);
    REQUIRE(parse_duration_seconds("0") == 0.0);

    // Rejected by returning < 0.
    REQUIRE(parse_duration_seconds("") < 0);
    REQUIRE(parse_duration_seconds("h") < 0);
    REQUIRE(parse_duration_seconds("soon") < 0);
    REQUIRE(parse_duration_seconds("1d") < 0);        // days are not a unit here
    REQUIRE(parse_duration_seconds("8:00:00") < 0);   // SLURM's spelling is not this one
    REQUIRE(parse_duration_seconds("-5m") < 0);
}

TEST_CASE("no option puts its negation in the middle of its name", "[cli]") {
    const std::regex mid_negation(R"(--[a-z0-9]+(-[a-z0-9]+)*-no-[a-z])");
    for (const auto& sub : subcommands()) {
        INFO("klasp " << sub);
        CHECK_FALSE(std::regex_search(subcommand_help(sub.c_str()), mid_negation));
    }
}

TEST_CASE("the short form -n means --top wherever it appears, and -k is gone", "[cli]") {
    const std::regex short_n(R"((^|\s)-n,--([a-z-]+))");
    const std::regex short_k(R"((^|\s)-k[,\s])");
    for (const auto& sub : subcommands()) {
        const std::string help = subcommand_help(sub.c_str());
        INFO("klasp " << sub);
        for (std::sregex_iterator it(help.begin(), help.end(), short_n), end; it != end; ++it)
            CHECK((*it)[2] == "top");
        CHECK_FALSE(std::regex_search(help, short_k));
    }
}

// Naming another command's flag is fine when that command is named in front of
// it, as in `csp --grad-tol`.
TEST_CASE("every option a subcommand's help names is one that subcommand has", "[cli]") {
    CLI::App app{"klasp"};
    cli::Commands commands(app);
    const std::regex flag(R"((^|[^A-Za-z0-9-])(--[a-z0-9][a-z0-9-]*))");
    const std::vector<std::string> command_names = subcommands();

    for (CLI::App* sub : app.get_subcommands([](CLI::App*) { return true; })) {
        std::set<std::string> registered;
        std::string text = sub->get_description();
        for (const CLI::Option* option : sub->get_options()) {
            for (const auto& lname : option->get_lnames()) registered.insert("--" + lname);
            text += "\n" + option->get_description();
        }

        for (std::sregex_iterator it(text.begin(), text.end(), flag), end; it != end; ++it) {
            const std::string name = (*it)[2];
            if (registered.count(name) > 0) continue;
            const auto at = static_cast<std::size_t>(it->position(2));
            const bool named_with_its_command =
                std::any_of(command_names.begin(), command_names.end(),
                            [&](const std::string& command) {
                                const std::string prefix = command + " ";
                                return at >= prefix.size() &&
                                       text.compare(at - prefix.size(), prefix.size(),
                                                    prefix) == 0;
                            });
            if (!named_with_its_command)
                FAIL_CHECK("klasp " << sub->get_name() << " --help names " << name
                                    << ", which it does not have");
        }
    }
}

// --- the binary ----------------------------------------------------------------

namespace {

struct Ran {
    int status = -1;
    std::string output;
};

// The shipped binary, run with `args`. stdout only unless `with_stderr`, so a
// --dump-config document parses as one.
Ran run_klasp(const std::string& args, bool with_stderr) {
    const std::string cmd = std::string(KLASP_BINARY_PATH) + " " + args +
                            (with_stderr ? " 2>&1" : " 2>/dev/null");
    Ran r;
    FILE* p = popen(cmd.c_str(), "r");
    REQUIRE(p != nullptr);
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), p)) r.output += buf;
    const int raw = pclose(p);
    r.status = WIFEXITED(raw) ? WEXITSTATUS(raw) : -1;
    return r;
}

}  // namespace

// --dump-config does not read the basis, so these can name one that does not exist.
TEST_CASE("a contradiction written in a config file is refused like the flag",
          "[cli][config]") {
    const std::string file =
        write_config("klasp-test-wolf.json", R"({"forcefield": {"wolf": true}})");
    const Ran r = run_klasp("csp basis.json --sg 14 --config " + file + " --dump-config",
                            /*with_stderr=*/true);
    CHECK(r.status == 2);
    CHECK(r.output.find("forcefield.wolf") != std::string::npos);
}

TEST_CASE("an unknown key in a config file is refused by name", "[cli][config]") {
    const std::string file =
        write_config("klasp-test-typo.json", R"({"optimizer": {"gradtol": 1e-6}})");
    const Ran r = run_klasp("csp basis.json --sg 14 --config " + file + " --dump-config",
                            /*with_stderr=*/true);
    CHECK(r.status == 2);
    CHECK(r.output.find("optimizer.gradtol") != std::string::npos);
}

TEST_CASE("a run with no inputs from the command line or a file is refused", "[cli][config]") {
    const Ran r = run_klasp("csp --sg 14", /*with_stderr=*/true);
    CHECK(r.status == 2);
    CHECK(r.output.find("control.basis_paths") != std::string::npos);
}

TEST_CASE("the defaults print as a configuration with no inputs at all", "[cli][config]") {
    const Ran r = run_klasp("csp --dump-config", /*with_stderr=*/false);
    REQUIRE(r.status == 0);
    CHECK(nlohmann::json::parse(r.output)["control"]["basis_paths"].empty());
}

// Only csp starts MPI.
TEST_CASE("a one-process command that fails explains itself and exits 1", "[cli]") {
    const Ran r = run_klasp("energy /nonexistent/structure.json", /*with_stderr=*/true);
    CHECK(r.status == 1);
    CHECK(r.output.find("klasp: ") != std::string::npos);
    CHECK(r.output.find("MPI_ABORT") == std::string::npos);
}
