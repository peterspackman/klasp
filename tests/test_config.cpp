// The configuration tree: defaults, round-trip, merge, and --set. Defaults are
// read from a default-constructed RunConfig, never restated as literals.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "klasp/config.hpp"
#include "klasp/csp_run.hpp"
#ifdef KLASP_HAS_PET
#include "klasp/pet_bridge.hpp"
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::WithinAbs;
using klasp::ConfigError;
using klasp::RunConfig;

// Test names never begin with "--": Catch2 would parse the name as an option.

TEST_CASE("A config round-trips through JSON unchanged", "[config]") {
    RunConfig cfg;
    cfg.generator.tvp = 1.35;
    cfg.generator.collision = klasp::CollisionMethod::RIGID_PRESS;
    cfg.optimizer.n_iterations = 321;
    cfg.optimizer.stream = true;
    cfg.forcefield.wolf = true;
    cfg.forcefield.ewald_accuracy = 1e-8;
    cfg.pet.relax = true;
    cfg.pet.model_json = "/models/pet-mad-xs.json";
    cfg.dedup.energy_bin = 3.0;
    cfg.output.dir = "run-out";
    cfg.control.space_groups = {2, 14, 19};
    cfg.control.targets = {{2, 100}, {14, 250}};
    cfg.control.batch_sizing = klasp::BatchSizing::LEGACY;

    const nlohmann::json j = cfg;
    const auto back = j.get<RunConfig>();

    CHECK_THAT(back.generator.tvp, WithinAbs(1.35, 0));
    CHECK(back.generator.collision == klasp::CollisionMethod::RIGID_PRESS);
    CHECK(back.optimizer.n_iterations == 321);
    CHECK(back.optimizer.stream);
    CHECK(back.forcefield.wolf);
    CHECK_THAT(back.forcefield.ewald_accuracy, WithinAbs(1e-8, 0));
    CHECK(back.pet.relax);
    CHECK(back.pet.model_json == "/models/pet-mad-xs.json");
    CHECK_THAT(back.dedup.energy_bin, WithinAbs(3.0, 0));
    CHECK(back.output.dir == "run-out");
    CHECK(back.control.space_groups == std::vector<int>{2, 14, 19});
    CHECK(back.control.targets.at(14) == 250);
    CHECK(back.control.batch_sizing == klasp::BatchSizing::LEGACY);

    CHECK(nlohmann::json(back) == j);
}

TEST_CASE("Enums serialise by name, not by ordinal", "[config]") {
    // Enums serialise by name, so declaration order cannot reinterpret old configs.
    RunConfig cfg;
    cfg.generator.collision = klasp::CollisionMethod::FFT_PACK;
    cfg.control.batch_sizing = klasp::BatchSizing::ATOMS;
    const nlohmann::json j = cfg;
    CHECK(j["generator"]["collision"] == "fft_pack");
    CHECK(j["control"]["batch_sizing"] == "atoms");
}

TEST_CASE("A partial config leaves everything it does not name alone", "[config]") {
    const RunConfig defaults;
    RunConfig cfg;
    klasp::apply_config_json(cfg, nlohmann::json::parse(R"({
        "optimizer": {"n_iterations": 42},
        "output": {"dir": "elsewhere"}
    })"));

    CHECK(cfg.optimizer.n_iterations == 42);
    CHECK(cfg.output.dir == "elsewhere");
    CHECK(cfg.optimizer.n_iterations_stage1 == defaults.optimizer.n_iterations_stage1);
    CHECK_THAT(cfg.optimizer.grad_tol, WithinAbs(defaults.optimizer.grad_tol, 0));
    CHECK(cfg.output.store_all_stages == defaults.output.store_all_stages);
    CHECK_THAT(cfg.generator.tvp, WithinAbs(defaults.generator.tvp, 0));
    CHECK_THAT(cfg.dedup.energy_bin, WithinAbs(defaults.dedup.energy_bin, 0));
}

TEST_CASE("An array in a config replaces rather than extends", "[config]") {
    RunConfig cfg;
    cfg.control.space_groups = {2, 14, 19, 61};
    klasp::apply_config_json(cfg, nlohmann::json::parse(R"({
        "control": {"space_groups": [33]}
    })"));
    CHECK(cfg.control.space_groups == std::vector<int>{33});
}

TEST_CASE("A --set override reaches a value and converts it", "[config]") {
    RunConfig cfg;
    std::string err;

    REQUIRE(klasp::apply_config_override(cfg, "optimizer.n_iterations=250", err));
    CHECK(cfg.optimizer.n_iterations == 250);

    REQUIRE(klasp::apply_config_override(cfg, "optimizer.stream=true", err));
    CHECK(cfg.optimizer.stream);

    REQUIRE(klasp::apply_config_override(cfg, "generator.tvp=1.45", err));
    CHECK_THAT(cfg.generator.tvp, WithinAbs(1.45, 1e-12));

    REQUIRE(klasp::apply_config_override(cfg, "output.dir=/scratch/run7", err));
    CHECK(cfg.output.dir == "/scratch/run7");

    REQUIRE(klasp::apply_config_override(cfg, "generator.collision=rigid_press", err));
    CHECK(cfg.generator.collision == klasp::CollisionMethod::RIGID_PRESS);

    REQUIRE(klasp::apply_config_override(cfg, "pet.model_json=/m/a=b.json", err));
    CHECK(cfg.pet.model_json == "/m/a=b.json");
}

TEST_CASE("A --set override is refused when it cannot apply, and says why", "[config]") {
    RunConfig cfg;
    const RunConfig before;
    std::string err;

    SECTION("an unknown section") {
        CHECK_FALSE(klasp::apply_config_override(cfg, "optimiser.n_iterations=5", err));
        CHECK_THAT(err, ContainsSubstring("optimiser"));
    }
    SECTION("an unknown key in a real section") {
        CHECK_FALSE(klasp::apply_config_override(cfg, "optimizer.iterations=5", err));
        CHECK_THAT(err, ContainsSubstring("optimizer.iterations"));
    }
    SECTION("a value of the wrong type") {
        CHECK_FALSE(klasp::apply_config_override(cfg, "optimizer.n_iterations=soon", err));
        CHECK_THAT(err, ContainsSubstring("n_iterations"));
    }
    SECTION("no '=' at all") {
        CHECK_FALSE(klasp::apply_config_override(cfg, "optimizer.n_iterations", err));
        CHECK_THAT(err, ContainsSubstring("section.key=value"));
    }

    CHECK(nlohmann::json(cfg) == nlohmann::json(before));
}

TEST_CASE("The effective config is a complete document", "[config]") {
    const RunConfig cfg;
    const auto text = klasp::effective_config_json(cfg);
    const auto doc = nlohmann::json::parse(text);
    for (const char* block : {"generator", "optimizer", "forcefield",
                              "pet", "dedup", "output", "control"})
        CHECK(doc.contains(block));
    CHECK(doc.get<RunConfig>().optimizer.n_iterations == cfg.optimizer.n_iterations);
}

TEST_CASE("target_for prefers the per-SG target and falls back", "[config]") {
    RunConfig cfg;
    cfg.control.target_per_sg = 1000;
    cfg.control.targets = {{14, 50}};
    CHECK(cfg.target_for(14) == 50);
    CHECK(cfg.target_for(2) == 1000);
}

namespace {

void apply(RunConfig& cfg, const char* text) {
    klasp::apply_config_json(cfg, nlohmann::json::parse(text));
}

bool mentions(const std::vector<std::string>& problems, const std::string& text) {
    for (const auto& p : problems)
        if (p.find(text) != std::string::npos) return true;
    return false;
}

}  // namespace

TEST_CASE("A config naming a section or key klasp does not have is refused", "[config]") {
    RunConfig cfg;
    const RunConfig before;

    SECTION("an unknown section") {
        CHECK_THROWS_MATCHES(apply(cfg, R"({"optimiser": {"n_iterations": 5}})"),
                             ConfigError, MessageMatches(ContainsSubstring("'optimiser'")));
    }
    SECTION("an unknown key in a real section") {
        CHECK_THROWS_MATCHES(apply(cfg, R"({"optimizer": {"gradtol": 1e-3}})"),
                             ConfigError,
                             MessageMatches(ContainsSubstring("optimizer.gradtol")));
    }
    SECTION("an unknown key beside a real one") {
        CHECK_THROWS_AS(apply(cfg, R"({"optimizer": {"n_iterations": 5, "iterations": 5}})"),
                        ConfigError);
    }

    CHECK(nlohmann::json(cfg) == nlohmann::json(before));
}

// nlohmann's enum macro reads an unknown name as the first entry.
TEST_CASE("An enum name klasp does not know is refused, not read as the first one",
          "[config]") {
    RunConfig cfg;

    SECTION("sampler") {
        CHECK_THROWS_MATCHES(apply(cfg, R"({"optimizer": {"sampler": "metropolis"}})"),
                             ConfigError, MessageMatches(ContainsSubstring("metropolis")));
    }
    SECTION("collision method, listing the real ones") {
        CHECK_THROWS_MATCHES(apply(cfg, R"({"generator": {"collision": "fft"}})"),
                             ConfigError, MessageMatches(ContainsSubstring("fft_pack")));
    }
    SECTION("batch sizing given as a number") {
        CHECK_THROWS_AS(apply(cfg, R"({"control": {"batch_sizing": 1}})"), ConfigError);
    }
    SECTION("the same name through an override") {
        std::string err;
        CHECK_FALSE(klasp::apply_config_override(cfg, "optimizer.sampler=metropolis", err));
        CHECK_THAT(err, ContainsSubstring("metropolis"));
    }
}

TEST_CASE("The default configuration has no problems", "[config]") {
    CHECK(klasp::config_problems(RunConfig{}).empty());
}

// The streaming kernel has no Wolf self-energy term.
TEST_CASE("Wolf electrostatics are refused under the streaming optimizer", "[config]") {
    RunConfig cfg;
    cfg.forcefield.wolf = true;
    CHECK(mentions(klasp::config_problems(cfg), "forcefield.wolf"));

    cfg.optimizer.stream = false;
    CHECK(klasp::config_problems(cfg).empty());
}

// Configs written before the damped L-BFGS stage 2 was removed still name its
// settings at the trust-region values.
TEST_CASE("Retired L-BFGS keys load at their old values and nowhere else", "[config]") {
    RunConfig cfg;
    klasp::apply_config_json(cfg, nlohmann::json::parse(R"({
        "optimizer": {"trust_region": true, "max_step": 0.05, "lbfgs_memory": 5,
                      "fd_strain": false, "log_trajectory": false, "grad_tol": 1e-7},
        "control": {"debug_seed": 0}
    })"));
    CHECK(cfg.optimizer.grad_tol == 1e-7);
    CHECK_FALSE(nlohmann::json(cfg)["optimizer"].contains("trust_region"));

    const auto refused = [](const char* text) {
        RunConfig c;
        CHECK_THROWS_MATCHES(klasp::apply_config_json(c, nlohmann::json::parse(text)),
                             ConfigError, MessageMatches(ContainsSubstring("L-BFGS")));
    };
    refused(R"({"optimizer": {"trust_region": false}})");
    refused(R"({"optimizer": {"max_step": 0.1}})");
    refused(R"({"control": {"debug_seed": 3}})");

    std::string err;
    CHECK(klasp::apply_config_override(cfg, "optimizer.lbfgs_memory=5", err));
    CHECK_FALSE(klasp::apply_config_override(cfg, "optimizer.fd_strain=true", err));
    CHECK_THAT(err, ContainsSubstring("optimizer.fd_strain"));
}

TEST_CASE("The Metropolis sampler is refused off the streaming engine", "[config]") {
    RunConfig cfg;
    cfg.optimizer.sampler = klasp::SamplerKind::METROPOLIS;
    REQUIRE(klasp::config_problems(cfg).empty());

    SECTION("lockstep stage 2") {
        cfg.optimizer.stream = false;
        CHECK(mentions(klasp::config_problems(cfg), "optimizer.sampler"));
    }
    SECTION("every stage stored") {
        cfg.output.store_all_stages = true;
        CHECK(mentions(klasp::config_problems(cfg), "output.store_all_stages"));
    }
}

TEST_CASE("The ML stage is refused when nothing would reach it", "[config]") {
    RunConfig cfg;
    cfg.pet.relax = true;

    SECTION("generation only") {
        cfg.control.generate_only = true;
        CHECK(mentions(klasp::config_problems(cfg), "control.generate_only"));
    }
    SECTION("charges only") {
        cfg.forcefield.max_order = 0;
        CHECK(mentions(klasp::config_problems(cfg), "forcefield.max_order=0"));
    }
}

TEST_CASE("A model's two files are given together or not at all", "[config]") {
    RunConfig cfg;
    cfg.pet.model_json = "/models/pet-mad-xs.json";
    CHECK(mentions(klasp::config_problems(cfg), "pet.model_weights"));

    cfg.pet.model_weights = "/models/pet-mad-xs.safetensors";
    CHECK(klasp::config_problems(cfg).empty());
}

TEST_CASE("A value that cannot apply is refused, not clamped or ignored", "[config]") {
    RunConfig cfg;
    std::string key;
    SECTION("multipole rank above 4") {
        cfg.forcefield.max_order = 5;
        key = "forcefield.max_order";
    }
    SECTION("Ewald tolerance of zero") {
        cfg.forcefield.ewald_accuracy = 0;
        key = "forcefield.ewald_accuracy";
    }
    SECTION("Ewald tolerance of one or more") {
        cfg.forcefield.ewald_accuracy = 1.5;
        key = "forcefield.ewald_accuracy";
    }
    SECTION("a number that is not a space group") {
        cfg.control.space_groups = {14, 231};
        key = "control.space_groups";
    }
    SECTION("a negative time budget") {
        cfg.control.max_runtime_s = -60;
        key = "control.max_runtime_s";
    }
    SECTION("forcing a resume that was not asked for") {
        cfg.control.resume_force = true;
        key = "control.resume_force";
    }
    CHECK(mentions(klasp::config_problems(cfg), key));
}

TEST_CASE("Every problem is reported at once", "[config]") {
    RunConfig cfg;
    cfg.forcefield.wolf = true;
    cfg.forcefield.max_order = 7;

    const auto problems = klasp::config_problems(cfg);
    REQUIRE(problems.size() == 2);
    const std::string message = ConfigError(problems).what();
    CHECK_THAT(message, ContainsSubstring("forcefield.wolf"));
    CHECK_THAT(message, ContainsSubstring("forcefield.max_order"));
}

TEST_CASE("A space-group spec that is not one is refused, naming what was typed",
          "[config]") {
    SECTION("a misspelt preset") {
        CHECK_THROWS_MATCHES(klasp::parse_sg_spec("common55"), ConfigError,
                             MessageMatches(ContainsSubstring("common55")));
    }
    SECTION("a word in a list") {
        CHECK_THROWS_AS(klasp::parse_sg_spec("2,x,14"), ConfigError);
    }
    SECTION("a number that is not a space group") {
        CHECK_THROWS_MATCHES(klasp::parse_sg_spec("14,241"), ConfigError,
                             MessageMatches(ContainsSubstring("241")));
    }
    SECTION("nothing at all") {
        CHECK_THROWS_AS(klasp::parse_sg_spec(""), ConfigError);
    }
}

TEST_CASE("A space-group list and a preset still parse", "[config]") {
    CHECK(klasp::parse_sg_spec("19,2,14,2") == std::vector<int>{2, 14, 19});

    std::map<int, int> targets;
    CHECK_FALSE(klasp::parse_sg_spec("efficientz2", &targets).empty());
    CHECK_FALSE(targets.empty());
}

TEST_CASE("A run refuses a contradictory configuration before touching anything",
          "[config]") {
    const auto dir = std::filesystem::temp_directory_path() / "klasp-test-refused-run";
    std::filesystem::remove_all(dir);

    RunConfig cfg;
    cfg.control.basis_paths = {"acetic_acid.json"};
    cfg.control.space_groups = {14};
    cfg.output.dir = dir.string();
    cfg.forcefield.wolf = true;

    CHECK_THROWS_MATCHES(klasp::run_local_csp(cfg), ConfigError,
                         MessageMatches(ContainsSubstring("forcefield.wolf")));
    CHECK_FALSE(std::filesystem::exists(dir));
}

TEST_CASE("A run with nothing to run on is refused by name", "[config]") {
    const RunConfig cfg;
    CHECK_THROWS_MATCHES(klasp::prepare_run_config(cfg), ConfigError,
                         MessageMatches(ContainsSubstring("control.basis_paths") &&
                                        ContainsSubstring("control.space_groups")));
}

#ifdef KLASP_HAS_PET
namespace {

RunConfig runnable() {
    RunConfig cfg;
    cfg.control.basis_paths = {"acetic_acid.json"};
    cfg.control.space_groups = {14};
    return cfg;
}

}  // namespace

TEST_CASE("A model named by path prefix is resolved when the run is prepared", "[config]") {
    RunConfig cfg = runnable();
    cfg.pet.relax = true;
    cfg.pet.model = "/models/my-pet";

    const RunConfig prepared = klasp::prepare_run_config(cfg);
    CHECK(prepared.pet.model_json == "/models/my-pet.json");
    CHECK(prepared.pet.model_weights == "/models/my-pet.safetensors");
}

TEST_CASE("A model is found by its bare name in the search path", "[config]") {
    const auto dir = std::filesystem::temp_directory_path() / "klasp-model-names";
    std::filesystem::create_directories(dir);
    for (const char* ext : {".json", ".safetensors"}) std::ofstream(dir / (std::string("pet-test-model") + ext)) << "";
    setenv("KLASP_PET_MODEL_DIR", dir.c_str(), 1);
    std::string j, w;
    klasp::resolve_pet_model("pet-test-model", j, w);
    CHECK(j == (dir / "pet-test-model.json").string());
    CHECK(w == (dir / "pet-test-model.safetensors").string());
    CHECK_THROWS(klasp::resolve_pet_model("pet-no-such-model", j, w));
    unsetenv("KLASP_PET_MODEL_DIR");
    std::filesystem::remove_all(dir);
}

TEST_CASE("A pair of model files is used as given, whatever the name says", "[config]") {
    RunConfig cfg = runnable();
    cfg.pet.relax = true;
    cfg.pet.model = "/models/not-this-one";
    cfg.pet.model_json = "/elsewhere/a.json";
    cfg.pet.model_weights = "/elsewhere/a.safetensors";

    const RunConfig prepared = klasp::prepare_run_config(cfg);
    CHECK(prepared.pet.model_json == "/elsewhere/a.json");
    CHECK(prepared.pet.model_weights == "/elsewhere/a.safetensors");
}

// The named default model is only looked up when the ML stage is on.
TEST_CASE("A force-field run never looks for model weights", "[config]") {
    const RunConfig cfg = runnable();
    RunConfig prepared;
    CHECK_NOTHROW(prepared = klasp::prepare_run_config(cfg));
    CHECK(prepared.pet.model_json.empty());
}
#endif

TEST_CASE("A PDD tolerance without stored PDDs is refused", "[config]") {
    RunConfig cfg;
    cfg.dedup.pdd_tol = 0.1;
    CHECK(mentions(klasp::config_problems(cfg), "output.store_pdd"));

    cfg.output.store_pdd = true;
    CHECK(klasp::config_problems(cfg).empty());
}
