#include "klasp/config.hpp"

#include <fmt/format.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace klasp {

namespace {

std::string describe_problems(const std::vector<std::string>& problems) {
    std::string text = problems.size() == 1
        ? std::string("this configuration cannot run:")
        : fmt::format("this configuration cannot run ({} problems):", problems.size());
    for (const auto& p : problems) text += "\n  " + p;
    return text;
}

// The name a config file gave, for an error message. A non-string is shown as
// JSON.
std::string given_name(const nlohmann::json& j) {
    return j.is_string() ? j.get<std::string>() : j.dump();
}

[[noreturn]] void refuse_name(const nlohmann::json& j, const char* what,
                              const char* choices) {
    throw ConfigError(fmt::format("'{}' is not a {}; the choices are {}",
                                  given_name(j), what, choices));
}

// Refuse a section or key the tree does not have, by its full path.
void check_known_keys(const nlohmann::json& base, const nlohmann::json& patch,
                      const std::string& where) {
    if (!base.is_object() || !patch.is_object()) return;
    for (const auto& [key, value] : patch.items()) {
        const std::string path = where.empty() ? key : where + "." + key;
        if (!base.contains(key)) {
            std::string known;
            for (const auto& [name, unused] : base.items())
                known += (known.empty() ? "" : ", ") + name;
            if (where.empty())
                throw ConfigError(fmt::format(
                    "no such config section: '{}' (the sections are {})", path, known));
            throw ConfigError(fmt::format(
                "no such config key: '{}' ({} has {})", path, where, known));
        }
        check_known_keys(base.at(key), value, path);
    }
}

// Recursive object merge: objects merge key by key; anything else, including
// arrays, replaces what is under it. Every patch key is known to be in the base.
void merge_into(nlohmann::json& base, const nlohmann::json& patch) {
    if (!base.is_object() || !patch.is_object()) {
        base = patch;
        return;
    }
    for (const auto& [key, value] : patch.items())
        merge_into(base[key], value);
}

// Parse `value` as JSON, falling back to a string, so --set needs no quoting.
nlohmann::json parse_scalar(const std::string& value) {
    auto parsed = nlohmann::json::parse(value, nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) return nlohmann::json(value);
    return parsed;
}

// Keys the tree no longer has, each with the one value it held on the path that
// remains (the removed L-BFGS stage 2). A file holding them at that value loads;
// any other value is refused.
struct RetiredKey {
    const char* section;
    const char* key;
    nlohmann::json value;
};

const std::vector<RetiredKey>& retired_keys() {
    static const std::vector<RetiredKey> keys{
        {"optimizer", "trust_region", true},
        {"optimizer", "max_step", 0.05},
        {"optimizer", "lbfgs_memory", 5},
        {"optimizer", "fd_strain", false},
        {"optimizer", "log_trajectory", false},
        {"control", "debug_seed", 0},
    };
    return keys;
}

const RetiredKey* find_retired_key(const std::string& section, const std::string& key) {
    for (const auto& r : retired_keys())
        if (section == r.section && key == r.key) return &r;
    return nullptr;
}

std::string describe_retired_key(const RetiredKey& r, const nlohmann::json& given) {
    return fmt::format(
        "{}.{} = {} configured the L-BFGS stage-2 optimizer, which klasp no longer "
        "has; the trust region is the only one. Remove the key (it is accepted only "
        "as {}, which changes nothing)",
        r.section, r.key, given.dump(), r.value.dump());
}

// `patch` without the retired keys that hold their retired value.
nlohmann::json without_retired_keys(const nlohmann::json& patch) {
    nlohmann::json out = patch;
    for (const auto& r : retired_keys()) {
        if (!out.contains(r.section) || !out[r.section].is_object()) continue;
        auto& section = out[r.section];
        if (!section.contains(r.key)) continue;
        if (section[r.key] != r.value)
            throw ConfigError(describe_retired_key(r, section[r.key]));
        section.erase(r.key);
    }
    return out;
}

}  // namespace

ConfigError::ConfigError(const std::vector<std::string>& problems)
    : std::runtime_error(describe_problems(problems)) {}

void to_json(nlohmann::json& j, BatchSizing v) {
    j = v == BatchSizing::LEGACY ? "legacy" : "atoms";
}

void from_json(const nlohmann::json& j, BatchSizing& v) {
    const std::string name = given_name(j);
    if (name == "atoms") v = BatchSizing::ATOMS;
    else if (name == "legacy") v = BatchSizing::LEGACY;
    else refuse_name(j, "batch sizing", "atoms, legacy");
}

void to_json(nlohmann::json& j, CollisionMethod v) {
    switch (v) {
        case CollisionMethod::DIRECT:      j = "direct"; return;
        case CollisionMethod::RIGID_PRESS: j = "rigid_press"; return;
        case CollisionMethod::FFT_PACK:    j = "fft_pack"; return;
    }
    throw std::logic_error("CollisionMethod has no name for this value");
}

void from_json(const nlohmann::json& j, CollisionMethod& v) {
    const std::string name = given_name(j);
    if (name == "direct") v = CollisionMethod::DIRECT;
    else if (name == "rigid_press") v = CollisionMethod::RIGID_PRESS;
    else if (name == "fft_pack") v = CollisionMethod::FFT_PACK;
    else refuse_name(j, "collision method", "direct, rigid_press, fft_pack");
}

void to_json(nlohmann::json& j, SamplerKind v) {
    j = v == SamplerKind::METROPOLIS ? "mc" : "trust_region";
}

void from_json(const nlohmann::json& j, SamplerKind& v) {
    const std::string name = given_name(j);
    if (name == "trust_region") v = SamplerKind::TRUST_REGION;
    else if (name == "mc") v = SamplerKind::METROPOLIS;
    else refuse_name(j, "sampler", "trust_region, mc");
}

void apply_config_json(RunConfig& cfg, const nlohmann::json& patch_in) {
    const nlohmann::json patch = without_retired_keys(patch_in);
    nlohmann::json base = cfg;
    check_known_keys(base, patch, "");
    merge_into(base, patch);
    cfg = base.get<RunConfig>();
}

void load_config_file(RunConfig& cfg, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw ConfigError("cannot open config file: " + path);
    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError("config file " + path + " is not valid JSON: " + e.what());
    }
    if (!doc.is_object())
        throw ConfigError("config file " + path + " must hold a JSON object");
    try {
        apply_config_json(cfg, doc);
    } catch (const ConfigError& e) {
        throw ConfigError("config file " + path + ": " + e.what());
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError("config file " + path + " has a value of the wrong type: "
                          + e.what());
    }
}

bool apply_config_override(RunConfig& cfg, const std::string& assignment,
                           std::string& error) {
    const auto eq = assignment.find('=');
    if (eq == std::string::npos || eq == 0) {
        error = "expected section.key=value, got '" + assignment + "'";
        return false;
    }
    const std::string path = assignment.substr(0, eq);
    const std::string value = assignment.substr(eq + 1);

    // Split on '.' and check every segment against the serialised tree, so a
    // typo is reported against the name the user typed.
    nlohmann::json base = cfg;
    nlohmann::json* node = &base;
    std::string walked;
    std::istringstream segments(path);
    std::string segment;
    std::vector<std::string> parts;
    while (std::getline(segments, segment, '.')) {
        if (segment.empty()) {
            error = "empty path segment in '" + path + "'";
            return false;
        }
        parts.push_back(segment);
    }
    if (parts.size() == 2) {
        if (const RetiredKey* retired = find_retired_key(parts[0], parts[1])) {
            const nlohmann::json given = parse_scalar(value);
            if (given == retired->value) return true;
            error = describe_retired_key(*retired, given);
            return false;
        }
    }
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        if (!node->is_object() || !node->contains(parts[i])) {
            error = "no such config section: '" + walked + parts[i] + "'";
            return false;
        }
        node = &(*node)[parts[i]];
        walked += parts[i] + ".";
    }
    const std::string& leaf = parts.back();
    if (!node->is_object() || !node->contains(leaf)) {
        error = "no such config key: '" + path + "'";
        return false;
    }

    (*node)[leaf] = parse_scalar(value);
    try {
        cfg = base.get<RunConfig>();
    } catch (const ConfigError& e) {
        error = std::string(e.what()) + " (" + path + ")";
        return false;
    } catch (const nlohmann::json::exception& e) {
        error = "'" + value + "' is not a valid value for " + path + " (" + e.what() + ")";
        return false;
    }
    return true;
}

std::vector<std::string> config_problems(const RunConfig& cfg) {
    std::vector<std::string> problems;
    const auto& opt = cfg.optimizer;
    const auto& ff = cfg.forcefield;
    const auto& pet = cfg.pet;
    const auto& ctl = cfg.control;

    // The Wolf self-energy shift is applied only in the lockstep stage-2
    // optimizer; streaming with Wolf on would omit it from every energy.
    if (ff.wolf && opt.stream)
        problems.push_back(
            "forcefield.wolf (--wolf) needs optimizer.stream=false (--lockstep-stage2): "
            "the Wolf self-energy term is computed only in the lockstep optimizer, so "
            "a streaming run leaves it out and every energy comes out displaced");

    // Metropolis only exists as a streaming-engine stepper.
    if (opt.sampler == SamplerKind::METROPOLIS) {
        if (!opt.stream)
            problems.push_back(
                "optimizer.sampler=mc (--sampler mc) runs on the streaming engine and "
                "cannot be combined with optimizer.stream=false (--lockstep-stage2)");
        if (cfg.output.store_all_stages)
            problems.push_back(
                "optimizer.sampler=mc (--sampler mc) cannot be combined with "
                "output.store_all_stages (--store-all-stages), which routes stage 2 "
                "through the lockstep trust region");
    }

    // The ML stage refines multipole-relaxed structures; SGWorker skips stage 3
    // on the generate-only and charge-only paths.
    if (pet.relax) {
#ifndef KLASP_HAS_PET
        problems.push_back(
            "pet.relax (--pet-relax) needs the ML backend, and this build has none; "
            "configure with -DKLASP_ENABLE_PET=ON");
#endif
        if (ctl.generate_only)
            problems.push_back(
                "pet.relax (--pet-relax) cannot be combined with control.generate_only "
                "(--generate-only): the ML stage refines relaxed structures, and nothing "
                "is relaxed");
        if (ff.max_order == 0)
            problems.push_back(
                "pet.relax (--pet-relax) cannot be combined with forcefield.max_order=0 "
                "(--max-order 0): the ML stage refines the multipole-relaxed structures, "
                "and rank 0 stops after the charge-only pass");
    }
    if (pet.model_json.empty() != pet.model_weights.empty())
        problems.push_back(
            "pet.model_json (--model-json) and pet.model_weights (--model-weights) "
            "name one model between them; give both, or neither and name the model "
            "with pet.model (--model)");

    // csp_setup zeroes the PDD tolerance when PDDs are not stored.
    if (cfg.dedup.pdd_tol > 0 && !cfg.output.store_pdd)
        problems.push_back(
            "dedup.pdd_tol (--dedup-pdd) needs output.store_pdd=true (--store-pdd): "
            "the PDD tier compares stored PDDs, and a run without them turns it off");

    if (ff.max_order < -1 || ff.max_order > 4)
        problems.push_back(fmt::format(
            "forcefield.max_order (--max-order) is {}; it must be 0 to 4, or -1 for "
            "the basis's own rank", ff.max_order));
    if (!(ff.ewald_accuracy < 0 || (ff.ewald_accuracy > 0 && ff.ewald_accuracy < 1)))
        problems.push_back(fmt::format(
            "forcefield.ewald_accuracy (--ewald-accuracy) is {}; it must be a relative "
            "tolerance in (0, 1), or negative for the basis's own", ff.ewald_accuracy));

    for (int sg : ctl.space_groups)
        if (sg < 1 || sg > 230)
            problems.push_back(fmt::format(
                "control.space_groups has {}, which is not a space group (1 to 230)", sg));
    if (ctl.max_runtime_s < 0)
        problems.push_back(fmt::format(
            "control.max_runtime_s is {}; a wall-clock budget cannot be negative "
            "(0 means none)", ctl.max_runtime_s));
    if (ctl.resume_force && !ctl.resume)
        problems.push_back(
            "control.resume_force does nothing without control.resume; --resume-force "
            "on the command line sets both");

    return problems;
}

std::string effective_config_json(const RunConfig& cfg) {
    const nlohmann::json j = cfg;
    return j.dump(2);
}

}  // namespace klasp
