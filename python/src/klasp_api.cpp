// The klasp half of the Python module: klasp headers, Kokkos and nlohmann::json,
// no nanobind (see klasp_api.hpp).

#include "klasp_api.hpp"

#include "klasp/cmd_energy.hpp"
#include "klasp/constants.hpp"
#include "klasp/config.hpp"
#include "klasp/csp_run.hpp"
#include "klasp/database.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/dedup.hpp"
#include "klasp/element.hpp"
#include "klasp/ff_structure.hpp"
#include "klasp/io.hpp"
#include "klasp/structure_generation.hpp"
#include "klasp/log.hpp"
#include "klasp/version.hpp"

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

#include <fmt/format.h>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace klasp::pyapi {

namespace {

// Kokkos wants a mutable argv valid for the duration of the call.
struct ArgvHolder {
    std::vector<std::string> storage;
    std::vector<char*> argv;

    explicit ArgvHolder(const std::vector<std::string>& args) {
        storage.reserve(args.size() + 1);
        storage.emplace_back("klasp");
        for (const auto& a : args) storage.push_back(a);
        argv.reserve(storage.size());
        for (auto& s : storage) argv.push_back(s.data());
    }
    int argc() const { return static_cast<int>(argv.size()); }
};

}  // namespace

// --- Kokkos lifetime -------------------------------------------------------

bool initialize(const std::vector<std::string>& args) {
    if (Kokkos::is_finalized())
        throw std::runtime_error(
            "klasp.initialize: Kokkos has already been finalized in this "
            "process and cannot be started again. Do not call finalize() until "
            "you are done with klasp for good.");
    if (Kokkos::is_initialized()) return false;
    ArgvHolder holder(args);
    int argc = holder.argc();
    Kokkos::initialize(argc, holder.argv.data());
    return true;
}

void finalize() {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized()) return;
    // The sink may hold a Python callable; drop it before the interpreter goes.
    logging::set_sink(nullptr);
    Kokkos::finalize();
}

bool is_initialized() { return Kokkos::is_initialized(); }
bool is_finalized() { return Kokkos::is_finalized(); }

// --- configuration ---------------------------------------------------------

std::string default_config_json() {
    const RunConfig cfg;
    return nlohmann::json(cfg).dump();
}

namespace {

// A configuration parsed as `klasp csp --config` parses one. Rejected values
// throw std::invalid_argument (ValueError in Python).
RunConfig parse_config(const std::string& json) {
    RunConfig cfg;
    try {
        apply_config_json(cfg, nlohmann::json::parse(json));
    } catch (const ConfigError& e) {
        throw std::invalid_argument(e.what());
    } catch (const nlohmann::json::exception& e) {
        throw std::invalid_argument(std::string("not a valid configuration: ") + e.what());
    }
    return cfg;
}

}  // namespace

std::string normalize_config_json(const std::string& json) {
    return nlohmann::json(parse_config(json)).dump();
}

std::string apply_override(const std::string& json, const std::string& assignment) {
    RunConfig cfg = parse_config(json);
    std::string error;
    if (!apply_config_override(cfg, assignment, error))
        throw std::invalid_argument(error);
    return nlohmann::json(cfg).dump();
}

std::vector<std::string> config_problems(const std::string& json) {
    return klasp::config_problems(parse_config(json));
}

std::vector<int> parse_space_groups(const std::string& spec,
                                    std::map<int, int>& targets) {
    try {
        return parse_sg_spec(spec, &targets);
    } catch (const ConfigError& e) {
        throw std::invalid_argument(e.what());
    }
}

// --- output ----------------------------------------------------------------

void set_log_sink(std::function<void(int, const std::string&)> sink) {
    if (!sink) {
        logging::set_sink(nullptr);
        return;
    }
    logging::set_sink([fn = std::move(sink)](logging::Level lv,
                                             std::string_view text) {
        fn(static_cast<int>(lv), std::string(text));
    });
}

void set_log_level(int level) {
    if (level < 0) level = 0;
    if (level > 4) level = 4;
    logging::set_level(static_cast<logging::Level>(level));
}

int log_level() { return static_cast<int>(logging::level()); }

// --- running a CSP ---------------------------------------------------------

RunOutcome run_csp(const std::string& config_json) {
    if (!Kokkos::is_initialized())
        throw std::runtime_error(
            "klasp.run: Kokkos is not initialized; call klasp.initialize()");

    const RunConfig cfg = parse_config(config_json);

    CspRunOptions options;
    // request_stop() is the supported way to stop a run from Python.
    options.install_signal_handlers = false;

    try {
        const CspRunResult r = run_local_csp(cfg, options);
        RunOutcome out;
        out.stored = r.stored;
        out.unique = r.unique;
        out.stopped_early = r.stopped_early;
        return out;
    } catch (const ConfigError& e) {
        throw std::invalid_argument(e.what());
    } catch (const RunRefused& e) {
        throw std::runtime_error(std::string("klasp refused the run: ") + e.what());
    }
}

void request_stop() { request_run_stop(); }

// --- reading a landscape ---------------------------------------------------

std::vector<LandscapeRow> read_landscape(const std::string& db_path,
                                         const LandscapeQuery& q) {
    const auto rows = klasp::read_landscape(
        db_path, q.stage, q.space_group, q.max_energy, q.unique_only,
        /*exclude_catastrophes=*/!q.keep_catastrophes, q.catastrophe_threshold,
        /*exclude_warned=*/!q.keep_warned, q.only_missing_stage);

    std::vector<LandscapeRow> out;
    out.reserve(rows.size());
    for (const auto& r : rows) {
        LandscapeRow row;
        row.seed = r.seed;
        row.space_group = r.space_group;
        row.stage = r.stage;
        row.n_mols = r.n_mols;
        row.energy = r.energy;
        row.density = r.density;
        row.vacuum_gap = r.vacuum_gap;
        row.warnings = r.warnings;
        row.grad_max = r.grad_max;
        row.iters_to_converge = r.iters_to_converge;
        // Stored in Bohr; this module speaks Angstrom.
        row.cell = {r.cell_params[0] * BOHR_TO_ANGSTROM, r.cell_params[1] * BOHR_TO_ANGSTROM,
                    r.cell_params[2] * BOHR_TO_ANGSTROM, r.cell_params[3],
                    r.cell_params[4], r.cell_params[5]};
        const int n = r.n_mols > 0 ? r.n_mols : 1;
        row.translation.assign(r.translation, r.translation + 3 * n);
        row.rotation.assign(r.rotation, r.rotation + 3 * n);
        if (q.descriptors) {
            row.amd.assign(r.amd, r.amd + AMD_K);
            row.powder.assign(r.powder, r.powder + POWDER_N_BINS);
        }
        out.push_back(std::move(row));
    }
    return out;
}

std::string database_meta(const std::string& db_path, const std::string& key) {
    CSPDatabase db;
    db.open(db_path);
    return db.get_meta(key);
}

// --- a single energy -------------------------------------------------------

EnergyRequest default_energy_request() {
    const EnergyOptions o;
    EnergyRequest r;
    r.relax = o.relax;
    r.backend = o.backend;
    r.perturb = o.perturb;
    r.isolated = o.isolated;
    r.fixed_cell = o.fixed_cell;
    r.pressure_gpa = o.pressure_gpa;
    r.no_taper = o.no_taper;
    r.max_order = o.max_order;
    r.ewald_accuracy = o.ewald_accuracy;
    r.iterations = o.iterations;
    r.grad_tol = o.grad_tol;
    r.output_dir = o.output_dir;
    r.write_json = o.write_json;
    r.trajectory = o.trajectory;
    r.summary = o.summary;
    r.symmetry = o.symmetry;
    r.expand_p1 = o.expand_p1;
    r.rigid = o.rigid;
    r.device = o.device;
    r.model = o.model;
    r.model_json = o.model_json;
    r.model_weights = o.model_st;
    return r;
}

EnergyOutcome energy(const EnergyRequest& request) {
    if (!Kokkos::is_initialized())
        throw std::runtime_error(
            "klasp.energy: Kokkos is not initialized; call klasp.initialize()");

    EnergyOptions o;
    o.relax = request.relax;
    o.backend = request.backend;
    o.perturb = request.perturb;
    o.isolated = request.isolated;
    o.fixed_cell = request.fixed_cell;
    o.pressure_gpa = request.pressure_gpa;
    o.no_taper = request.no_taper;
    o.max_order = request.max_order;
    o.ewald_accuracy = request.ewald_accuracy;
    o.iterations = request.iterations;
    o.grad_tol = request.grad_tol;
    o.output_dir = request.output_dir;
    o.write_json = request.write_json;
    o.trajectory = request.trajectory;
    o.summary = request.summary;
    o.symmetry = request.symmetry;
    o.expand_p1 = request.expand_p1;
    o.rigid = request.rigid;
    o.device = request.device;
    o.model = request.model;
    o.model_json = request.model_json;
    o.model_st = request.model_weights;

    const EnergyResult r = cmd_energy(request.paths, o);
    EnergyOutcome out;
    out.energy_kj_per_mol = r.energy_kj_per_mol;
    out.structures = r.structures;
    out.failed = r.failed;
    return out;
}

// --- one structure ----------------------------------------------------------

namespace {

FFRelaxOptions to_relax_options(const StructureOptions& s) {
    FFRelaxOptions o;
    o.model.max_order = s.max_order;
    o.model.ewald_accuracy = s.ewald_accuracy;
    o.model.no_taper = s.no_taper;
    o.fixed_cell = s.fixed_cell;
    o.pressure_gpa = s.pressure_gpa;
    o.iterations = s.iterations;
    o.grad_tol = s.grad_tol;
    o.perturb = s.perturb;
    o.trajectory = s.trajectory;
    return o;
}

// A malformed document throws std::invalid_argument (ValueError).
nlohmann::json parse_structure(const std::string& text) {
    nlohmann::json doc = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded()) throw std::invalid_argument("a structure document must be JSON");
    return doc;
}

template <typename Fn>
auto with_document_errors(Fn&& fn) {
    try {
        return fn();
    } catch (const nlohmann::json::exception& e) {
        throw std::invalid_argument(std::string("the structure document is missing or has a "
                                                "wrongly typed field: ") + e.what());
    }
}

void require_kokkos(const char* what) {
    if (!Kokkos::is_initialized())
        throw std::runtime_error(std::string(what) +
                                 ": Kokkos is not initialized; call klasp.initialize()");
}

StructureOutcome outcome_of(const FFStructure& s) {
    StructureOutcome o;
    o.label = s.label;
    o.space_group = s.space_group;
    o.n_molecules = s.n_molecules;
    o.n_atoms = s.n_atoms;
    o.atomic_numbers = s.atomic_numbers;
    for (int z : s.atomic_numbers) o.symbols.emplace_back(element_symbol(z));
    o.initial_energy = s.initial_energy;
    o.energy = s.energy;
    o.total = s.components.total;
    o.electrostatic = s.components.electrostatic;
    o.buckingham = s.components.buckingham;
    o.relaxed = s.relaxed;
    o.converged = s.converged;
    o.iterations = s.iterations;
    o.grad_max = s.grad_max;
    o.trust_radius = s.trust_radius;
    o.restarts = s.restarts;
    o.cell_params.assign(s.cell_params.begin(), s.cell_params.end());
    o.cell.assign(s.cell.begin(), s.cell.end());
    o.positions = s.positions;
    o.timings = timings_by_name(s.timings);
    if (!s.structure.is_null()) {
        o.structure_json = s.structure.dump();
        std::ostringstream cif;
        write_structure_cif(cif, s, s.structure.value("title", std::string("structure")));
        o.cif = cif.str();
    }
    if (!s.trajectory.empty()) {
        std::ostringstream xyz;
        write_trajectory_xyz(xyz, s);
        o.trajectory_xyz = xyz.str();
    }
    o.trajectory.reserve(s.trajectory.size());
    for (const auto& f : s.trajectory) {
        StructureFrame fr;
        fr.iteration = f.iteration;
        fr.energy = f.energy;
        fr.trial_energy = f.trial_energy;
        fr.accepted = f.accepted;
        fr.grad_max = f.grad_max;
        fr.trust_radius = f.trust_radius;
        fr.cell.assign(f.cell.begin(), f.cell.end());
        fr.positions = f.positions;
        o.trajectory.push_back(std::move(fr));
    }
    return o;
}

}  // namespace

StructureOptions default_structure_options() {
    const FFRelaxOptions o;
    StructureOptions s;
    s.max_order = o.model.max_order;
    s.ewald_accuracy = o.model.ewald_accuracy;
    s.no_taper = o.model.no_taper;
    s.fixed_cell = o.fixed_cell;
    s.pressure_gpa = o.pressure_gpa;
    s.iterations = o.iterations;
    s.grad_tol = o.grad_tol;
    s.perturb = o.perturb;
    s.trajectory = o.trajectory;
    return s;
}

std::string normalize_structure(const std::string& json) {
    return normalize_structure_json(parse_structure(json)).dump();
}

StructureOutcome evaluate_structure(const std::string& json, const StructureOptions& options) {
    require_kokkos("klasp.evaluate");
    const nlohmann::json doc = parse_structure(json);
    return with_document_errors(
        [&] { return outcome_of(ff_evaluate(doc, to_relax_options(options).model)); });
}

std::vector<StructureOutcome> relax_structure(const std::string& json,
                                              const StructureOptions& options) {
    require_kokkos("klasp.optimize");
    const nlohmann::json doc = parse_structure(json);
    return with_document_errors([&] {
        std::vector<StructureOutcome> out;
        for (const auto& s : ff_relax(doc, to_relax_options(options)))
            out.push_back(outcome_of(s));
        return out;
    });
}

std::string landscape_structure(const std::string& db_path, int seed, int space_group,
                                int stage) {
    // Every row: a caller asking for a seed by name wants that seed.
    const auto rows = klasp::read_landscape(db_path, stage, space_group, 1e30,
                                            /*unique_only=*/false,
                                            /*exclude_catastrophes=*/false, 0.0,
                                            /*exclude_warned=*/false);
    for (const auto& r : rows) {
        if (r.seed != seed || r.space_group != space_group) continue;
        const BasisConfig basis = load_csp_basis("", db_path);
        nlohmann::json doc = build_crystal_json(basis, r.space_group, r.cell_params,
                                                r.translation, r.rotation,
                                                std::max(1, r.n_mols), r.energy);
        doc["title"] = fmt::format("seed {} in space group {}, stage {}", r.seed,
                                   r.space_group, r.stage);
        return normalize_structure_json(doc).dump();
    }
    throw std::invalid_argument(fmt::format(
        "{} has no structure with seed {} in space group {}{}", db_path, seed, space_group,
        stage > 0 ? fmt::format(" at stage {}", stage) : std::string()));
}

// --- structure generation ----------------------------------------------------

GenerateRequest default_generate_request() {
    const GenerateOptions o;
    GenerateRequest r;
    r.space_group = o.space_group;
    r.count = o.count;
    r.seed = o.seed;
    r.max_seeds = o.max_seeds;
    r.tvp = o.generator.tvp;
    r.collision = nlohmann::json(o.generator.collision).get<std::string>();
    r.nudge_budget = o.generator.nudge_budget;
    r.contract = o.generator.contract;
    r.fft_k_per_trial = o.generator.fft_k_per_trial;
    r.fft_grid_spacing = o.generator.fft_grid_spacing;
    r.fft_collision_tol = o.generator.fft_collision_tol;
    r.fft_max_aspect = o.generator.fft_max_aspect;
    return r;
}

std::vector<std::string> generate_structures(const std::vector<std::string>& basis,
                                             const GenerateRequest& request) {
    require_kokkos("klasp.generate");
    GenerateOptions o;
    o.space_group = request.space_group;
    o.count = request.count;
    o.seed = request.seed;
    o.max_seeds = request.max_seeds;
    o.generator.tvp = request.tvp;
    try {
        o.generator.collision = nlohmann::json(request.collision).get<CollisionMethod>();
    } catch (const ConfigError& e) {
        throw std::invalid_argument(e.what());
    }
    o.generator.nudge_budget = request.nudge_budget;
    o.generator.contract = request.contract;
    o.generator.fft_k_per_trial = request.fft_k_per_trial;
    o.generator.fft_grid_spacing = request.fft_grid_spacing;
    o.generator.fft_collision_tol = request.fft_collision_tol;
    o.generator.fft_max_aspect = request.fft_max_aspect;

    std::vector<nlohmann::json> docs;
    for (const auto& text : basis) {
        nlohmann::json doc = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
        if (doc.is_discarded()) throw std::invalid_argument("a basis document must be JSON");
        docs.push_back(std::move(doc));
    }
    return with_document_errors([&] {
        std::vector<std::string> out;
        for (const auto& g : klasp::generate_structures(docs, o)) out.push_back(g.structure.dump());
        return out;
    });
}

// --- provenance ------------------------------------------------------------

std::string version_string() {
    return std::string(GIT_HASH_SHORT) + (GIT_DIRTY ? "-dirty" : "") + " (" +
           GIT_BRANCH + ", built " + BUILD_DATE + ")";
}

std::string git_hash() { return GIT_HASH; }

}  // namespace klasp::pyapi
