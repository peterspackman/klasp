// The nanobind half of the Python module. Includes nanobind and klasp_api.hpp
// only, no klasp headers or Kokkos; see klasp_api.hpp.
//
// The GIL is released around long-running calls so another Python thread can
// call request_stop() and the log sink can reacquire it. Exceptions from a
// Python log callback are reported on stderr, never propagated into klasp.

#include "klasp_api.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace api = klasp::pyapi;

namespace {

// energy()'s keyword options, by the names the `energy` subcommand gives them,
// applied onto default_energy_request().
using EnergySetter = std::function<void(api::EnergyRequest&, nb::handle)>;

const std::map<std::string, EnergySetter>& energy_options() {
    static const std::map<std::string, EnergySetter> setters = {
        {"relax",          [](api::EnergyRequest& r, nb::handle v) { r.relax = nb::cast<bool>(v); }},
        {"backend",        [](api::EnergyRequest& r, nb::handle v) { r.backend = nb::cast<std::string>(v); }},
        {"perturb",        [](api::EnergyRequest& r, nb::handle v) { r.perturb = nb::cast<int>(v); }},
        {"isolated",       [](api::EnergyRequest& r, nb::handle v) { r.isolated = nb::cast<bool>(v); }},
        {"fixed_cell",     [](api::EnergyRequest& r, nb::handle v) { r.fixed_cell = nb::cast<bool>(v); }},
        {"pressure_gpa",   [](api::EnergyRequest& r, nb::handle v) { r.pressure_gpa = nb::cast<double>(v); }},
        {"no_taper",       [](api::EnergyRequest& r, nb::handle v) { r.no_taper = nb::cast<bool>(v); }},
        {"max_order",      [](api::EnergyRequest& r, nb::handle v) { r.max_order = nb::cast<int>(v); }},
        {"ewald_accuracy", [](api::EnergyRequest& r, nb::handle v) { r.ewald_accuracy = nb::cast<double>(v); }},
        {"iterations",     [](api::EnergyRequest& r, nb::handle v) { r.iterations = nb::cast<int>(v); }},
        {"grad_tol",       [](api::EnergyRequest& r, nb::handle v) { r.grad_tol = nb::cast<double>(v); }},
        {"output_dir",     [](api::EnergyRequest& r, nb::handle v) { r.output_dir = nb::cast<std::string>(v); }},
        {"write_json",     [](api::EnergyRequest& r, nb::handle v) { r.write_json = nb::cast<std::string>(v); }},
        {"trajectory",     [](api::EnergyRequest& r, nb::handle v) { r.trajectory = nb::cast<bool>(v); }},
        {"summary",        [](api::EnergyRequest& r, nb::handle v) { r.summary = nb::cast<std::string>(v); }},
        {"symmetry",       [](api::EnergyRequest& r, nb::handle v) { r.symmetry = nb::cast<bool>(v); }},
        {"expand_p1",      [](api::EnergyRequest& r, nb::handle v) { r.expand_p1 = nb::cast<bool>(v); }},
        {"rigid",          [](api::EnergyRequest& r, nb::handle v) { r.rigid = nb::cast<bool>(v); }},
        {"device",         [](api::EnergyRequest& r, nb::handle v) { r.device = nb::cast<bool>(v); }},
        {"model",          [](api::EnergyRequest& r, nb::handle v) { r.model = nb::cast<std::string>(v); }},
        {"model_json",     [](api::EnergyRequest& r, nb::handle v) { r.model_json = nb::cast<std::string>(v); }},
        {"model_weights",  [](api::EnergyRequest& r, nb::handle v) { r.model_weights = nb::cast<std::string>(v); }},
    };
    return setters;
}

// The options evaluate() and optimize() take, applied onto
// default_structure_options(). evaluate() accepts only the model's own.
using StructureSetter = std::function<void(api::StructureOptions&, nb::handle)>;

const std::map<std::string, StructureSetter>& structure_options() {
    static const std::map<std::string, StructureSetter> setters = {
        {"max_order",      [](api::StructureOptions& o, nb::handle v) { o.max_order = nb::cast<int>(v); }},
        {"ewald_accuracy", [](api::StructureOptions& o, nb::handle v) { o.ewald_accuracy = nb::cast<double>(v); }},
        {"no_taper",       [](api::StructureOptions& o, nb::handle v) { o.no_taper = nb::cast<bool>(v); }},
        {"fixed_cell",     [](api::StructureOptions& o, nb::handle v) { o.fixed_cell = nb::cast<bool>(v); }},
        {"pressure_gpa",   [](api::StructureOptions& o, nb::handle v) { o.pressure_gpa = nb::cast<double>(v); }},
        {"iterations",     [](api::StructureOptions& o, nb::handle v) { o.iterations = nb::cast<int>(v); }},
        {"grad_tol",       [](api::StructureOptions& o, nb::handle v) { o.grad_tol = nb::cast<double>(v); }},
        {"perturb",        [](api::StructureOptions& o, nb::handle v) { o.perturb = nb::cast<int>(v); }},
        {"trajectory",     [](api::StructureOptions& o, nb::handle v) { o.trajectory = nb::cast<bool>(v); }},
    };
    return setters;
}

api::StructureOptions structure_request(const nb::kwargs& options, const char* what,
                                        bool model_only) {
    static const std::vector<std::string> model_names = {"max_order", "ewald_accuracy",
                                                         "no_taper"};
    const auto allowed = [&](const std::string& name) {
        return !model_only || std::find(model_names.begin(), model_names.end(), name)
                              != model_names.end();
    };
    api::StructureOptions o = api::default_structure_options();
    for (auto [key, value] : options) {
        const std::string name = nb::cast<std::string>(key);
        const auto setter = structure_options().find(name);
        if (setter == structure_options().end() || !allowed(name)) {
            std::string known;
            for (const auto& [option, unused] : structure_options())
                if (allowed(option)) known += (known.empty() ? "" : ", ") + option;
            throw nb::type_error((std::string(what) + "() has no option '" + name +
                                  "'; the options are " + known).c_str());
        }
        setter->second(o, value);
    }
    return o;
}

// generate()'s keyword options, applied onto default_generate_request().
using GenerateSetter = std::function<void(api::GenerateRequest&, nb::handle)>;

const std::map<std::string, GenerateSetter>& generate_options() {
    static const std::map<std::string, GenerateSetter> setters = {
        {"seed",              [](api::GenerateRequest& r, nb::handle v) { r.seed = nb::cast<int>(v); }},
        {"max_seeds",         [](api::GenerateRequest& r, nb::handle v) { r.max_seeds = nb::cast<int>(v); }},
        {"tvp",               [](api::GenerateRequest& r, nb::handle v) { r.tvp = nb::cast<double>(v); }},
        {"collision",         [](api::GenerateRequest& r, nb::handle v) { r.collision = nb::cast<std::string>(v); }},
        {"nudge_budget",      [](api::GenerateRequest& r, nb::handle v) { r.nudge_budget = nb::cast<int>(v); }},
        {"contract",          [](api::GenerateRequest& r, nb::handle v) { r.contract = nb::cast<bool>(v); }},
        {"fft_k_per_trial",   [](api::GenerateRequest& r, nb::handle v) { r.fft_k_per_trial = nb::cast<int>(v); }},
        {"fft_grid_spacing",  [](api::GenerateRequest& r, nb::handle v) { r.fft_grid_spacing = nb::cast<double>(v); }},
        {"fft_collision_tol", [](api::GenerateRequest& r, nb::handle v) { r.fft_collision_tol = nb::cast<double>(v); }},
        {"fft_max_aspect",    [](api::GenerateRequest& r, nb::handle v) { r.fft_max_aspect = nb::cast<double>(v); }},
    };
    return setters;
}

nb::dict outcome_to_dict(const api::StructureOutcome& s) {
    nb::dict d;
    d["label"] = s.label;
    d["space_group"] = s.space_group;
    d["n_molecules"] = s.n_molecules;
    d["n_atoms"] = s.n_atoms;
    d["atomic_numbers"] = nb::cast(s.atomic_numbers);
    d["symbols"] = nb::cast(s.symbols);
    d["initial_energy"] = s.initial_energy;
    d["energy"] = s.energy;
    nb::dict components;
    components["total"] = s.total;
    components["electrostatic"] = s.electrostatic;
    components["buckingham"] = s.buckingham;
    d["components"] = components;
    d["relaxed"] = s.relaxed;
    d["converged"] = s.converged;
    d["iterations"] = s.iterations;
    d["grad_max"] = s.grad_max;
    d["trust_radius"] = s.trust_radius;
    d["restarts"] = s.restarts;
    d["cell_params"] = nb::cast(s.cell_params);
    d["cell"] = nb::cast(s.cell);
    d["positions"] = nb::cast(s.positions);
    d["structure_json"] = s.structure_json;
    d["cif"] = s.cif;
    d["trajectory_xyz"] = s.trajectory_xyz;
    nb::list frames;
    for (const auto& f : s.trajectory) {
        nb::dict fd;
        fd["iteration"] = f.iteration;
        fd["energy"] = f.energy;
        fd["trial_energy"] = f.trial_energy;
        fd["accepted"] = f.accepted;
        fd["grad_max"] = f.grad_max;
        fd["trust_radius"] = f.trust_radius;
        fd["cell"] = nb::cast(f.cell);
        fd["positions"] = nb::cast(f.positions);
        frames.append(fd);
    }
    d["trajectory"] = frames;
    d["timings"] = nb::cast(s.timings);
    return d;
}

nb::dict row_to_dict(const api::LandscapeRow& r) {
    nb::dict d;
    d["seed"] = r.seed;
    d["space_group"] = r.space_group;
    d["stage"] = r.stage;
    d["n_mols"] = r.n_mols;
    d["energy"] = r.energy;
    d["density"] = r.density;
    d["vacuum_gap"] = r.vacuum_gap;
    d["warnings"] = r.warnings;
    d["grad_max"] = r.grad_max;
    d["iters_to_converge"] = r.iters_to_converge;
    d["cell"] = nb::cast(r.cell);
    d["translation"] = nb::cast(r.translation);
    d["rotation"] = nb::cast(r.rotation);
    if (!r.amd.empty()) d["amd"] = nb::cast(r.amd);
    if (!r.powder.empty()) d["powder"] = nb::cast(r.powder);
    return d;
}

}  // namespace

NB_MODULE(_klasp, m) {
    m.doc() = "klasp: crystal structure prediction, as a library (see klasp/__init__.py)";

    // --- Kokkos lifetime ---------------------------------------------------
    m.def("initialize",
          [](const std::vector<std::string>& args) { return api::initialize(args); },
          nb::arg("args") = std::vector<std::string>(),
          "Start Kokkos. Returns True if this call did it, False if it was "
          "already running. Raises if Kokkos was already finalized.");
    m.def("finalize", &api::finalize,
          "Shut Kokkos down and drop the log sink. Harmless if never started, "
          "harmless twice.");
    m.def("is_initialized", &api::is_initialized);
    m.def("is_finalized", &api::is_finalized);

    // --- configuration -----------------------------------------------------
    m.def("default_config_json", &api::default_config_json);
    m.def("normalize_config_json", &api::normalize_config_json, nb::arg("json"));
    m.def("apply_override", &api::apply_override, nb::arg("json"),
          nb::arg("assignment"));
    m.def("config_problems", &api::config_problems, nb::arg("json"),
          "Every reason klasp would refuse to run this configuration.");
    m.def("parse_space_groups",
          [](const std::string& spec) {
              std::map<int, int> targets;
              auto sgs = api::parse_space_groups(spec, targets);
              return nb::make_tuple(nb::cast(sgs), nb::cast(targets));
          },
          nb::arg("spec"),
          "A number, a comma list, or a preset name. Returns (space_groups, "
          "per_group_targets).");

    // --- output ------------------------------------------------------------
    m.def("set_log_sink",
          [](nb::object callback) {
              if (callback.is_none()) {
                  api::set_log_sink(nullptr);
                  return;
              }
              // Held by value: called from non-Python threads, possibly after the
              // naming expression is gone.
              api::set_log_sink([cb = std::move(callback)](int level,
                                                           const std::string& text) {
                  nb::gil_scoped_acquire gil;
                  try {
                      cb(level, text);
                  } catch (const std::exception& e) {
                      std::fprintf(stderr,
                                   "klasp: log callback raised, ignoring: %s\n",
                                   e.what());
                  } catch (...) {
                      std::fprintf(stderr, "klasp: log callback raised, ignoring\n");
                  }
              });
          },
          nb::arg("callback").none(),
          "Route klasp's narration to callback(level, text). None restores "
          "klasp's own stdout/stderr writer.");
    m.def("set_log_level", &api::set_log_level, nb::arg("level"));
    m.def("log_level", &api::log_level);

    // --- running a CSP -----------------------------------------------------
    m.def("run_csp",
          [](const std::string& config_json) {
              api::RunOutcome r;
              {
                  nb::gil_scoped_release unlock;
                  r = api::run_csp(config_json);
              }
              nb::dict d;
              d["stored"] = r.stored;
              d["unique"] = r.unique;
              d["stopped_early"] = r.stopped_early;
              return d;
          },
          nb::arg("config_json"),
          "Run a CSP in this process. No MPI is involved.");
    m.def("request_stop", &api::request_stop,
          "Ask a running CSP to finish the unit in hand and write out.");

    // --- reading a landscape -----------------------------------------------
    const api::LandscapeQuery query_defaults;
    m.def("read_landscape",
          [](const std::string& db_path, int stage, int space_group,
             double max_energy, bool unique_only, bool keep_catastrophes,
             double catastrophe_threshold, bool keep_warned,
             int only_missing_stage, bool descriptors) {
              api::LandscapeQuery q;
              q.stage = stage;
              q.space_group = space_group;
              q.max_energy = max_energy;
              q.unique_only = unique_only;
              q.keep_catastrophes = keep_catastrophes;
              q.catastrophe_threshold = catastrophe_threshold;
              q.keep_warned = keep_warned;
              q.only_missing_stage = only_missing_stage;
              q.descriptors = descriptors;

              std::vector<api::LandscapeRow> rows;
              {
                  nb::gil_scoped_release unlock;
                  rows = api::read_landscape(db_path, q);
              }
              nb::list out;
              for (const auto& r : rows) out.append(row_to_dict(r));
              return out;
          },
          nb::arg("db_path"),
          nb::arg("stage") = query_defaults.stage,
          nb::arg("space_group") = query_defaults.space_group,
          nb::arg("max_energy") = query_defaults.max_energy,
          nb::arg("unique_only") = query_defaults.unique_only,
          nb::arg("keep_catastrophes") = query_defaults.keep_catastrophes,
          nb::arg("catastrophe_threshold") = query_defaults.catastrophe_threshold,
          nb::arg("keep_warned") = query_defaults.keep_warned,
          nb::arg("only_missing_stage") = query_defaults.only_missing_stage,
          nb::arg("descriptors") = query_defaults.descriptors,
          "Rows of a landscape, sorted by energy ascending, as dicts.");

    m.def("database_meta", &api::database_meta, nb::arg("db_path"), nb::arg("key"));

    // --- a single energy ---------------------------------------------------
    m.def("energy",
          [](const std::vector<std::string>& paths, const nb::kwargs& options) {
              api::EnergyRequest req = api::default_energy_request();
              req.paths = paths;
              for (auto [key, value] : options) {
                  const std::string name = nb::cast<std::string>(key);
                  const auto setter = energy_options().find(name);
                  if (setter == energy_options().end()) {
                      std::string known;
                      for (const auto& [option, unused] : energy_options())
                          known += (known.empty() ? "" : ", ") + option;
                      throw nb::type_error(("energy() has no option '" + name +
                                            "'; the options are " + known).c_str());
                  }
                  setter->second(req, value);
              }

              api::EnergyOutcome r;
              {
                  nb::gil_scoped_release unlock;
                  r = api::energy(req);
              }
              nb::dict d;
              d["energy_kj_per_mol"] = nb::cast(r.energy_kj_per_mol);
              d["structures"] = r.structures;
              d["failed"] = r.failed;
              return d;
          },
          nb::arg("paths"), nb::arg("options"),
          "Evaluate or relax structures, taking the energy subcommand's options "
          "by name.");

    // --- one structure ----------------------------------------------------
    m.def("normalize_structure", &api::normalize_structure, nb::arg("json"));
    m.def("default_structure_options", [] {
        const api::StructureOptions o = api::default_structure_options();
        nb::dict d;
        d["max_order"] = o.max_order;
        d["ewald_accuracy"] = o.ewald_accuracy;
        d["no_taper"] = o.no_taper;
        d["fixed_cell"] = o.fixed_cell;
        d["pressure_gpa"] = o.pressure_gpa;
        d["iterations"] = o.iterations;
        d["grad_tol"] = o.grad_tol;
        d["perturb"] = o.perturb;
        d["trajectory"] = o.trajectory;
        return d;
    });
    m.def("evaluate_structure",
          [](const std::string& json, const nb::kwargs& options) {
              const api::StructureOptions o = structure_request(options, "evaluate", true);
              api::StructureOutcome r;
              {
                  nb::gil_scoped_release unlock;
                  r = api::evaluate_structure(json, o);
              }
              return outcome_to_dict(r);
          },
          nb::arg("json"), nb::arg("options"),
          "Single-point force-field energy of a structure document.");
    m.def("relax_structure",
          [](const std::string& json, const nb::kwargs& options) {
              const api::StructureOptions o = structure_request(options, "optimize", false);
              std::vector<api::StructureOutcome> r;
              {
                  nb::gil_scoped_release unlock;
                  r = api::relax_structure(json, o);
              }
              nb::list out;
              for (const auto& s : r) out.append(outcome_to_dict(s));
              return out;
          },
          nb::arg("json"), nb::arg("options"),
          "Relax a structure document; the structure as given first, then any perturbed starts.");
    m.def("landscape_structure", &api::landscape_structure, nb::arg("db_path"),
          nb::arg("seed"), nb::arg("space_group"), nb::arg("stage"));

    // --- structure generation ----------------------------------------------
    m.def("default_generate_options", [] {
        const api::GenerateRequest r = api::default_generate_request();
        nb::dict d;
        d["seed"] = r.seed;
        d["max_seeds"] = r.max_seeds;
        d["tvp"] = r.tvp;
        d["collision"] = r.collision;
        d["nudge_budget"] = r.nudge_budget;
        d["contract"] = r.contract;
        d["fft_k_per_trial"] = r.fft_k_per_trial;
        d["fft_grid_spacing"] = r.fft_grid_spacing;
        d["fft_collision_tol"] = r.fft_collision_tol;
        d["fft_max_aspect"] = r.fft_max_aspect;
        return d;
    });
    m.def("generate_structures",
          [](const std::vector<std::string>& basis, int space_group, int count,
             const nb::kwargs& options) {
              api::GenerateRequest req = api::default_generate_request();
              req.space_group = space_group;
              req.count = count;
              for (auto [key, value] : options) {
                  const std::string name = nb::cast<std::string>(key);
                  const auto setter = generate_options().find(name);
                  if (setter == generate_options().end()) {
                      std::string known;
                      for (const auto& [option, unused] : generate_options())
                          known += (known.empty() ? "" : ", ") + option;
                      throw nb::type_error(("generate() has no option '" + name +
                                            "'; the options are " + known).c_str());
                  }
                  setter->second(req, value);
              }
              std::vector<std::string> docs;
              {
                  nb::gil_scoped_release unlock;
                  docs = api::generate_structures(basis, req);
              }
              return docs;
          },
          nb::arg("basis"), nb::arg("space_group"), nb::arg("count"), nb::arg("options"),
          "Draw structures from the CSP generator, in memory.");

    // --- provenance --------------------------------------------------------
    m.attr("__version__") = api::version_string();
    m.attr("git_hash") = api::git_hash();
    m.attr("backend") = KLASP_PY_BACKEND;
#ifdef KLASP_HAS_PET
    m.attr("has_pet") = true;
#else
    m.attr("has_pet") = false;
#endif
}
