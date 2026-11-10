#pragma once

// The seam between klasp and nanobind. This header uses only the standard
// library, so bindings.cpp can include it and nanobind without Kokkos and be
// compiled host-only under nvcc_wrapper; klasp_api.cpp holds the klasp side.
// Configurations cross as JSON strings.

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace klasp::pyapi {

// --- Kokkos lifetime -------------------------------------------------------
// Both idempotent. Initialising again after finalize throws.
bool initialize(const std::vector<std::string>& args);
void finalize();
bool is_initialized();
bool is_finalized();

// --- configuration ---------------------------------------------------------
// The default tree, every key present.
std::string default_config_json();
// Parse and re-emit: unknown keys, unknown enum names and wrong types throw.
std::string normalize_config_json(const std::string& json);
// One `section.key=value`, as --set takes it. Throws naming the assignment.
std::string apply_override(const std::string& json, const std::string& assignment);
// Every reason klasp would refuse to run this configuration, one sentence each,
// as `klasp csp` applies them; empty means none. Missing inputs or space groups
// are not reported. Parse errors throw std::invalid_argument (ValueError).
std::vector<std::string> config_problems(const std::string& json);
// A space-group spec: a number, a comma list, or a named preset. Fills
// `targets` when the preset carries per-group targets.
std::vector<int> parse_space_groups(const std::string& spec,
                                    std::map<int, int>& targets);

// --- output ----------------------------------------------------------------
// Replace klasp's log sink. `sink` receives (level, text), text already
// formatted with its newline. An empty std::function restores the default.
void set_log_sink(std::function<void(int, const std::string&)> sink);
void set_log_level(int level);
int log_level();

// --- running a CSP ---------------------------------------------------------
struct RunOutcome {
    long long stored = 0;
    long long unique = 0;
    bool stopped_early = false;
};
// In-process, single-process CSP. Throws std::runtime_error for a refused run.
RunOutcome run_csp(const std::string& config_json);
// Ask a running CSP to finish the unit in hand and write out. Thread-safe.
void request_stop();

// --- reading a landscape ---------------------------------------------------
struct LandscapeRow {
    int seed = 0;
    int space_group = 0;
    int stage = 0;
    int n_mols = 1;
    double energy = 0;
    double density = 0;
    double vacuum_gap = -1;
    int warnings = 0;
    double grad_max = -1;
    int iters_to_converge = -1;
    std::vector<double> cell;         // a,b,c (Angstrom), alpha,beta,gamma (deg)
    std::vector<double> translation;  // 3 per molecule, fractional
    std::vector<double> rotation;     // 3 per molecule, angle-axis
    // Empty unless descriptors were asked for.
    std::vector<double> amd;
    std::vector<double> powder;
};

// Filters named and defaulted as `extract` and `prune` name theirs: structures
// below the catastrophe threshold or carrying a warning are dropped unless kept.
struct LandscapeQuery {
    int stage = 0;              // 0 = the highest stage present
    int space_group = 0;        // 0 = every group
    double max_energy = 1e30;
    bool unique_only = true;
    bool keep_catastrophes = false;
    double catastrophe_threshold = -1000.0;
    bool keep_warned = false;
    int only_missing_stage = 0;
    bool descriptors = false;
};

std::vector<LandscapeRow> read_landscape(const std::string& db_path,
                                         const LandscapeQuery& q);

// A value from the database's meta table (`config_json`, `basis_json`, ...).
std::string database_meta(const std::string& db_path, const std::string& key);

// --- a single energy -------------------------------------------------------
// Every option the `energy` subcommand takes, by the same names. Start from
// default_energy_request(), which reads EnergyOptions' defaults.
struct EnergyRequest {
    std::vector<std::string> paths;
    bool relax;
    std::string backend;
    int perturb;
    bool isolated;
    bool fixed_cell;
    double pressure_gpa;
    bool no_taper;
    int max_order;
    double ewald_accuracy;
    int iterations;
    double grad_tol;
    std::string output_dir;
    std::string write_json;
    bool trajectory;
    std::string summary;
    bool symmetry;
    bool expand_p1;
    bool rigid;
    bool device;
    std::string model;
    std::string model_json;
    std::string model_weights;
};
EnergyRequest default_energy_request();
struct EnergyOutcome {
    std::vector<double> energy_kj_per_mol;  // NaN where a structure failed
    int structures = 0;
    int failed = 0;
};
EnergyOutcome energy(const EnergyRequest& request);

// --- one structure ----------------------------------------------------------
// A structure crosses as its JSON document (see klasp/ff_structure.hpp). Start
// from default_structure_options(), which reads FFRelaxOptions' defaults.
struct StructureOptions {
    int max_order;
    double ewald_accuracy;
    bool no_taper;
    bool fixed_cell;
    double pressure_gpa;
    int iterations;
    double grad_tol;
    int perturb;
    bool trajectory;
};
StructureOptions default_structure_options();

struct StructureFrame {
    int iteration = 0;
    double energy = 0;
    double trial_energy = 0;
    bool accepted = true;
    double grad_max = 0;
    double trust_radius = 0;
    std::vector<double> cell;       // 9, row-major; the columns are the lattice vectors
    std::vector<double> positions;  // 3 per atom, Angstrom
};

struct StructureOutcome {
    std::string label;
    int space_group = 0;
    int n_molecules = 0;
    int n_atoms = 0;
    std::vector<int> atomic_numbers;
    std::vector<std::string> symbols;
    double initial_energy = 0;
    double energy = 0;
    double total = 0;
    double electrostatic = 0;
    double buckingham = 0;
    bool relaxed = false;
    bool converged = true;
    int iterations = 0;
    double grad_max = -1;
    double trust_radius = 0;
    int restarts = 0;
    std::vector<double> cell_params;  // a, b, c, alpha, beta, gamma
    std::vector<double> cell;
    std::vector<double> positions;
    std::string structure_json;       // empty when the energy was not finite
    std::string cif;
    std::string trajectory_xyz;       // empty unless a trajectory was recorded
    std::vector<StructureFrame> trajectory;
    std::map<std::string, double> timings;  // seconds; see FFTimings
};

std::string normalize_structure(const std::string& json);
StructureOutcome evaluate_structure(const std::string& json, const StructureOptions& options);
std::vector<StructureOutcome> relax_structure(const std::string& json,
                                              const StructureOptions& options);
// One landscape row as a structure document, with the basis the run recorded.
std::string landscape_structure(const std::string& db_path, int seed, int space_group,
                                int stage);

// --- structure generation ----------------------------------------------------
// Every GenerateOptions field, by name; start from default_generate_request().
struct GenerateRequest {
    int space_group;
    int count;
    int seed;
    int max_seeds;
    double tvp;
    std::string collision;
    int nudge_budget;
    bool contract;
    int fft_k_per_trial;
    double fft_grid_spacing;
    double fft_collision_tol;
    double fft_max_aspect;
};
GenerateRequest default_generate_request();
// One structure document (JSON text) per structure; `basis` holds one basis
// document per molecule type.
std::vector<std::string> generate_structures(const std::vector<std::string>& basis,
                                             const GenerateRequest& request);

// --- provenance ------------------------------------------------------------
std::string version_string();
std::string git_hash();

}  // namespace klasp::pyapi
