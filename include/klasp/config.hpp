#pragma once

// One configuration tree for a CSP run. Defaults are the member initialisers;
// the tree round-trips through JSON; the resume fingerprint is an explicit
// projection of the values that change the answer (see canonical_run_config).

#include "klasp/csp.hpp"
#include "klasp/dedup.hpp"
#include "klasp/warnings.hpp"

#include <nlohmann/json.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace klasp {

// A configuration klasp will not run: an unknown key or name, or contradictory
// settings. Distinct from run failures (usage error, exit 2).
class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

    // One problem per line.
    explicit ConfigError(const std::vector<std::string>& problems);
};

// How the batch size is chosen for a launch.
enum class BatchSizing { ATOMS, LEGACY };

// Which stepper the streaming engine runs.
//   TRUST_REGION  descend to the nearest local minimum, finish on a small gradient.
//   METROPOLIS    Boltzmann-accepted moves of rigid-body and strain DOFs; finish
//                 after a fixed number of moves and keep the best structure seen.
enum class SamplerKind { TRUST_REGION, METROPOLIS };

// Enums are written and read by name; an unknown name is refused.
void to_json(nlohmann::json& j, BatchSizing v);
void from_json(const nlohmann::json& j, BatchSizing& v);
void to_json(nlohmann::json& j, CollisionMethod v);
void from_json(const nlohmann::json& j, CollisionMethod& v);
void to_json(nlohmann::json& j, SamplerKind v);
void from_json(const nlohmann::json& j, SamplerKind& v);

// ---------------------------------------------------------------------------
// Where structures come from
// ---------------------------------------------------------------------------
struct GeneratorConfig {
    // Target volume packing coefficient: the generator aims for a cell this
    // many times the summed molecular volume.
    double tvp = 1.2;

    CollisionMethod collision = CollisionMethod::DIRECT;

    // Per-seed hull-MTV centre-of-mass nudge budget; 0 disables the nudge.
    int nudge_budget = 16;

    // cspy-style expand-then-contract densification: shrink the cell back to the
    // clash boundary once the structure is clear.
    bool contract = true;

    // Rigid-press stage 0 (only when collision == RIGID_PRESS).
    int    rp_iterations = 20;
    double rp_pressure_gpa = 5.0;
    double rp_max_strain_step = 0.05;

    // FFT packing (only when collision == FFT_PACK).
    // FFT packing (only when collision == FFT_PACK). A trial emits up to
    // fft_k_per_trial structures with ids [t*k, t*k + k), so k is part of the
    // resume check.
    int    fft_k_per_trial = 8;
    double fft_grid_spacing = 0.5;    // Angstrom
    double fft_collision_tol = 0.1;   // A^3 of overlap the map will still pass on
    double fft_max_aspect = 5.0;      // reject cells longer than this, any axis pair
};

// ---------------------------------------------------------------------------
// The force-field relaxation
// ---------------------------------------------------------------------------
struct OptimizerConfig {
    int    n_iterations = 150;          // stage-2 (multipole) cap
    int    n_iterations_stage1 = 100;   // stage-1 (charge-only) cap
    double stage1_pressure_gpa = 0.0;   // external pressure, stage 1 only
    // With forcefield.mp_single: the scaled-gradient level below which a slot
    // switches its multipole term from single to double precision. Below about
    // 1e-5 the trust region churns on single-precision noise.
    double mp_single_switch_tol = 1e-5;

    // Convergence on |grad|_inf over the scaled DOFs; 0 disables. In streaming
    // stage 2 this is the harvest trigger.
    double grad_tol = 1e-8;

    bool   fixed_cell = false;
    bool   fix_angles_stage1 = true;

    // Pipelined stage 2: refill a slot as soon as one converges instead of
    // running the whole pool in lockstep.
    bool   stream = true;
    int    stream_pool_mult = 4;
    int    refill_interval = 2;
    int    max_iters_per_slot = 0;
    bool   stream_stage1 = true;

    // Take the rigid-body gradient from the pair kernel's own contraction
    // rather than scattering per-atom forces for the chain rule to gather.
    bool   contract_pairs = true;

    // Metropolis requires the streaming engine, so it refuses --lockstep-stage2.
    SamplerKind sampler = SamplerKind::TRUST_REGION;

    // Metropolis controls; ignored unless sampler == METROPOLIS.
    // Temperature in K; the criterion is exp(-dE/RT) with energies in kJ/mol.
    double mc_temperature_k = 300.0;
    // Moves per structure; the stepper finishes on this instead of a convergence test.
    int    mc_moves = 400;
    // Move sizes. Translations in fractional coordinates (wrapped); rotations
    // in degrees, composed with the current orientation; strain dimensionless,
    // in the lattice-allowed subspace only.
    double mc_step_trans = 0.03;
    double mc_step_rot_deg = 5.0;
    double mc_step_strain = 0.01;
    // Shortest intermolecular contact a Metropolis proposal may have, Angstrom;
    // 0 disables. Required because the force field is unbounded below at short
    // range. See McSettings.
    double mc_min_contact = 1.5;
};

// ---------------------------------------------------------------------------
// The force field itself
// ---------------------------------------------------------------------------
struct ForceFieldConfig {
    // Upper bound on the multipole rank; -1 means "take the input's".
    int    max_order = -1;

    // Relative truncation target for both Ewald sums; negative means "use
    // settings.ewald_accuracy from the basis JSON".
    double ewald_accuracy = -1.0;

    // Wolf damped-shifted-force electrostatics instead of Ewald: no reciprocal
    // sum at all.
    bool   wolf = false;
    double wolf_cutoff = 0;             // Bohr; 0 = use the Ewald real cutoff

    // Cutoff for the higher-multipole term (Angstrom), tapered to zero over its
    // last 2 A; 0 = use the table's. For timing only, not ranking: the
    // charge-quadrupole part decays as 1/r^3 and truncating it reorders structures.
    double mp_cutoff_ang = 0;
    // Run the higher-multipole term in single precision (EnergyModelParams::mp_single).
    bool mp_single = false;
    // Evaluate it through pair moments (EnergyModelParams::mp_pair_moments).
    bool mp_pair_moments = true;
};

// ---------------------------------------------------------------------------
// The machine-learned refinement stage
// ---------------------------------------------------------------------------
struct PetConfig {
    bool relax = false;                 // run stage 3 at all
    bool symmetry = true;               // relax in the symmetry-adapted DOFs
    bool dedup = true;                  // one structure per basin reaches PET
    bool precondition = true;
    int  pass_rounds = 0;               // 0 = run to convergence

    // Which model: a short alias (xs, mols), a canonical name, or a path prefix,
    // resolved at run start only if the stage runs. model_json and
    // model_weights name the pair directly, together or not at all, and win.
    std::string model = "xs";
    std::string model_json;
    std::string model_weights;

    int    iterations = 0;              // 0 = the backend's default
    double force_tol = 0.0;             // 0 = the backend's default
    double stress_tol = 0.0;
    double min_contact = 0.7;           // reject candidates closer than this

    // Relax on the device rather than the host. KLASP_PET_NODEVICE is still
    // honoured but deprecated. Not in the resume fingerprint.
    bool device = true;

    // Cap the atoms in one sub-batch; 0 sizes it from the device's free memory.
    int max_batch_atoms = 0;
};

// ---------------------------------------------------------------------------
// What is written, and where
// ---------------------------------------------------------------------------
struct OutputConfig {
    std::string dir = ".";

    // Persist every stage, not just the one the landscape is ranked on.
    bool store_all_stages = false;
    bool store_pdd = false;
    bool store_diagnostics = true;   // --no-store-diagnostics turns it off

    bool perf = false;                  // per-batch and per-kernel CSVs
    // With perf: fence after every kernel so GPU timings are per kernel.
    // Serialises the device.
    bool perf_sync = false;

    // Largest vacuum gap (Angstrom) before a structure is dropped. Negative
    // derives it from the potential cutoff (VACUUM_GAP_CUTOFF_FRACTION) once
    // the basis is read; 0 keeps every structure and only sets the warning bit.
    double max_vacuum = -1.0;
};

// ---------------------------------------------------------------------------
// How much work, for how long, and whether this is a resume
// ---------------------------------------------------------------------------
struct RunControl {
    // Compound name for CIF block keys and the database record. Empty means
    // "the first basis file's stem", resolved once at run start.
    std::string name;
    std::vector<std::string> basis_paths;
    std::vector<int> space_groups;

    std::map<int, int> targets;         // per-SG, from a preset
    int target_per_sg = 10000;          // used where `targets` says nothing

    // 0 sizes it from the device and the molecule; a positive value pins it.
    int batch_size = 0;
    BatchSizing batch_sizing = BatchSizing::ATOMS;
    int batch_atoms = 0;                // 0 = pick by device

    int    result_queue_mb = 256;
    double max_runtime_s = 0;           // 0 = no wall-clock budget
    bool   sg_affinity = false;

    bool resume = false;
    // Resume even when the fingerprint disagrees. Energies and dedup verdicts
    // are then not comparable across the join.
    bool resume_force = false;

    // Stop after generation; stage 0 is then the stage counted and deduplicated.
    bool generate_only = false;
};

// ---------------------------------------------------------------------------
// The whole thing
// ---------------------------------------------------------------------------
struct RunConfig {
    GeneratorConfig  generator;
    OptimizerConfig  optimizer;
    ForceFieldConfig forcefield;
    PetConfig        pet;
    DedupConfig      dedup;
    OutputConfig     output;
    RunControl       control;

    // The target for one space group: its entry in `targets`, or the default.
    int target_for(int sg) const {
        const auto it = control.targets.find(sg);
        return it != control.targets.end() ? it->second : control.target_per_sg;
    }
};

// WITH_DEFAULT: a key absent from the JSON leaves the member initialiser.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneratorConfig,
    tvp, collision, nudge_budget, contract,
    rp_iterations, rp_pressure_gpa, rp_max_strain_step,
    fft_k_per_trial, fft_grid_spacing, fft_collision_tol, fft_max_aspect)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(OptimizerConfig,
    n_iterations, n_iterations_stage1, stage1_pressure_gpa, grad_tol,
    mp_single_switch_tol,
    fixed_cell, fix_angles_stage1, stream, stream_pool_mult, refill_interval,
    max_iters_per_slot, stream_stage1, contract_pairs,
    sampler, mc_temperature_k, mc_moves, mc_step_trans, mc_step_rot_deg,
    mc_step_strain, mc_min_contact)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ForceFieldConfig,
    max_order, ewald_accuracy, wolf, wolf_cutoff, mp_cutoff_ang, mp_single, mp_pair_moments)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(PetConfig,
    relax, symmetry, dedup, precondition, pass_rounds,
    model, model_json, model_weights, iterations, force_tol, stress_tol,
    min_contact, device, max_batch_atoms)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DedupConfig,
    energy_bin, density_bin, amd_tol, use_powder, powder_cos_thresh,
    powder_cdtw_thresh, powder_cdtw_band, cross_sg, pdd_tol, pdd_rows, pdd_k)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(OutputConfig,
    dir, store_all_stages, store_pdd, store_diagnostics, perf, perf_sync,
    max_vacuum)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RunControl,
    name, basis_paths, space_groups, targets, target_per_sg,
    batch_size, batch_sizing, batch_atoms, result_queue_mb, max_runtime_s,
    sg_affinity, resume, resume_force, generate_only)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RunConfig,
    generator, optimizer, forcefield, pet, dedup, output, control)

// Merge `patch` into `cfg` key by key (what --config does). Throws ConfigError,
// leaving `cfg` untouched, on an unknown section, key or enum name. A retired
// key is accepted only at the value it held; see retired_keys in config.cpp.
void apply_config_json(RunConfig& cfg, const nlohmann::json& patch);

// Read a config file into `cfg`. Throws ConfigError naming the path on a missing
// file, a parse error, an unknown key or a value of the wrong type.
void load_config_file(RunConfig& cfg, const std::string& path);

// Apply one `section.key=value` override, as --set gives it. Returns false and
// fills `error` when the path names nothing or the value will not convert.
bool apply_config_override(RunConfig& cfg, const std::string& assignment,
                           std::string& error);

// Every reason this configuration cannot run as written, one sentence each,
// naming the key and the flag that sets it. Empty means none. Does not check
// that inputs and space groups exist; prepare_run_config (csp_run.hpp) does.
std::vector<std::string> config_problems(const RunConfig& cfg);

// The whole effective configuration, for `{output.dir}/config.json`.
std::string effective_config_json(const RunConfig& cfg);

}  // namespace klasp
