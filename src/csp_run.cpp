#include "klasp/compack.hpp"
#include "klasp/csp_run.hpp"

#include "klasp/log.hpp"
#include "klasp/exit_code.hpp"
#include "klasp/constants.hpp"
#include "klasp/run_state.hpp"
#include "klasp/element.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/kernel_profiler.hpp"
#ifdef KLASP_HAS_PET
#include "klasp/pet_relax.hpp"
#include <array>
#endif

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <memory>
#include <sstream>
#include <thread>
#include <tuple>
#include <utility>

namespace klasp {

namespace {

// Set from a signal handler, so nothing may do more than store to it. The work
// loop polls it and makes the WorkSource stop request.
std::atomic<bool> g_signal_stop{false};

extern "C" void on_terminate_signal(int) {
    g_signal_stop.store(true, std::memory_order_relaxed);
}

// SIGTERM (scheduler grace period) and SIGINT both mean "wrap up". Installed
// only when CspRunOptions asks for it.
void install_stop_handlers() {
    std::signal(SIGTERM, on_terminate_signal);
    std::signal(SIGINT, on_terminate_signal);
}

}  // namespace

void request_run_stop() { g_signal_stop.store(true, std::memory_order_relaxed); }
bool run_stop_requested() { return g_signal_stop.load(std::memory_order_relaxed); }
void clear_run_stop() { g_signal_stop.store(false, std::memory_order_relaxed); }


// mol-cspy preset SG lists (ordered by frequency)
// Single-molecule: 25 most common organic SGs
static const std::vector<int> SG_SINGLE_25 = {
    61, 14, 19, 2, 4, 15, 33, 9, 29, 5,
    1, 60, 7, 18, 96, 76, 145, 43, 56, 13,
    169, 88, 20, 86, 148
};
// Co-crystal: 10 most common
static const std::vector<int> SG_COCRYSTAL = {2, 14, 15, 4, 19, 61, 1, 33, 5, 9};

struct SGPreset {
    std::vector<int> sgs;
    std::map<int, int> targets;
};

static const std::map<std::string, SGPreset> SG_PRESETS = {
    {"common5",  {{2, 14, 19, 33, 61}, {}}},
    {"common10", {{2, 4, 14, 15, 19, 29, 33, 61, 62, 148}, {}}},

    // Single-molecule presets (from mol-cspy)
    {"fine10",   {{61, 14, 19, 2, 4, 15, 33, 9, 29, 5},
                  {{61,10000},{14,10000},{19,10000},{2,10000},{4,10000},
                   {15,10000},{33,10000},{9,10000},{29,10000},{5,10000}}}},
    {"fine25",   {SG_SINGLE_25,
                  []{std::map<int,int> m; for(int s:SG_SINGLE_25) m[s]=10000; return m;}()}},
    {"coarse10", {{61, 14, 19, 2, 4, 15, 33, 9, 29, 5},
                  {{61,1000},{14,1000},{19,1000},{2,1000},{4,1000},
                   {15,1000},{33,1000},{9,1000},{29,1000},{5,1000}}}},
    {"coarse25", {SG_SINGLE_25,
                  []{std::map<int,int> m; for(int s:SG_SINGLE_25) m[s]=1000; return m;}()}},
    {"common25", {SG_SINGLE_25, {}}},

    // Z' presets
    {"efficientz1", {{14, 19, 2, 4, 61, 15, 33, 9, 29},
                     {{14,5000},{19,1000},{2,1000},{4,1000},{61,1000},
                      {15,1000},{33,1000},{9,1000},{29,1000}}}},
    {"efficientz2", {{2, 14, 4},
                     {{2,12500},{14,7500},{4,7500}}}},

    // Co-crystal presets
    {"cocrystal",       {SG_COCRYSTAL, {}}},
    {"cocrystal_fine",  {SG_COCRYSTAL,
                         {{2,10000},{14,50000},{15,50000},{4,20000},{19,10000},
                          {61,20000},{1,10000},{33,20000},{5,20000},{9,20000}}}},
    {"cocrystal_coarse",{{2, 14, 15, 4, 19, 61},
                         {{2,2500},{14,5000},{15,5000},{4,2500},{19,3000},{61,3000}}}},
};

std::vector<int> parse_sg_spec(const std::string& spec,
                               std::map<int, int>* targets) {
    auto it = SG_PRESETS.find(spec);
    if (it != SG_PRESETS.end()) {
        if (targets && !it->second.targets.empty())
            *targets = it->second.targets;
        return it->second.sgs;
    }
    // A comma-separated list, every entry of which must be a space group 1-230.
    const auto not_a_spec = [&] {
        std::string presets;
        for (const auto& [name, preset] : SG_PRESETS)
            presets += (presets.empty() ? "" : ", ") + name;
        return ConfigError("'" + spec + "' is neither a space group list (such as "
                           "2,14,19) nor a preset (" + presets + ")");
    };
    std::vector<int> result;
    std::istringstream entries(spec);
    std::string entry;
    while (std::getline(entries, entry, ',')) {
        std::size_t used = 0;
        int sg = 0;
        try {
            sg = std::stoi(entry, &used);
        } catch (const std::exception&) {
            throw not_a_spec();
        }
        if (used != entry.size()) throw not_a_spec();
        if (sg < 1 || sg > 230)
            throw ConfigError("space group " + entry + " in '" + spec +
                              "' is not between 1 and 230");
        result.push_back(sg);
    }
    if (result.empty()) throw not_a_spec();
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

RunConfig prepare_run_config(const RunConfig& config) {
    std::vector<std::string> problems = config_problems(config);
    if (config.control.basis_paths.empty())
        problems.push_back("control.basis_paths is empty: a run needs at least one "
                           "basis JSON, named on the command line or in the "
                           "configuration");
    if (config.control.space_groups.empty())
        problems.push_back("control.space_groups is empty: a run needs at least one "
                           "space group, from --sg or the configuration");
    if (!problems.empty()) throw ConfigError(problems);

    RunConfig prepared = config;
#ifdef KLASP_HAS_PET
    // Resolved here so a model named in a config file, from Python or on the
    // command line is found the same way, and only when the stage runs. The
    // resolved pair is what the fingerprint and config.json record.
    if (prepared.pet.relax && prepared.pet.model_json.empty())
        resolve_pet_model(prepared.pet.model, prepared.pet.model_json,
                          prepared.pet.model_weights);
#endif
    return prepared;
}

// The settings two landscapes must share for a merge to mean anything: what
// decides which structures exist (generator), their energy (basis, optimiser)
// and whether two count as the same (dedup thresholds). Excludes throughput
// settings, per-SG targets and the space-group list, which a resume may change.
// ewald_accuracy is passed in because only the loaded workspace knows whether
// --ewald-accuracy replaced the input's value.
static std::string canonical_run_config(const RunConfig& c,
                                        const BasisConfig& basis,
                                        int effective_max_rank,
                                        RealType ewald_accuracy) {
    nlohmann::json j;
    j["basis"] = basis.basis_json;
    j["generator"] = {
        {"tvp", c.generator.tvp},
        {"collision", static_cast<int>(c.generator.collision)},
        {"gen_nudge", c.generator.nudge_budget},
        {"gen_contract", c.generator.contract},
        {"rp_iterations", c.generator.rp_iterations},
        {"rp_pressure_gpa", c.generator.rp_pressure_gpa},
        {"rp_max_strain_step", c.generator.rp_max_strain_step},
    };
    j["optimizer"] = {
        {"max_rank", effective_max_rank},
        {"n_iterations", c.optimizer.n_iterations},
        {"n_iterations_stage1", c.optimizer.n_iterations_stage1},
        {"stage1_pressure_gpa", c.optimizer.stage1_pressure_gpa},
        {"grad_tol", c.optimizer.grad_tol},
        {"fixed_cell", c.optimizer.fixed_cell},
        {"fix_angles_stage1", c.optimizer.fix_angles_stage1},
        {"wolf", c.forcefield.wolf},
        {"wolf_cutoff", c.forcefield.wolf_cutoff},
        {"mp_cutoff_ang", c.forcefield.mp_cutoff_ang},
        {"mp_single", c.forcefield.mp_single},
        {"mp_pair_moments", c.forcefield.mp_pair_moments},
        {"ewald_accuracy", double(ewald_accuracy)},
        // Streaming and lockstep stop a structure on different rules, so they
        // do not land on the same geometry.
        {"stream", c.optimizer.stream},
        // Fixed values for the removed damped L-BFGS stage 2, so older
        // trust-region landscapes still resume.
        {"trust_region", true},
        {"max_step", 0.05},
        {"fd_strain", false},
        {"lbfgs_memory", 5},
    };
    // Stage-3 energies from different ML checkpoints cannot be ranked together.
    j["pet"] = {
        {"relax", c.pet.relax},
        {"symmetry", c.pet.symmetry},
        {"model_json", c.pet.model_json},
        {"model_safetensors", c.pet.model_weights},
        // Both change the stage-3 energy.
        {"iterations", c.pet.iterations},
        {"force_tol", c.pet.force_tol},
        {"stress_tol", c.pet.stress_tol},
        {"min_contact", c.pet.min_contact},
        // With the gate on only one structure per basin carries a stage-3 row,
        // so gated and ungated landscapes differ in coverage.
        {"dedup", c.pet.dedup},
        // Both change how far the minimisation gets.
        {"precondition", c.pet.precondition},
        {"pass_rounds", c.pet.pass_rounds},
    };
    j["dedup"] = {
        {"energy_bin", c.dedup.energy_bin},
        {"density_bin", c.dedup.density_bin},
        {"amd_tol", c.dedup.amd_tol},
        {"use_powder", c.dedup.use_powder},
        {"powder_cos_thresh", c.dedup.powder_cos_thresh},
        {"powder_cdtw_thresh", c.dedup.powder_cdtw_thresh},
        {"powder_cdtw_band", c.dedup.powder_cdtw_band},
        {"pdd_tol", c.dedup.pdd_tol},
        {"cross_sg", c.dedup.cross_sg},
        {"store_pdd", c.output.store_pdd},
    };
    // nlohmann's object is key-ordered, so dump() is already canonical.
    return j.dump();
}

// Everything a run needs before it can be told how much work there is: the
// basis, the batch geometry, the ML evaluator, and one worker per space group.
void csp_setup(CspRun& run) {
    clear_run_stop();
    if (run.options.install_signal_handlers) install_stop_handlers();
    run.rank = run.comm.rank();
    run.nranks = run.comm.size();
    run.n_sgs = static_cast<int>(run.config.control.space_groups.size());

    const auto& gen = run.config.generator;
    const auto& opt = run.config.optimizer;
    const auto& ff  = run.config.forcefield;
    const auto& pet = run.config.pet;
    const auto& out = run.config.output;
    const auto& ctl = run.config.control;
    const RunConfig& config = run.config;
    const int rank = run.rank;
    const int nranks = run.nranks;
    auto& ws = run.ws;
    auto& basis = run.basis;
    int& effective_max_rank = run.effective_max_rank;
    auto& bsettings = run.bsettings;
    const int n_sgs = run.n_sgs;
    auto& perf = run.perf;
    auto& per_sg_batch = run.per_sg_batch;
#ifdef KLASP_HAS_PET
    auto& pet_eval = run.pet_eval;
#endif
    double& pet_reference = run.pet_reference;
    auto& dcfg = run.dcfg;
    auto& workers = run.workers;
    ws = allocate_workspace(512, 27*27*27, 50000, 32, 4);
    basis = ctl.basis_paths.size() > 1
        ? load_basis_json_multi(ctl.basis_paths, ws)
        : load_basis_json(ctl.basis_paths.front(), ws);
    apply_ewald_accuracy_override(ff.ewald_accuracy, ws);

    // Negative max_vacuum derives the threshold from the cutoff; an explicit
    // value, 0 included, is honoured. basis.cutoff is Bohr, the gap Angstrom.
    run.max_vacuum = resolve_max_vacuum(out.max_vacuum,
                                        basis.cutoff * BOHR_TO_ANGSTROM);
    if (out.max_vacuum < 0.0)
        logging::detail("vacuum: dropping structures with a gap over {:.1f} A "
                        "({:.0f}% of the {:.1f} A potential cutoff)\n",
                        run.max_vacuum, 100.0 * VACUUM_GAP_CUTOFF_FRACTION,
                        basis.cutoff * BOHR_TO_ANGSTROM);
    effective_max_rank = basis.max_rank;
    if (ff.max_order >= 0)
        effective_max_rank = ff.max_order;

    // For co-crystals: body data is n_mols * n_sites (padded to max per type)
    int total_body = basis.n_mols * basis.n_sites;
    std::vector<double> cov(total_body), vdw(total_body);
    double mol_mass = 0;
    for (int m = 0; m < basis.n_mols; ++m) {
        int ns = (basis.n_mols > 1) ? basis.mol_n_sites[m] : basis.n_sites;
        for (int i = 0; i < ns; ++i) {
            int idx = m * basis.n_sites + i;
            auto& el = element(basis.elements[idx]);
            cov[idx] = std::max(0.68, static_cast<double>(el.cov_radius));
            vdw[idx] = el.vdw_radius;
            mol_mass += el.mass;
        }
        // Ghost padding atoms get zero radii (no collision contribution)
        for (int i = ns; i < basis.n_sites; ++i) {
            int idx = m * basis.n_sites + i;
            cov[idx] = 0.0;
            vdw[idx] = 0.0;
        }
    }
    std::vector<double> body_pos(total_body * 3);
    for (int i = 0; i < total_body; ++i)
        for (int d = 0; d < 3; ++d)
            body_pos[i*3+d] = basis.body_positions[i][d];

    bsettings.n_iterations = opt.n_iterations;
    bsettings.model.taper_on = basis.taper_on;
    bsettings.model.taper_off = basis.taper_off;
    bsettings.contract_pairs = opt.contract_pairs;
    bsettings.model.wolf = ff.wolf;
    bsettings.model.wolf_cutoff = ff.wolf_cutoff;
    bsettings.model.buck_damping = basis.buck_damping;
    bsettings.grad_tol = opt.grad_tol;
    bsettings.mp_single_switch_tol = static_cast<RealType>(opt.mp_single_switch_tol);
    bsettings.model.mp_cutoff = ff.mp_cutoff_ang > 0
        ? static_cast<RealType>(ff.mp_cutoff_ang * ANGSTROM_TO_BOHR) : RealType(0);
    bsettings.model.mp_single = ff.mp_single;
    bsettings.model.mp_pair_moments = ff.mp_pair_moments;

    perf.set_enabled(out.perf);
    // Per-kernel timings via Kokkos Tools callbacks; csp_shutdown writes them.
    if (out.perf) KernelProfiler::instance().enable(out.perf_sync);

    // Per-SG adaptive batch size: keep batch x n_atoms roughly constant so every
    // kernel launch has enough parallel work regardless of SG. --batch-size
    // overrides with a single value for all SGs. Not capped by the per-rank
    // share; the work counters cap each grab.
    per_sg_batch.assign(n_sgs, 0);
    if (ctl.batch_size > 0) {
        for (int i = 0; i < n_sgs; ++i)
            per_sg_batch[i] = std::max(1, ctl.batch_size);
    } else {
        const char* device_name = Kokkos::DefaultExecutionSpace::name();
        std::string dev = device_name ? device_name : "";
        bool is_gpu = (dev.find("HIP") != std::string::npos)
                   || (dev.find("Cuda") != std::string::npos)
                   || (dev.find("CUDA") != std::string::npos);
        int target_atoms = ctl.batch_atoms > 0 ? ctl.batch_atoms
                                                  : (is_gpu ? 32768 : 2048);
        int n_sites = basis.n_sites;
        // A co-crystal puts each component through the whole space group, so
        // the atom count scales with n_mols.
        const int n_comp = std::max(1, basis.n_mols);
        for (int i = 0; i < n_sgs; ++i) {
            int n_symops = get_space_group(ctl.space_groups[i]).n_symops;
            int n_atoms = std::max(1, n_sites * n_symops * n_comp);
            int b = std::max(16, target_atoms / n_atoms);
            b = std::min(b, 1024);
            if (ctl.batch_sizing == BatchSizing::LEGACY) {
                // Round down to a power of two. Nothing requires a power of two;
                // this policy is kept for comparison.
                int p = 1;
                while (p * 2 <= b) p *= 2;
                b = p;
            }
            per_sg_batch[i] = b;
        }
        if (rank == 0) {
            const bool legacy = ctl.batch_sizing == BatchSizing::LEGACY;
            logging::info("  Auto batch sizes ({} sites/mol{}, target {} atoms/kernel, "
                       "{} sizing):\n",
                       n_sites,
                       n_comp > 1 ? fmt::format(", {} components", n_comp) : "",
                       target_atoms, legacy ? "legacy" : "atoms");
            // How much of the requested launch each space group actually gets.
            double achieved = 0;
            for (int i = 0; i < n_sgs; ++i) {
                const int na = n_sites * get_space_group(ctl.space_groups[i]).n_symops
                             * n_comp;
                const int got = per_sg_batch[i] * na;
                achieved += double(std::min(got, target_atoms)) / double(target_atoms);
                logging::info("    SG {:>3d} ({} symops, {} atoms/struct): batch={:<5} "
                           "{} atoms/kernel ({:.0f}% of target)\n",
                           ctl.space_groups[i],
                           get_space_group(ctl.space_groups[i]).n_symops,
                           na, per_sg_batch[i], got,
                           100.0 * double(got) / double(target_atoms));
            }
            logging::info("    mean {:.0f}% of the atom target\n",
                       100.0 * achieved / double(n_sgs));
        }
    }

    // PET-MAD evaluator: one model load per rank, shared across this rank's
    // workers. Declared before `workers` so it outlives their borrowed pointer.
#ifdef KLASP_HAS_PET
    if (pet.relax) {
        pet_eval = pet.model_json.empty()
                       ? std::make_unique<PetEvaluator>()
                       : std::make_unique<PetEvaluator>(pet.model_json,
                                                        pet.model_weights);
        // Computed on rank 0 and broadcast so every stage-3 lattice energy uses
        // the same reference.
        if (rank == 0) pet_reference = pet_isolated_reference(*pet_eval, basis);
            run.comm.broadcast(&pet_reference, 1, 0);
        if (rank == 0)
            logging::info("PET-MAD final relaxation enabled (stage 3, symmetry={}); "
                       "isolated-molecule reference = {:.4f} kJ/mol/fu (stage-3 energies are "
                       "lattice energies).\n",
                       pet.symmetry ? "on" : "off", pet_reference);
    }
#endif

    // Dedup thresholds plus the PDD shape (fixed per run: n_asym x AMD_K). Built
    // on every rank so the workers' stage-3 gate matches rank 0's index.
    dcfg = config.dedup;
    if (out.store_pdd) {
        dcfg.pdd_rows = basis.n_asym_atoms();
        dcfg.pdd_k = AMD_K;
        dcfg.pdd_blocks = pdd_symmetry_blocks(basis);
    } else {
        dcfg.pdd_tol = 0.0;  // can't run the PDD stage without stored PDDs
    }

    for (int i = 0; i < n_sgs; ++i) {
        workers[i].configure(config, basis.n_asym_atoms(), run.max_vacuum);
        // Runtime wiring that configure() leaves alone: pointers and callbacks.
        workers[i].set_abort_flag(&g_signal_stop);
        workers[i].set_perf(&perf);
        workers[i].set_primary_stage(ctl.generate_only ? STAGE_GENERATED
                                     : (effective_max_rank > 0 ? STAGE_MULTIPOLE
                                                               : STAGE_CHARGES));
        workers[i].init(ctl.space_groups[i], config.target_for(ctl.space_groups[i]),
                        per_sg_batch[i], basis, ws, bsettings,
                        body_pos, cov, vdw, mol_mass, gen.tvp,
                        effective_max_rank, rank, nranks);
#ifdef KLASP_HAS_PET
        if (pet.relax) {
            workers[i].set_pet_relax(pet_eval.get(), pet.symmetry, pet_reference);
            workers[i].set_pet_dedup(dcfg, pet.dedup);
        }
#endif
    }

    // Every producer hands results to a ResultSink feeding one ResultConsumer
    // (dedup, database, status table) on rank 0.
    if (rank == 0) std::filesystem::create_directories(out.dir);
    run.comm.barrier();
}

// How much work there is, where each space group's generator sequence picks up,
// and whether this run may touch the output directory. The only phase that can
// refuse, and it refuses on every participant at once.
void csp_plan_work(CspRun& run) {
    const auto& opt = run.config.optimizer;
    const auto& out = run.config.output;
    const auto& ctl = run.config.control;
    const RunConfig& config = run.config;
    const int rank = run.rank;
    auto& ws = run.ws;
    auto& basis = run.basis;
    int& effective_max_rank = run.effective_max_rank;
    const int n_sgs = run.n_sgs;
    auto& per_sg_batch = run.per_sg_batch;
    auto& targets = run.targets;
    auto& seed_start = run.seed_start;
    auto& unit_sizes = run.unit_sizes;
    auto& cfg_json = run.cfg_json;
    targets.assign(n_sgs, 0);
    unit_sizes.assign(n_sgs, 0);
    // Per-SG targets and unit sizes (the largest chunk handed out for an SG: the
    // streaming pool size when streaming, else one batch). Built identically on
    // every rank.
    for (int i = 0; i < n_sgs; ++i) {
        targets[i] = config.target_for(ctl.space_groups[i]);
        unit_sizes[i] = opt.stream
            ? std::max(1, per_sg_batch[i] * std::max(1, opt.stream_pool_mult))
            : per_sg_batch[i];
    }

    // The stage that produces exactly one row per structure. A resume counts
    // structures at this stage.
    run.primary_stage =
        ctl.generate_only ? 0 : (effective_max_rank > 0 ? 2 : 1);
    const int primary_stage = run.primary_stage;

    // --- resume ---
    // Must run before the work counters are created, since it sets remaining
    // work and seed cursors. Rank 0 probes the database and broadcasts.
    seed_start.assign(n_sgs, 0);
    // Broadcast rather than abort so every rank returns cleanly and rank 0's
    // message is the last thing in the log.
    int refuse = 0;
    RunState prior;
    cfg_json = canonical_run_config(config, basis, effective_max_rank, ws.ewald_accuracy);
    if (rank == 0) {
        prior = probe_run(out.dir + "/csp.db", ctl.space_groups,
                          primary_stage);
        if (prior.schema_version > KLASP_DB_SCHEMA_VERSION) {
            logging::error("\n{}/csp.db was written by a newer klasp (schema {} > {}).\n"
                "Refusing to write into it -- upgrade, or point -o somewhere else.\n",
                out.dir, prior.schema_version, KLASP_DB_SCHEMA_VERSION);
            refuse = +ExitCode::RefusedNewerSchema;
        }
        // Starting fresh on an existing landscape would overwrite every row, since
        // the generator replays the same (seed, space_group, stage) keys.
        if (!ctl.resume && prior.exists && prior.total_rows > 0) {
            logging::error("\n{}/csp.db already holds {} structures and this is not a --resume.\n"
                "Seeds regenerate identically, so continuing would overwrite every one of\n"
                "them. Pass --resume to carry the run on, or point -o somewhere else.\n",
                out.dir, prior.total_rows);
            refuse = +ExitCode::RefusedWouldOverwrite;
        }
        if (ctl.resume) {
            if (prior.schema_version < 0) {
                // A database that exists but cannot be read.
                logging::error("\n--resume refused: {}/csp.db exists but could not be read.\n"
                    "Continuing would restart the generator at seed 0 and overwrite what is\n"
                    "already stored. Move it aside, or drop --resume to start deliberately.\n",
                    out.dir);
                refuse = +ExitCode::RefusedResumeUnreadable;
            }
            if (!prior.exists) {
                logging::info("\n--resume: nothing to resume from in {} -- starting fresh.\n",
                           out.dir);
            } else if (prior.config_hash.empty() && !ctl.resume_force) {
                // No fingerprint to check the current settings against.
                logging::error("\n--resume refused: {}/csp.db records no settings fingerprint, so\n"
                    "there is no way to tell whether it was computed the way this run is\n"
                    "configured. Continuing could merge structures relaxed under different\n"
                    "settings, whose energies are not comparable. Pass --resume-force if you\n"
                    "know the settings match.\n",
                    out.dir);
                refuse = +ExitCode::RefusedResumeMismatch;
            } else if (!prior.config_hash.empty()
                       && prior.config_hash != config_fingerprint(cfg_json)
                       && !ctl.resume_force) {
                // Energies and dedup verdicts would not be comparable across the join.
                logging::error("\n--resume refused: {}/csp.db was computed with different settings.\n",
                    out.dir);
                for (const auto& d : config_differences(prior.config_json, cfg_json))
                    logging::error("  {}\n", d);
                logging::error("\nEnergies would not be comparable across the join, and the stored\n"
                    "duplicate verdicts were taken against the old thresholds. Re-run with\n"
                    "the original settings, start a new output directory, or pass\n"
                    "--resume-force if you have decided the difference does not matter.\n");
                refuse = +ExitCode::RefusedResumeMismatch;
            } else {
                if (prior.config_hash.empty())
                    logging::error("\nWARNING: --resume-force -- continuing into a landscape with no\n"
                        "settings fingerprint. Nothing checked that it was computed the way\n"
                        "this run is configured.\n");
                else if (prior.config_hash != config_fingerprint(cfg_json))
                    logging::error("\nWARNING: --resume-force -- continuing into a landscape computed\n"
                        "with different settings. Energies across the join are not comparable.\n");
                for (int i = 0; i < n_sgs; ++i) {
                    const auto* p = prior.find(ctl.space_groups[i]);
                    if (!p) continue;
                    targets[i] = std::max(0LL, targets[i] - p->rows);
                    // Continue the sequence so existing rows are not overwritten.
                    seed_start[i] = p->max_seed + 1;
                }
            }
        }
    }
    run.comm.broadcast(&refuse, 1, 0);
    if (refuse != 0) {
        // Thrown on every rank, so no collective is left waiting.
        throw RunRefused{refuse};
    }
    run.comm.broadcast(targets.data(), n_sgs, 0);
    run.comm.broadcast(seed_start.data(), n_sgs, 0);
}

// Wire the workers to where their results go and where their generator seeds
// come from. Runs after the work source exists.
void csp_attach(CspRun& run) {
    const auto& ctl = run.config.control;
    const int rank = run.rank;
    const int nranks = run.nranks;
    auto& basis = run.basis;
    const int n_sgs = run.n_sgs;
    auto& workers = run.workers;
    auto& targets = run.targets;
    WorkSource& work = *run.work;
    ResultSink& sink = run.sink;
    // Stream results out as each chunk completes rather than per work unit.
    for (auto& w : workers)
        w.set_emit([&sink](const std::vector<StructureResult>& rows,
                             const std::vector<std::vector<double>>& pdds,
                             const std::unordered_map<int, FlexibleStructure>& flex) {
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const std::vector<double>* p =
                    (i < pdds.size() && !pdds[i].empty()) ? &pdds[i] : nullptr;
                const FlexibleStructure* f = nullptr;
                if (rows[i].stage == STAGE_FINAL) {  // only PET rows carry explicit geometry
                    auto it = flex.find(rows[i].seed);
                    if (it != flex.end()) f = &it->second;
                }
                sink.send(rows[i], p ? p->data() : nullptr,
                            p ? static_cast<int>(p->size()) : 0, f);
            }
        });
    // Space-group affinity, computed identically on every rank. Weighted by cost
    // per structure (symops x sites x components).
    if (ctl.sg_affinity && nranks > 1 && n_sgs > 1) {
        std::vector<long long> cost(n_sgs);
        for (int i = 0; i < n_sgs; ++i)
            cost[i] = static_cast<long long>(get_space_group(ctl.space_groups[i]).n_symops)
                    * std::max(1, basis.n_sites) * std::max(1, basis.n_mols);
        const auto home = assign_home_sets(targets, cost, nranks);
        work.set_home(home[rank]);
        if (rank == 0) {
            int total = 0;
            for (const auto& h : home)
                for (unsigned char v : h) total += v ? 1 : 0;
            logging::info("  Space-group affinity: {:.1f} space groups per rank on average "
                       "(off by default; --sg-affinity enabled it)\n",
                       double(total) / double(nranks));
        }
    }

    // Seeds come from the shared per-SG cursor, so the sampled structures do not
    // depend on rank count or device.
    for (int i = 0; i < n_sgs; ++i)
        workers[i].set_seed_claimer([&work, i](int n) { return work.claim_seeds(i, n); });
}

// Rank 0 opens the database, seeds the dedup index from what is already there,
// and starts storing. Other ranks do nothing here.
void csp_start_consumer(CspRun& run) {
    const auto& opt = run.config.optimizer;
    const auto& pet = run.config.pet;
    const auto& out = run.config.output;
    const auto& ctl = run.config.control;
    const RunConfig& config = run.config;
    const int rank = run.rank;
    const int nranks = run.nranks;
    auto& basis = run.basis;
    const int n_sgs = run.n_sgs;
    auto& per_sg_batch = run.per_sg_batch;
    auto& dcfg = run.dcfg;
    auto& targets = run.targets;
    const int primary_stage = run.primary_stage;
    auto& cfg_json = run.cfg_json;
    auto& consumer = run.consumer;
    run.queue.set_max_queue_bytes(ctl.result_queue_mb > 0
        ? static_cast<std::size_t>(ctl.result_queue_mb) << 20
        : 0);
    if (rank == 0) {
        // The target column counts the whole landscape, existing rows included.
        std::vector<int> targets_i(n_sgs);
        for (int i = 0; i < n_sgs; ++i)
            targets_i[i] = static_cast<int>(config.target_for(ctl.space_groups[i]));
        consumer = std::make_unique<ResultConsumer>(
            out.dir, nranks, ctl.space_groups, targets_i, run.queue,
            primary_stage, ctl.resume);
        consumer->set_basis(&basis);
        consumer->show_stage3(pet.relax);
        consumer->report_stage3_gap(pet.relax && pet.dedup);
        // Largest work unit any rank runs before reporting: stream_pool_mult *
        // batch_size, per space group.
        {
            int biggest = 0;
            for (int i = 0; i < n_sgs; ++i) biggest = std::max(biggest, per_sg_batch[i]);
            consumer->set_work_unit(std::max(1, opt.stream_pool_mult) * biggest);
        }
        consumer->set_pet_stage(pet.relax);
        consumer->set_meta("basis_json", basis.basis_json.dump());
        consumer->set_meta("name", ctl.name.empty()
                                       ? std::filesystem::path(ctl.basis_paths.front()).stem().string()
                                       : ctl.name);
        // The fingerprint and the document it was taken over, so a --resume can
        // say which setting changed.
        consumer->set_meta("schema_version", std::to_string(KLASP_DB_SCHEMA_VERSION));
        consumer->set_meta("config_hash", config_fingerprint(cfg_json));
        consumer->set_meta("config_json", cfg_json);

        // The whole configuration, beside the database. `klasp csp --config
        // <output>/config.json` reruns what ran.
        {
            const auto path = std::filesystem::path(out.dir) / "config.json";
            std::ofstream f(path);
            if (f) {
                f << effective_config_json(config) << "\n";
            } else {
                logging::warn("warning: could not write {}\n", path.string());
            }
        }
        consumer->configure_dedup(dcfg);
        // After configure_dedup (the index bins by energy and density) and
        // before start (the worker thread must not race the seeding).
        consumer->resume_dedup_index();
        consumer->start();

        const char* device = Kokkos::DefaultExecutionSpace::name();
        long long total_target = 0;
        for (int i = 0; i < n_sgs; ++i) total_target += targets[i];
        logging::info("\n=== CSP: {} SG{}, {} total target, {} x {} ===\n",
                   n_sgs, n_sgs > 1 ? "s" : "", total_target, nranks, device);
        for (int i = 0; i < n_sgs; ++i)
            logging::info("  SG {:>3d}: {} symops, target {}\n",
                       ctl.space_groups[i],
                       get_space_group(ctl.space_groups[i]).n_symops,
                       config.target_for(ctl.space_groups[i]));
    }
}

namespace {

// Flush the per-SG counter deltas, one message for all kRejFields. Shared by
// the periodic and shutdown flushes.
void flush_counters(CspRun& run, bool blocking) {
    const int n = kRejFields * run.n_sgs;
    std::vector<int> delta(n);
    bool any = false;
    for (int i = 0; i < n; ++i) {
        delta[i] = run.local_rej[i] - run.last_sent_rej[i];
        if (delta[i] > 0) any = true;
    }
    if (!any) return;
    run.last_sent_rej = run.local_rej;
    run.sink.send_rejections(delta, blocking);
}

}  // namespace

void csp_loop(CspRun& run) {
    const auto& ctl = run.config.control;
    const int rank = run.rank;
    const int nranks = run.nranks;
    const int n_sgs = run.n_sgs;
    auto& workers = run.workers;
    auto& consumer = run.consumer;
    auto& local_rej = run.local_rej;
    auto& last_sent_rej = run.last_sent_rej;
    double& unit_seconds_max = run.unit_seconds_max;
    WorkSource& work = *run.work;
    ResultSink& sink = run.sink;
    // Main loop. Every rank claims a (space group, chunk) by atomic RMA on rank
    // 0's WorkWindow, runs one unit, sends results to rank 0, and reports the
    // produced count so any shortfall reopens. WAIT means all remaining work is in
    // flight; DONE means every target is met.

    using Clock = std::chrono::steady_clock;
    // Startup counts against the budget; falls back to now if process_start is unset.
    const auto run_start = run.options.process_start == Clock::time_point{}
                               ? Clock::now()
                               : run.options.process_start;
    auto last_batch_finish = Clock::now();
    auto last_idle_warn = Clock::now();
    // Latched so a rank requests a stop once.
    bool stop_asked = false;
    ZeroYieldGuard zero_yield;
    Clock::time_point wait_since{};
    long long wait_owed = -1;
    int wait_backoff_ms = 1;

    // Patience before deciding another rank is not coming back: three work units,
    // floored at 5 minutes and capped at 30.
    auto stall_limit_s = [&] {
        return std::clamp(3.0 * unit_seconds_max, 300.0, 1800.0);
    };
    // Three per-space-group counters share one message (kRejFields): rejected
    // seeds, PET candidates with no stage-3 row, and PET candidates the dedup gate
    // skipped. The last two let rank 0 decide a space group's ML stage is finished.
    local_rej.assign(kRejFields * n_sgs, 0);
    last_sent_rej.assign(kRejFields * n_sgs, 0);
    bool done = false;
    int iter = 0;

    // Idle-detection logger: if this rank goes >5 s without completing a batch,
    // log a warning explaining why (in run_batch or sending).
    const char* last_stall_reason = "unknown";

    while (!done) {
        // Watchdog. A busy unit (an ML stage can take many minutes) is reported
        // rarely; anything else is a real stall and warns loudly.
        {
            auto now = Clock::now();
            double stall_ms = std::chrono::duration<double, std::milli>(
                now - last_batch_finish).count();
            double since_warn = std::chrono::duration<double, std::milli>(
                now - last_idle_warn).count();
            const bool working = (std::string_view(last_stall_reason) == "in run_batch");
            const double quiet_ms = working ? 60000.0 : 5000.0;
            if (stall_ms > quiet_ms && since_warn > quiet_ms) {
                if (working)
                    logging::warn("[rank {}] working: {:.1f} min in the current work unit "
                        "(results are reported per unit, so the status table stays "
                        "flat until it finishes) iter={}\n",
                        rank, stall_ms / 60000.0, iter);
                else
                    logging::warn("[rank {}] IDLE WARN: {:.1f} s since last batch, "
                        "reason='{}' iter={}\n",
                        rank, stall_ms / 1000.0, last_stall_reason, iter);
                last_idle_warn = now;
            }
        }

        // 1. Is it time to wrap up? Checked before claiming. Both conditions set the
        //    shared flag, which grab() turns into DONE on every rank.
        if (!stop_asked) {
            const double elapsed =
                std::chrono::duration<double>(Clock::now() - run_start).count();
            // Reserve room for one more unit of the size this rank has run. With no
            // estimate yet, the first unit always runs.
            const double needed = elapsed + unit_seconds_max;
            const bool out_of_time =
                ctl.max_runtime_s > 0 && needed >= ctl.max_runtime_s;
            const bool signalled = run_stop_requested();
            if (out_of_time || signalled) {
                if (out_of_time)
                    logging::warn("[rank {}] runtime budget: {:.1f} of {:.1f} min used and a work "
                        "unit takes up to {:.1f} min, so this rank is stopping rather "
                        "than starting one it cannot finish. Re-run with --resume.\n",
                        rank, elapsed / 60.0, ctl.max_runtime_s / 60.0,
                        unit_seconds_max / 60.0);
                else
                    logging::warn("[rank {}] termination signal after {:.1f} min -- finishing the "
                        "current work unit, then writing out. Re-run with --resume to "
                        "continue.\n", rank, elapsed / 60.0);
                work.request_stop();
                stop_asked = true;
            }
        }

        // 2. Claim the next (sg, chunk) by atomic RMA on the shared counters.
        Grab g = work.grab();

        // Refresh the status table's stop count once per work unit.
        if (rank == 0 && consumer)
            consumer->set_winding_down(work.ranks_stopped(), nranks);
        if (g.kind == GrabKind::DONE) { done = true; break; }
        if (g.kind == GrabKind::WAIT) {
            // Work is briefly all in flight near the end: back off and retry. Each
            // retry costs remote atomics against rank 0's window, hence the backoff.
            last_stall_reason = "awaiting work (in flight)";
            // Restart the clock whenever the total owed moves; only a total that
            // does not move means nobody is progressing.
            const long long owed = work.last_remaining_total();
            if (owed != wait_owed) { wait_owed = owed; wait_since = Clock::now(); }
            if (wait_since == Clock::time_point{}) wait_since = Clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_backoff_ms));
            wait_backoff_ms = std::min(250, wait_backoff_ms * 2);

            // Nothing claimable and nothing arriving: the rank holding the inflight
            // work is gone or wedged.
            const double waited =
                std::chrono::duration<double>(Clock::now() - wait_since).count();
            if (waited > stall_limit_s() && !stop_asked) {
                logging::warn("\n[rank {}] {:.0f} min with work claimed and nothing produced "
                    "anywhere -- a rank holding a claim is gone or wedged.\nStopping so the "
                    "landscape so far is written out; --resume picks up what was left.\n",
                    rank, waited / 60.0);
                work.request_stop();
                stop_asked = true;
            }
            continue;
        }
        wait_since = Clock::time_point{};
        wait_owed = -1;
        wait_backoff_ms = 1;
        const int selected = g.sg_index;
        const int chunk = g.chunk;

        // 3. Run one work unit for `selected`, capped to `chunk` (the chunk caps
        //    the lockstep batch and the streaming pool via set_remaining).
        auto& w = workers[selected];
        w.set_remaining(chunk);
        last_stall_reason = "in run_batch";
        const auto unit_start = Clock::now();
        auto results = w.run_batch(chunk);
        last_batch_finish = Clock::now();
        unit_seconds_max = std::max(
            unit_seconds_max,
            std::chrono::duration<double>(last_batch_finish - unit_start).count());
        // Structures, not rows: stage 0 and the PET stage add rows for a
        // structure already counted.
        const int produced = w.last_batch_structures();
        local_rej[selected] += w.last_rejected();
        local_rej[3 * n_sgs + selected] += static_cast<int>(w.take_vacuum_dropped());
        local_rej[n_sgs + selected] += w.take_pet_dropped();
        local_rej[2 * n_sgs + selected] += w.take_pet_skipped();

        // 4. Deliver results, then report the produced count into the counters
        //    so any shortfall (produced < chunk) reopens for another rank.
        last_stall_reason = "in sink.send (waiting on the result path)";
        {
            // Empty when the worker streamed everything out through set_emit;
            // this is the fallback for drivers that do not.
            const auto& pdds = w.last_batch_pdds();
            const auto& flex = w.last_batch_flex();
            for (size_t i = 0; i < results.size(); ++i) {
                const std::vector<double>* p =
                    (i < pdds.size() && !pdds[i].empty()) ? &pdds[i] : nullptr;
                // Only PET stage-3 rows have explicit non-rigid geometry.
                const FlexibleStructure* f = nullptr;
                if (results[i].stage == STAGE_FINAL) {
                    auto it = flex.find(results[i].seed);
                    if (it != flex.end()) f = &it->second;
                }
                sink.send(results[i], p ? p->data() : nullptr,
                            p ? static_cast<int>(p->size()) : 0, f);
            }
        }
        work.report(selected, chunk, produced);
        last_stall_reason = "between batches";

        // A space group that keeps handing its whole chunk back stays the most-
        // behind group forever.
        zero_yield.record(selected, produced);
        if (zero_yield.tripped()) {
            // Retire the group so the others can finish and termination is reachable.
            logging::warn("\n[rank {}] SG {} has returned {} consecutive work units without "
                "producing a single structure.\nIts generator cannot place this molecule "
                "under the current constraints, and because unclaimed\nwork is handed "
                "straight back it would be chosen ahead of every other space group for "
                "the\nrest of the run. Retiring it; the run finishes short by its target. "
                "Drop SG {} from --sg,\nor loosen the cell constraints.\n",
                rank, ctl.space_groups[selected], zero_yield.streak(),
                ctl.space_groups[selected]);
            work.retire(selected);
            zero_yield.reset();
        }

        // 5. Flush the per-SG counter deltas every 4 iters, non-blocking.
        if ((iter % 4) == 3) flush_counters(run, /*blocking=*/false);
        ++iter;
    }
}

// The order here must not change: count the sign-offs, stop the drain, mop up
// what the stop raced, stop the consumer, then rendezvous.
void csp_shutdown(CspRun& run) {
    double& unit_seconds_max = run.unit_seconds_max;
    const auto& out = run.config.output;
    const int rank = run.rank;
    const int nranks = run.nranks;
    auto& perf = run.perf;
    auto& consumer = run.consumer;
    WorkSource& work = *run.work;
    ResultDrain& drain = *run.drain;
    ResultSink& sink = run.sink;
    using Clock = std::chrono::steady_clock;
    // The same bound the work loop waits on.
    auto stall_limit_s = [&] {
        return std::clamp(3.0 * unit_seconds_max, 300.0, 1800.0);
    };
    // Flush anything accumulated since the last periodic send.
    flush_counters(run, /*blocking=*/true);
    // A deferred send from the last periodic flush may still be outstanding
    // when the final delta was empty, and its buffer dies with the sink.
    sink.flush_rejections();
    // Workers reach here only after grab() returned DONE, so no claims are
    // outstanding.
    sink.flush();                        // every deferred send has released its buffer
    sink.send_sentinel();                // ...and this is ordered behind them

    // Rank 0 waits for n-1 sentinels, not a barrier: MPI keeps each sentinel
    // behind that rank's earlier results on the same tag, so the count proves
    // everything sent has landed, and the wait is bounded.
    if (rank == 0) {
        const int expected = nranks - 1;
        const double budget = stall_limit_s();
        const auto wait_start = Clock::now();
        while (drain.producers_signed_off() < expected) {
            // Keep the wind-down line moving until the slowest rank signs off. The
            // +1 is rank 0 itself. Sentinels are counted locally by the drain thread.
            consumer->set_winding_down(
                std::max(work.ranks_stopped(), drain.producers_signed_off() + 1), nranks);
            if (std::chrono::duration<double>(Clock::now() - wait_start).count() > budget) {
                logging::warn("\nWARNING: {} of {} worker ranks never signed off within {:.0f} min.\n"
                    "Anything still in flight from them is not in the database; everything\n"
                    "already received is. Finalising with what arrived.\n",
                    expected - drain.producers_signed_off(), expected, budget / 60.0);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        drain.stop();
        drain.mop_up();               // TAG_REJ rides a different tag: sweep it
        consumer->stop();
    }

    if (perf.enabled()) {
        perf.print_summary();
        std::string perf_path = fmt::format("{}/perf_rank{}.csv", out.dir, rank);
        perf.write(perf_path);
    }
    if (rank == 0) consumer->finalize();

    // The exit rendezvous, with the stall detector's patience. See MpiComm for
    // who may act on the deadline.
    run.comm.exit_rendezvous(stall_limit_s());
    // Fence so the GPU is idle before local Views destruct; a View freed with
    // pending HIP stream work can block hipFree.
    Kokkos::fence();

    // The per-kernel summary, written before Kokkos::finalize tears down the
    // Tools callbacks. Every rank collected; rank 0's view is the one written.
    if (out.perf) {
        KernelProfiler::instance().disable();
        if (rank == 0) {
            const std::string path = fmt::format("{}/perf_kernels.csv", out.dir);
            KernelProfiler::instance().write_summary(path);
            logging::info("Wrote per-kernel profile to {}\n", path);
        }
    }
}


const char* RunRefused::what() const noexcept {
    switch (static_cast<ExitCode>(code)) {
        case ExitCode::RefusedNewerSchema:
            return "klasp refused to start: the output directory holds a database "
                   "written by a newer klasp";
        case ExitCode::RefusedResumeMismatch:
            return "klasp refused to resume: the stored settings disagree with this run's";
        case ExitCode::RefusedResumeUnreadable:
            return "klasp refused to resume: the stored landscape could not be read";
        case ExitCode::RefusedWouldOverwrite:
            return "klasp refused to start: a fresh run here would overwrite an "
                   "existing landscape";
        default:
            return "klasp refused to start";
    }
}

// A whole CSP in one process, with no MPI linked, loaded or initialised: a
// WorkLedger for the counters, a QueueSink for results, and a SerialComm.
CspRunResult run_local_csp(const RunConfig& requested, CspRunOptions options) {
    const RunConfig config = prepare_run_config(requested);
    SerialComm comm;
    ResultQueue queue;
    CspRun* runp = nullptr;
    QueueSink sink(queue, [&runp](const std::vector<int>& delta) {
        record_rejection_delta(*runp->consumer,
                               runp->config.control.space_groups, delta);
    });
    CspRun run(config, comm, queue, sink);
    runp = &run;
    run.options = options;

    csp_setup(run);
    csp_plan_work(run);

    LedgerWorkSource work(WorkLedger(run.targets, run.unit_sizes, /*nranks=*/1,
                                     run.per_sg_batch));
    work.ledger().set_seed_start(run.seed_start);
    run.work = &work;

    csp_attach(run);
    csp_start_consumer(run);

    // In one process nothing is in flight, so the base ResultDrain is correct.
    ResultDrain nothing_in_flight;
    run.drain = &nothing_in_flight;

    csp_loop(run);

    csp_shutdown(run);

    // After shutdown, when the consumer has stored everything.
    CspRunResult result;
    for (int sg : config.control.space_groups) {
        result.stored += run.consumer->total_count(sg);
        result.unique += run.consumer->unique_count(sg);
    }
    result.stopped_early = run_stop_requested() || work.ranks_stopped() > 0;
    return result;
}

}  // namespace klasp
