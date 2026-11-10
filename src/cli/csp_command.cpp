// csp: the command line as one layer of a run's configuration. Each flag binds
// to a RunConfig holding only what was typed and is applied only if typed, so a
// flag default never overrides --config or --set.

#include "commands.hpp"

#include "cli_options.hpp"
#include "klasp/constants.hpp"
#include "klasp/csp_run.hpp"

#include <nlohmann/json.hpp>

#include <type_traits>

namespace klasp::cli {

namespace {

// The option groups --help lists, one per block of the configuration.
const char* const CONFIGURATION = "Configuration";
const char* const RUN = "Run (control.*)";
const char* const GENERATOR = "Generator (generator.*)";
const char* const OPTIMIZER = "Optimizer (optimizer.*)";
const char* const FORCEFIELD = "Force field (forcefield.*)";
#ifdef KLASP_HAS_PET
const char* const ML_STAGE = "ML stage (pet.*)";
#endif
const char* const DEDUP = "Deduplication (dedup.*)";
const char* const OUTPUT = "Output (output.*)";

std::string with_key(const std::string& help, const std::string& key) {
    return help + "  (" + key + ")";
}

}  // namespace

template <class Field>
void CspCommand::when_typed(CLI::Option* option, Field field) {
    apply_.push_back([this, option, field](RunConfig& cfg) {
        if (option->count() > 0) field(cfg) = field(typed_);
    });
}

template <class Field>
CLI::Option* CspCommand::keyed(CLI::Option* option, const std::string& key, Field field,
                               const char* group) {
    option->description(with_key(option->get_description(), key));
    option->group(group);
    when_typed(option, field);
    settings_.push_back({option, key});
    return option;
}

template <class Field>
CLI::Option* CspCommand::setting(const std::string& names, const std::string& key,
                                 Field field, const std::string& help, const char* group) {
    return keyed(app_->add_option(names, field(typed_), help)->capture_default_str(),
                 key, field, group);
}

template <class Field>
CLI::Option* CspCommand::flag(const std::string& names, const std::string& key, Field field,
                              const std::string& help, const char* group) {
    return keyed(app_->add_flag(names, field(typed_), help), key, field, group);
}

template <class Field>
CLI::Option* CspCommand::toggle(const std::string& name, const std::string& key, Field field,
                                const std::string& help, const char* group) {
    return keyed(add_switch(app_, name, field(typed_), help), key, field, group);
}

// An enum, typed by the name the configuration writes, with that name as its
// --help default.
template <class Field>
CLI::Option* CspCommand::choice(const std::string& name, const std::string& key, Field field,
                                const std::vector<std::string>& names,
                                const std::string& help, const char* group) {
    using Value = std::decay_t<decltype(field(typed_))>;
    CLI::Option* option = app_->add_option_function<std::string>(
                                  name,
                                  [this, field](const std::string& text) {
                                      field(typed_) = nlohmann::json(text).get<Value>();
                                  },
                                  help)
                              ->check(CLI::IsMember(names))
                              ->default_str(nlohmann::json(field(typed_)).get<std::string>());
    return keyed(option, key, field, group);
}

CspCommand::CspCommand(CLI::App& parent)
    : app_(parent.add_subcommand("csp", "Crystal structure prediction")) {
    // Optional: a config file can name inputs, and prepare_run_config refuses a
    // run with none.
    app_->add_option("input", inputs_,
        with_key("Basis JSON(s): one per molecule type for co-crystals. Replaces the "
                 "list a --config file gives.", "control.basis_paths"));

    // --- Configuration -------------------------------------------------------
    // The file sets a baseline and flags win over it. --set reaches any key and
    // is checked against the schema.
    app_->add_option("--config", config_path_,
        "Read the run configuration from this JSON file. Command-line flags "
        "override what it says; keys it omits keep their defaults.")
        ->check(CLI::ExistingFile)
        ->group(CONFIGURATION);
    app_->add_option("--set", sets_,
        "Override one configuration value, as section.key=value (for example "
        "optimizer.grad_tol=1e-6). Repeatable. Applied after --config and before "
        "the flags.")
        ->group(CONFIGURATION);
    app_->add_flag("--dump-config", dump_config_,
        "Print the effective configuration as JSON and exit, without running. "
        "Every run also writes this to {output}/config.json.")
        ->group(CONFIGURATION);

    // --- What to run, how much, and for how long ------------------------------
    auto* sg = app_->add_option("--sg", sg_spec_,
        with_key("Space group(s): a number, a comma list, or a preset such as "
                 "common10, fine25 or cocrystal_fine", "control.space_groups"))
        ->group(RUN);
    apply_.push_back([this, sg](RunConfig& cfg) {
        if (sg->count() == 0) return;
        try {
            cfg.control.space_groups = parse_sg_spec(sg_spec_, &cfg.control.targets);
        } catch (const ConfigError& e) {
            throw ConfigError(std::string("--sg: ") + e.what());
        }
    });
    // After --sg, whose preset may have filled the per-group targets this clears.
    auto* target = app_->add_option("--target", typed_.control.target_per_sg,
        with_key("Target valid structures per space group. Replaces the per-group "
                 "targets a preset carries.", "control.target_per_sg"))
        ->capture_default_str()
        ->group(RUN);
    apply_.push_back([this, target](RunConfig& cfg) {
        if (target->count() == 0) return;
        cfg.control.target_per_sg = typed_.control.target_per_sg;
        cfg.control.targets.clear();
    });
    setting("--name", "control.name", [](RunConfig& c) -> auto& { return c.control.name; },
        "Compound name recorded in the database and used to key extracted CIF blocks "
        "as {name}-QR-{sg}-{seed}-{stage}. Defaults to the input file's stem.", RUN);
    flag("--resume", "control.resume", [](RunConfig& c) -> auto& { return c.control.resume; },
        "Continue the run already in the output directory instead of starting a new one: "
        "top up each space group's target from what is stored, carry the generator on past "
        "the highest seed already used (rather than regenerating and overwriting those "
        "structures), and load the stored representatives into the dedup index so new "
        "structures are compared against the whole landscape. Refuses if the recorded "
        "settings differ from this run's.", RUN);
    auto* resume_force = app_->add_flag("--resume-force", typed_.control.resume_force,
        with_key("With --resume, continue even when the recorded settings differ. The "
                 "landscape then mixes structures relaxed under different settings: "
                 "energies are not comparable across the join and the stored duplicate "
                 "verdicts were taken against the old thresholds. Implies --resume.",
                 "control.resume_force"))
        ->group(RUN);
    apply_.push_back([resume_force](RunConfig& cfg) {
        if (resume_force->count() == 0) return;
        cfg.control.resume = true;
        cfg.control.resume_force = true;
    });
    app_->add_option("--max-runtime", max_runtime_,
        with_key("Wall-clock budget: stop claiming work after this long and exit cleanly "
                 "(30s, 90m, 7.5h; a bare number is minutes). Every rank finishes the unit "
                 "it is running, delivers it, and rank 0 writes the summary and closes the "
                 "database, so the run is resumable with --resume. Set it below the "
                 "scheduler's limit by at least one work unit -- otherwise the job is killed "
                 "part-way through one and that unit's structures are lost.",
                 "control.max_runtime_s, in seconds"))
        ->group(RUN);
    apply_.push_back([this](RunConfig& cfg) {
        if (max_runtime_.empty()) return;
        const double seconds = parse_duration_seconds(max_runtime_);
        if (seconds < 0)
            throw ConfigError("--max-runtime: '" + max_runtime_ + "' is not a duration. "
                              "Use 30s, 90m, 7.5h, or a bare number of minutes.");
        cfg.control.max_runtime_s = seconds;
    });
    flag("--generate-only", "control.generate_only",
        [](RunConfig& c) -> auto& { return c.control.generate_only; },
        "Stop after structure generation: no relaxation at all. The generated "
        "landscape is still scored, deduplicated and written, with stage 0 as the "
        "stage it is counted at, so extract/prune/summary work on it as usual.", RUN);
    setting("--batch-size", "control.batch_size",
        [](RunConfig& c) -> auto& { return c.control.batch_size; },
        "Structures per batch (0 = sized from the device and the molecule)", RUN);
    choice("--batch-sizing", "control.batch_sizing",
        [](RunConfig& c) -> auto& { return c.control.batch_sizing; }, {"atoms", "legacy"},
        "How the automatic batch size is chosen: 'atoms' uses however many structures "
        "fill --batch-atoms, 'legacy' rounds that down to a power of two as earlier "
        "versions did. Legacy lands the average launch at about two thirds of the atom "
        "target on a 25-space-group run. Provided so the two can be measured against "
        "each other; ignored when --batch-size is given.", RUN);
    setting("--batch-atoms", "control.batch_atoms",
        [](RunConfig& c) -> auto& { return c.control.batch_atoms; },
        "Atoms per kernel launch the automatic sizing aims for (0 = 32768 on GPU, 2048 "
        "on CPU). Sweep this to find where the pair kernel stops getting faster with "
        "more structures per launch.", RUN);
    setting("--result-queue-mb", "control.result_queue_mb",
        [](RunConfig& c) -> auto& { return c.control.result_queue_mb; },
        "Ceiling on rank 0's pending-result queue (0 = no ceiling). Results arrive "
        "faster than one thread can deduplicate and store them; past this much "
        "waiting work the ranks feeding rank 0 wait instead of rank 0 growing until "
        "the node runs out of memory. Raise it if the run reports having throttled "
        "and rank 0 has memory to spare.", RUN);
    flag("--sg-affinity", "control.sg_affinity",
        [](RunConfig& c) -> auto& { return c.control.sg_affinity; },
        "Give each rank a preferred set of space groups, weighted by how much work "
        "each carries, instead of letting every rank claim from all of them. A rank "
        "whose own set is exhausted still takes work from anywhere, so it never idles. "
        "OFF by default: it was meant to stop the ML stage relaxing a basin once per "
        "rank that touched it, but measured on 32 ranks the relaxations per basin were "
        "1.09 without it and 1.08 with, and the wall clock was unchanged. Worth trying "
        "on a full-size run, where there are more work units per space group and so "
        "more ranks touching each basin.", RUN);

    // --- Generator -----------------------------------------------------------
    setting("--tvp", "generator.tvp", [](RunConfig& c) -> auto& { return c.generator.tvp; },
        "Target volume parameter: the generator aims for a cell this many times the "
        "summed molecular volume", GENERATOR);
    choice("--collision", "generator.collision",
        [](RunConfig& c) -> auto& { return c.generator.collision; },
        {"direct", "rigid_press", "fft_pack"},
        "Collision handling: direct (reject overlapping seeds), rigid_press "
        "(post-generation rigid-body press) or fft_pack (FFT overlap search)", GENERATOR);
    setting("--gen-nudge", "generator.nudge_budget",
        [](RunConfig& c) -> auto& { return c.generator.nudge_budget; },
        "Generator hull-MTV COM-nudge budget per seed (0 = disable nudge)", GENERATOR);
    toggle("gen-contract", "generator.contract",
        [](RunConfig& c) -> auto& { return c.generator.contract; },
        "Expand-then-contract densification in the generator: shrink the cell back "
        "to the clash boundary once a structure is clear. Off reproduces the older, "
        "roughly 4x looser cells.", GENERATOR);
    setting("--fft-k", "generator.fft_k_per_trial",
        [](RunConfig& c) -> auto& { return c.generator.fft_k_per_trial; },
        "FFT packing: positions kept per (cell, orientation) trial. One FFT gives "
        "the whole translation landscape, so this is how much of it is used -- and "
        "it also sizes each trial's block of structure identities, which is why "
        "--resume refuses a run that changes it.", GENERATOR);
    setting("--fft-grid", "generator.fft_grid_spacing",
        [](RunConfig& c) -> auto& { return c.generator.fft_grid_spacing; },
        "FFT packing: grid spacing in Angstrom. Finer resolves the repulsive wall "
        "better and costs the cube of the refinement.", GENERATOR);
    setting("--fft-tol", "generator.fft_collision_tol",
        [](RunConfig& c) -> auto& { return c.generator.fft_collision_tol; },
        "FFT packing: overlap in A^3 the map will still pass to the exact check", GENERATOR);
    setting("--rp-iterations", "generator.rp_iterations",
        [](RunConfig& c) -> auto& { return c.generator.rp_iterations; },
        "Rigid press stage 0 iterations", GENERATOR);
    setting("--rp-pressure", "generator.rp_pressure_gpa",
        [](RunConfig& c) -> auto& { return c.generator.rp_pressure_gpa; },
        "Rigid press external pressure (GPa)", GENERATOR);
    setting("--rp-max-strain-step", "generator.rp_max_strain_step",
        [](RunConfig& c) -> auto& { return c.generator.rp_max_strain_step; },
        "Rigid press max strain change per component per iteration", GENERATOR);

    // --- Optimizer -----------------------------------------------------------
    setting("--iterations", "optimizer.n_iterations",
        [](RunConfig& c) -> auto& { return c.optimizer.n_iterations; },
        "Stage-2 (multipole) iteration cap per structure", OPTIMIZER);
    setting("--iterations-stage1", "optimizer.n_iterations_stage1",
        [](RunConfig& c) -> auto& { return c.optimizer.n_iterations_stage1; },
        "Stage-1 (charge-only) iteration cap per structure", OPTIMIZER);
    setting("--stage1-pressure", "optimizer.stage1_pressure_gpa",
        [](RunConfig& c) -> auto& { return c.optimizer.stage1_pressure_gpa; },
        "External pressure applied during stage-1 charge-only optimization (GPa). "
        "mol-cspy uses 0.1. Stage-2 multipole opt always runs at P=0 so reported "
        "energies remain lattice energies.", OPTIMIZER);
    // Streaming stops a structure as soon as it meets this tolerance, so it sets
    // the landscape's quality: a loose tolerance inflates the unique count.
    setting("--grad-tol", "optimizer.grad_tol",
        [](RunConfig& c) -> auto& { return c.optimizer.grad_tol; },
        "Optimizer convergence tolerance: max |gradient| over the scaled DOFs, "
        "i.e. Hartree per Bohr of atomic displacement (0 disables). It is the "
        "harvest trigger for both streaming stages, so it decides directly how "
        "polished a stored geometry is. Under --lockstep-stage1 / "
        "--lockstep-stage2 that stage runs its full iteration count regardless "
        "and the tolerance only decides what the s1conv / conv columns report.",
        OPTIMIZER);
    flag("--fixed-cell", "optimizer.fixed_cell",
        [](RunConfig& c) -> auto& { return c.optimizer.fixed_cell; },
        "Disable cell optimization (keep cell parameters fixed during optimization)",
        OPTIMIZER);
    toggle("fix-angles-stage1", "optimizer.fix_angles_stage1",
        [](RunConfig& c) -> auto& { return c.optimizer.fix_angles_stage1; },
        "During charge-only stage 1, hold cell angles fixed (length-only strain DOFs) "
        "to prevent sheared-flat cells from no-multipole optima.", OPTIMIZER);
    // Streaming harvests a slot when it converges and refills from the pending
    // pool; lockstep runs every slot for the full iteration count.
    auto* lockstep_stage2 = app_->add_flag("--lockstep-stage2")
        ->description(with_key(
            "Run stage 2 in lockstep: every slot iterates the full --iterations "
            "count instead of being harvested when it converges. Slower for the "
            "same result; the streaming default is normally what you want. Takes "
            "stage 1 with it.", "optimizer.stream=false"))
        ->group(OPTIMIZER);
    apply_.push_back([lockstep_stage2](RunConfig& cfg) {
        if (lockstep_stage2->count() == 0) return;
        // Stage 1 can only stream into a streaming stage 2.
        cfg.optimizer.stream = false;
        cfg.optimizer.stream_stage1 = false;
    });
    flag("!--lockstep-stage1", "optimizer.stream_stage1",
        [](RunConfig& c) -> auto& { return c.optimizer.stream_stage1; },
        "Run stage 1 in lockstep: every structure iterates the full "
        "--iterations-stage1 instead of leaving when it converges. Uses one pool "
        "instead of two, so it is the lower-memory option.", OPTIMIZER);
    setting("--stream-pool-mult", "optimizer.stream_pool_mult",
        [](RunConfig& c) -> auto& { return c.optimizer.stream_pool_mult; },
        "Streaming pool size as a multiple of batch size. Also the size of the work "
        "unit a rank claims, and therefore how much a killed run loses: results are "
        "written only once a unit finishes, so a crash discards everything in flight. "
        "Throughput saturates at 4 on the molecules measured, so raising it buys little "
        "and widens that window.", OPTIMIZER);
    setting("--refill-interval", "optimizer.refill_interval",
        [](RunConfig& c) -> auto& { return c.optimizer.refill_interval; },
        "Streaming harvest/refill checkpoint cadence in iterations", OPTIMIZER);
    setting("--max-iters-per-slot", "optimizer.max_iters_per_slot",
        [](RunConfig& c) -> auto& { return c.optimizer.max_iters_per_slot; },
        "Streaming per-slot iteration cap (0 = use --iterations)", OPTIMIZER);
    choice("--sampler", "optimizer.sampler",
        [](RunConfig& c) -> auto& { return c.optimizer.sampler; }, {"trust_region", "mc"},
        "Stage-2 sampler: trust_region (relax each structure to its nearest local "
        "minimum -- what a klasp landscape has always been) or mc (a Metropolis "
        "chain that perturbs the rigid-body and strain DOFs, accepts by the "
        "Boltzmann criterion, and keeps the best structure it saw). mc runs on the "
        "same streaming engine, so it is incompatible with --lockstep-stage2.",
        OPTIMIZER);
    setting("--mc-temperature", "optimizer.mc_temperature_k",
        [](RunConfig& c) -> auto& { return c.optimizer.mc_temperature_k; },
        "Metropolis temperature in Kelvin. The criterion is exp(-dE/RT) over "
        "kJ/mol, so 300 K is kT = 2.5 kJ/mol -- cold against a lattice energy, and "
        "a chain that mostly walks downhill. Raise it to buy barrier crossings.",
        OPTIMIZER);
    setting("--mc-moves", "optimizer.mc_moves",
        [](RunConfig& c) -> auto& { return c.optimizer.mc_moves; },
        "Metropolis moves per structure. This is what the chain finishes on in "
        "place of a convergence test, so it is also how long a slot is held.", OPTIMIZER);
    setting("--mc-step-trans", "optimizer.mc_step_trans",
        [](RunConfig& c) -> auto& { return c.optimizer.mc_step_trans; },
        "Metropolis COM move, in fractional coordinates.", OPTIMIZER);
    setting("--mc-step-rot", "optimizer.mc_step_rot_deg",
        [](RunConfig& c) -> auto& { return c.optimizer.mc_step_rot_deg; },
        "Metropolis orientation move, in degrees.", OPTIMIZER);
    setting("--mc-step-strain", "optimizer.mc_step_strain",
        [](RunConfig& c) -> auto& { return c.optimizer.mc_step_strain; },
        "Metropolis cell move, as a dimensionless strain along the lattice-allowed "
        "directions.", OPTIMIZER);
    setting("--mc-min-contact", "optimizer.mc_min_contact",
        [](RunConfig& c) -> auto& { return c.optimizer.mc_min_contact; },
        "Shortest intermolecular contact a Metropolis proposal may have, in "
        "Angstrom. Not a tolerance: the force field is unbounded below at short "
        "range, so without this the chain walks into the singularity and every "
        "step down it is honestly downhill. 0 disables it.", OPTIMIZER);

    // --- Force field ---------------------------------------------------------
    // Any rank 0 to 4; range-checked so a typo is a usage error.
    setting("--max-order", "forcefield.max_order",
        [](RunConfig& c) -> auto& { return c.forcefield.max_order; },
        "Override multipole rank, 0 to 4. Interactions are kept to lA + lB <= rank. "
        "-1 uses whatever rank the basis carries.", FORCEFIELD)
        ->check(CLI::Range(-1, 4));
    keyed(add_ewald_accuracy_option(app_, typed_.forcefield.ewald_accuracy),
          "forcefield.ewald_accuracy",
          [](RunConfig& c) -> auto& { return c.forcefield.ewald_accuracy; }, FORCEFIELD);
    flag("--wolf", "forcefield.wolf", [](RunConfig& c) -> auto& { return c.forcefield.wolf; },
        "Use Wolf DSF instead of Ewald (no reciprocal sum, faster but less accurate). "
        "Implemented in the lockstep stage-2 kernel only, so it needs "
        "--lockstep-stage2.", FORCEFIELD);
    // Stored in Bohr; the flag takes Angstrom.
    wolf_cutoff_ang_ = typed_.forcefield.wolf_cutoff / ANGSTROM_TO_BOHR;
    auto* wolf_cutoff = app_->add_option("--wolf-cutoff", wolf_cutoff_ang_,
        with_key("Wolf sum cutoff in Angstrom (0 = auto from Ewald rcut)",
                 "forcefield.wolf_cutoff, in Bohr"))
        ->capture_default_str()
        ->group(FORCEFIELD);
    apply_.push_back([this, wolf_cutoff](RunConfig& cfg) {
        if (wolf_cutoff->count() > 0)
            cfg.forcefield.wolf_cutoff = wolf_cutoff_ang_ * ANGSTROM_TO_BOHR;
    });
    setting("--mp-cutoff", "forcefield.mp_cutoff_ang",
        [](RunConfig& c) -> auto& { return c.forcefield.mp_cutoff_ang; },
        "Shorter cutoff (Angstrom) for the rank>=2 multipole term (e.g. 10-12), "
        "tapered to zero over its last 2 A. Big stage-2 speedup at an accuracy cost "
        "(multipoles decay >=1/r^3); measure it with `klasp energy --mp-cutoff` on "
        "a landscape's structures. 0 = full cutoff.", FORCEFIELD);
    flag("--mp-single", "forcefield.mp_single",
        [](RunConfig& c) -> auto& { return c.forcefield.mp_single; },
        "Run the rank>=2 multipole term in single precision while a structure's "
        "scaled gradient is above optimizer.mp_single_switch_tol, then in double "
        "to convergence. The single-precision kernel is 2.6x faster and its "
        "energies are within 1e-5 kJ/mol, but its force noise sits near 3e-7 on "
        "the scaled gradient, and below ~1e-5 the trust region churns on it -- so "
        "the switch cannot go lower, and the gain is whatever fraction of the "
        "iterations a structure spends above it. Measured: paracetamol 16% faster "
        "at --grad-tol 1e-6, nothing at the default 1e-8, and slower on acetic "
        "acid. Off by default.", FORCEFIELD);

#ifdef KLASP_HAS_PET
    // --- ML stage ------------------------------------------------------------
    flag("--pet-relax", "pet.relax", [](RunConfig& c) -> auto& { return c.pet.relax; },
        "Final PET-MAD relaxation + re-scoring stage (emits stage-3 results)", ML_STAGE);
    toggle("pet-symmetry", "pet.symmetry", [](RunConfig& c) -> auto& { return c.pet.symmetry; },
        "PET stage: relax with the space group held fixed rather than moving atoms "
        "freely.", ML_STAGE);
    add_pet_model_options(app_, typed_.pet.model, typed_.pet.model_json,
                          typed_.pet.model_weights);
    auto* model = app_->get_option("--model");
    auto* model_json = app_->get_option("--model-json");
    auto* model_weights = app_->get_option("--model-weights");
    model->description(with_key(model->get_description(), "pet.model"))->group(ML_STAGE);
    model_json->description(with_key(model_json->get_description(), "pet.model_json"))
        ->group(ML_STAGE);
    model_weights->description(with_key(model_weights->get_description(), "pet.model_weights"))
        ->group(ML_STAGE);
    // A model named here replaces the config's resolved paths, unless the paths
    // are typed too (applied after this).
    apply_.push_back([this, model](RunConfig& cfg) {
        if (model->count() == 0) return;
        cfg.pet.model = typed_.pet.model;
        cfg.pet.model_json.clear();
        cfg.pet.model_weights.clear();
    });
    when_typed(model_json, [](RunConfig& c) -> auto& { return c.pet.model_json; });
    when_typed(model_weights, [](RunConfig& c) -> auto& { return c.pet.model_weights; });
    setting("--pet-iterations", "pet.iterations",
        [](RunConfig& c) -> auto& { return c.pet.iterations; },
        "PET stage: max L-BFGS iterations (0 = the backend's 300)", ML_STAGE);
    setting("--pet-force-tol", "pet.force_tol",
        [](RunConfig& c) -> auto& { return c.pet.force_tol; },
        "PET stage: force tolerance, Ha/Bohr (0 = the backend's 1e-3)", ML_STAGE);
    setting("--pet-stress-tol", "pet.stress_tol",
        [](RunConfig& c) -> auto& { return c.pet.stress_tol; },
        "PET stage: stress tolerance, Ha/Bohr^3 (0 = the backend's 5e-6)", ML_STAGE);
    toggle("pet-precondition", "pet.precondition",
        [](RunConfig& c) -> auto& { return c.pet.precondition; },
        "PET stage: precondition with the model Hessian. Worth about 1.5x in "
        "iterations, with more structures converged and deeper; --no-pet-precondition "
        "is there to measure that, not to run without it.", ML_STAGE);
    setting("--pet-pass", "pet.pass_rounds",
        [](RunConfig& c) -> auto& { return c.pet.pass_rounds; },
        "Relax stage 3 in passes of this many L-BFGS rounds, carrying unconverged "
        "structures forward and topping the batch up from those not yet started, "
        "instead of one pass to --pet-iterations. Costs curvature at every boundary "
        "and buys a few more converged structures at about the same wall time "
        "(2045/2048 against 2035/2048 on one 2048-structure run). 0 = one pass.",
        ML_STAGE);
    setting("--pet-min-contact", "pet.min_contact",
        [](RunConfig& c) -> auto& { return c.pet.min_contact; },
        "PET stage: drop candidates with any contact below this (Angstrom; 0 = off)",
        ML_STAGE);
    toggle("pet-dedup", "pet.dedup", [](RunConfig& c) -> auto& { return c.pet.dedup; },
        "PET stage: relax one candidate per basin, by the same dedup thresholds the "
        "run reports its unique count with. The stage is the most expensive part of "
        "a run, and nearly all of it is otherwise spent re-relaxing the same crystals. "
        "--no-pet-dedup relaxes every structure, to measure what the gate costs or to "
        "put a stage-3 energy on every row.", ML_STAGE);
#endif

    // --- Deduplication -------------------------------------------------------
    // Shared with prune's offline rule; defaults live on DedupConfig.
    add_dedup_options(app_, typed_.dedup);
    keyed(app_->get_option("--amd-tol"), "dedup.amd_tol",
          [](RunConfig& c) -> auto& { return c.dedup.amd_tol; }, DEDUP);
    keyed(app_->get_option("--energy-bin"), "dedup.energy_bin",
          [](RunConfig& c) -> auto& { return c.dedup.energy_bin; }, DEDUP);
    keyed(app_->get_option("--density-bin"), "dedup.density_bin",
          [](RunConfig& c) -> auto& { return c.dedup.density_bin; }, DEDUP);
    keyed(app_->get_option("--powder-dedup"), "dedup.use_powder",
          [](RunConfig& c) -> auto& { return c.dedup.use_powder; }, DEDUP);
    keyed(app_->get_option("--powder-cos-thresh"), "dedup.powder_cos_thresh",
          [](RunConfig& c) -> auto& { return c.dedup.powder_cos_thresh; }, DEDUP);
    keyed(app_->get_option("--powder-cdtw-thresh"), "dedup.powder_cdtw_thresh",
          [](RunConfig& c) -> auto& { return c.dedup.powder_cdtw_thresh; }, DEDUP);
    keyed(app_->get_option("--powder-cdtw-band"), "dedup.powder_cdtw_band",
          [](RunConfig& c) -> auto& { return c.dedup.powder_cdtw_band; }, DEDUP);
    keyed(app_->get_option("--cross-sg"), "dedup.cross_sg",
          [](RunConfig& c) -> auto& { return c.dedup.cross_sg; }, DEDUP);
    // Not keyed(): typing it also turns on output.store_pdd.
    auto* dedup_pdd = app_->get_option("--dedup-pdd");
    dedup_pdd->description(with_key(dedup_pdd->get_description(), "dedup.pdd_tol"));
    dedup_pdd->group(DEDUP);
    when_typed(dedup_pdd, [](RunConfig& c) -> auto& { return c.dedup.pdd_tol; });
    // Follows --dedup-pdd unless given explicitly. Gates the PDD computation and
    // its MPI payload as well as storage.
    auto* store_pdd = app_->add_flag("--store-pdd,!--no-store-pdd", typed_.output.store_pdd,
        with_key("Store the PDD (pointwise distance distribution) per structure. Defaults "
                 "to whether --dedup-pdd is in use, since nothing else reads it; it is the "
                 "largest descriptor by some margin.", "output.store_pdd"))
        ->group(DEDUP);
    apply_.push_back([this, store_pdd, dedup_pdd](RunConfig& cfg) {
        if (store_pdd->count() > 0)
            cfg.output.store_pdd = typed_.output.store_pdd;
        else if (dedup_pdd->count() > 0 && typed_.dedup.pdd_tol > 0)
            cfg.output.store_pdd = true;
    });

    // --- Output --------------------------------------------------------------
    setting("-o,--output-dir", "output.dir", [](RunConfig& c) -> auto& { return c.output.dir; },
        "Output directory", OUTPUT);
    flag("--store-all-stages", "output.store_all_stages",
        [](RunConfig& c) -> auto& { return c.output.store_all_stages; },
        "Persist every optimisation stage, not just the one the landscape is ranked "
        "on: stage 0 (raw generated) and stage 1 (charge-only) as well. Diagnostic; "
        "forces the lockstep path, because streaming harvests stage 1 into the "
        "stage-2 pool without packing it.", OUTPUT);
    toggle("store-diagnostics", "output.store_diagnostics",
        [](RunConfig& c) -> auto& { return c.output.store_diagnostics; },
        "Store the per-structure convergence diagnostics (gradient, iterations, "
        "trust-region exit state, stage-1 convergence) in the database and the "
        "status table.", OUTPUT);
    setting("--max-vacuum", "output.max_vacuum",
        [](RunConfig& c) -> auto& { return c.output.max_vacuum; },
        "Drop structures whose largest empty layer across the cell exceeds this "
        "(Angstrom). Negative, the default, works it out from the potential "
        "cutoff -- half of it, which is 7.5 A at the usual 15 A. That is the "
        "quantity that decides the question: two slabs interact until the gap "
        "reaches the cutoff, and past it every structure in a space group comes "
        "back with the same energy because the molecules cannot see their own "
        "images. 0 keeps them all and only sets the vacuum-slab warning bit. "
        "Measured over low-index lattice planes, so it sees a layer on a diagonal "
        "that density cannot: a dense packing sits near 1.5 A and real slabs run "
        "from 10 A up. A dropped structure has already consumed its seed by the "
        "time it is measured, so the run finishes that many short of the target "
        "and says so at the end rather than generating a replacement.", OUTPUT);
    flag("--perf", "output.perf", [](RunConfig& c) -> auto& { return c.output.perf; },
        "Write per-batch and per-kernel performance CSVs", OUTPUT);
    flag("--perf-sync", "output.perf_sync",
        [](RunConfig& c) -> auto& { return c.output.perf_sync; },
        "With --perf: fence after every kernel for accurate GPU timing (serializes; slow)",
        OUTPUT);
}

RunConfig CspCommand::config() const {
    RunConfig cfg;
    if (!config_path_.empty()) load_config_file(cfg, config_path_);
    for (const auto& assignment : sets_) {
        std::string error;
        if (!apply_config_override(cfg, assignment, error))
            throw ConfigError("--set " + assignment + ": " + error);
    }
    if (!inputs_.empty()) cfg.control.basis_paths = inputs_;
    for (const auto& apply : apply_) apply(cfg);
    return cfg;
}

}  // namespace klasp::cli
