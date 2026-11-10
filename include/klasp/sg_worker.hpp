#pragma once

#include <atomic>

#include "klasp/types.hpp"
#include "klasp/molecule.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_stream.hpp"
#include "klasp/stream_mc.hpp"
#include "klasp/csp.hpp"
#include "klasp/io.hpp"
#include "klasp/database.hpp"
#include "klasp/config.hpp"
#include "klasp/dedup.hpp"
#include "klasp/descriptors.hpp"
#include "klasp/powder.hpp"
#include "klasp/perf_recorder.hpp"
#ifdef KLASP_HAS_PET
// Held by value while stage-3 inputs accumulate (see pet_stage_inputs).
#include "klasp/pet_relax.hpp"
#endif

#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <utility>

namespace klasp {

class PetEvaluator;  // include/klasp/pet_bridge.hpp (pointer only)


// Per-space-group worker: generation and batched optimisation.
class SGWorker {
public:
    SGWorker() = default;

    void init(int space_group, int target, int batch_size,
              const BasisConfig& basis, Workspace& ws,
              const BatchOptimizeSettings& bsettings,
              const std::vector<double>& body_pos,
              const std::vector<double>& cov_radii,
              const std::vector<double>& vdw_radii,
              double mol_mass, double tvp,
              int effective_max_rank, int rank, int nranks);

    // Generate and optimize one batch. Returns results (caller sends them).
    // batch_override: if > 0, run this many structures instead of batch_size_
    // (must be <= batch_size_), to taper the final batch.
    std::vector<StructureResult> run_batch(int batch_override = 0);

    // Everything a worker takes from the run's configuration. Runtime wiring
    // (recorder, abort flag, callbacks, PET evaluator) is set separately.
    // `n_asym_atoms` and `max_vacuum` come from the basis; max_vacuum is already
    // resolved (see resolve_max_vacuum).
    void configure(const RunConfig& cfg, int n_asym_atoms, double max_vacuum) {
        const auto& gen = cfg.generator;
        const auto& opt = cfg.optimizer;
        const auto& out = cfg.output;
        const auto& ctl = cfg.control;

        max_vacuum_ = max_vacuum;
        // Off under --batch-sizing legacy.
        wide_slots_ = ctl.batch_sizing != BatchSizing::LEGACY;

        collision_method_ = gen.collision;
        gen_nudge_ = gen.nudge_budget;
        gen_contract_ = gen.contract;
        rp_iterations_ = gen.rp_iterations;
        rp_pressure_gpa_ = gen.rp_pressure_gpa;
        rp_max_strain_step_ = gen.rp_max_strain_step;
        fft_k_per_trial_ = gen.fft_k_per_trial;
        fft_grid_spacing_ = gen.fft_grid_spacing;
        fft_collision_tol_ = gen.fft_collision_tol;
        fft_max_aspect_ = gen.fft_max_aspect;

        store_pdd_ = out.store_pdd;
        pdd_rows_ = n_asym_atoms;
        store_diagnostics_ = out.store_diagnostics;
        store_all_stages_ = out.store_all_stages;

        stream_ = opt.stream;
        stream_pool_mult_ = opt.stream_pool_mult;
        refill_interval_ = opt.refill_interval;
        max_iters_per_slot_ = opt.max_iters_per_slot;
        stream_stage1_ = opt.stream_stage1 && stream_;

        // Metropolis is a streaming-engine stepper; main.cpp rejects it with
        // --lockstep-stage2.
        sampler_ = opt.sampler;
        mc_kT_ = static_cast<RealType>(MC_GAS_CONSTANT_KJ * opt.mc_temperature_k);
        mc_moves_ = opt.mc_moves;
        mc_step_trans_ = static_cast<RealType>(opt.mc_step_trans);
        mc_step_rot_ = static_cast<RealType>(opt.mc_step_rot_deg * M_PI / 180.0);
        mc_step_strain_ = static_cast<RealType>(opt.mc_step_strain);
        mc_min_contact_ = static_cast<RealType>(opt.mc_min_contact);

        fixed_cell_ = opt.fixed_cell;
        fix_angles_stage1_ = opt.fix_angles_stage1;
        stage1_iterations_ = opt.n_iterations_stage1;
        stage1_pressure_gpa_ = opt.stage1_pressure_gpa;

        generate_only_ = ctl.generate_only;

#ifdef KLASP_HAS_PET
        pet_symmetry_ = cfg.pet.symmetry;
        pet_max_iter_ = cfg.pet.iterations;
        pet_force_tol_ = cfg.pet.force_tol;
        pet_stress_tol_ = cfg.pet.stress_tol;
        pet_min_contact_ = cfg.pet.min_contact;
        pet_precondition_ = cfg.pet.precondition;
        pet_pass_rounds_ = cfg.pet.pass_rounds;
        pet_device_ = cfg.pet.device;
#endif
    }

    void set_perf(PerfRecorder* p) { perf_ = p; }
    // Structures produced by the most recent run_batch(). This is what the CSP
    // target counts; it differs from results.size() because optional stages
    // (0 = pre-optimisation, 3 = PET) add rows for the same crystal.
    int last_batch_structures() const { return last_batch_structures_; }
    // The stage whose rows correspond one-to-one with structures; only rows at
    // that stage may be dropped.
    void set_primary_stage(int s) { primary_stage_ = s; }
    // Stop optimising at the next checkpoint when this goes true. See
    // BatchOptimizeSettings::abort.
    void set_abort_flag(const std::atomic<bool>* f) { abort_flag_ = f; }
    long long take_vacuum_dropped() { const auto n = n_vacuum_dropped_; n_vacuum_dropped_ = 0; return n; }
    // PDDs aligned 1:1 with the results from the most recent run_batch().
    const std::vector<std::vector<double>>& last_batch_pdds() const { return last_batch_pdds_; }
    // Explicit PET geometry for the stage-3 rows of the most recent run_batch(),
    // keyed by seed. Empty unless the PET stage ran.
    const std::unordered_map<int, FlexibleStructure>& last_batch_flex() const {
        return last_batch_flex_;
    }
    // Deliver results as they become ready instead of at the end of the work
    // unit. When set, run_batch returns nothing. The map is the PET geometry
    // keyed by seed, non-empty only for stage-3 rows.
    using EmitFn = std::function<void(const std::vector<StructureResult>&,
                                      const std::vector<std::vector<double>>&,
                                      const std::unordered_map<int, FlexibleStructure>&)>;
    void set_emit(EmitFn f) { emit_ = std::move(f); }

    // Enable a PET-MAD final relaxation stage: after stage 2, each candidate is
    // re-relaxed under PET and emitted as a stage-3 result. `e` is shared and
    // orchestrator-owned; `symmetry` keeps the space group fixed during relax.
    void set_pet_relax(PetEvaluator* e, bool symmetry, double reference_kjmol) {
        pet_eval_ = e;
        pet_relax_ = (e != nullptr);
        pet_symmetry_ = symmetry;
        pet_reference_ = reference_kjmol;
    }
    // Relax one candidate per basin instead of every structure, using a
    // rank-local dedup index over the run's thresholds. Having seen fewer
    // structures, it merges no more than rank 0's index, so every basin keeps its
    // stage-3 row (coverage test in tests/test_pdd.cpp). enabled=false relaxes all.
    void set_pet_dedup(const DedupConfig& cfg, bool enabled) {
        pet_dedup_.configure(cfg);
        pet_dedup_enabled_ = enabled;
    }

    void mark_done() { done_ = true; }
    bool done() const { return done_; }
    int local_valid() const { return local_valid_; }
    int last_rejected() const { return last_rejected_; }
    // Candidates the PET stage produced no stage-3 row for (clash screen or
    // non-finite energy). Drains on read; rank 0 needs it to tell when a space
    // group's ML work is finished.
    int take_pet_dropped() { const int n = pet_dropped_; pet_dropped_ = 0; return n; }
    // Candidates the rank-local dedup gate kept out of the ML stage. Drains on
    // read, like take_pet_dropped.
    int take_pet_skipped() { const int n = pet_skipped_; pet_skipped_ = 0; return n; }
    int space_group() const { return space_group_; }
    int target() const { return target_; }

    // For adaptive batch sizing.
    void set_remaining(int r) { remaining_ = r; }

    // Source of generator seeds: returns the first of `n` contiguous global seeds
    // for this SG, shared across ranks (see WorkWindow), so the sampled set does
    // not depend on the rank count. Unset: a local contiguous counter.
    void set_seed_claimer(std::function<long long(int)> claimer) {
        claim_seeds_ = std::move(claimer);
    }

private:
    long long claim_seeds(int n) {
        long long base = claim_seeds_ ? claim_seeds_(n)
                                      : static_cast<long long>(next_seed_);
        next_seed_ = static_cast<int>(base + n);   // keep as a perf-label position
        return base;
    }

    int space_group_ = 0;
    int Z_ = 0;
    int target_ = 0;
    int local_valid_ = 0;
    int next_seed_ = 1;   // local fallback when no global seed claimer is set
    std::function<long long(int)> claim_seeds_;
    int remaining_ = 0;
    bool done_ = false;

    int batch_size_ = 256;
    int effective_max_rank_ = 0;
    int rank_ = 0, nranks_ = 1;
    double mol_mass_ = 0;
    double tvp_ = 1.2;

    MoleculeData csp_mol_;
    BatchWorkspace bws_;
    bool workspace_allocated_ = false;

    Kokkos::View<RealType**, Kokkos::HostSpace> h_cell_params_;

    PerfRecorder* perf_ = nullptr;
    CollisionMethod collision_method_ = CollisionMethod::DIRECT;
    bool fixed_cell_ = false;
    int gen_nudge_ = 16;        // generator hull-MTV nudge budget (0 = off)
    bool gen_contract_ = true;  // generator expand-then-contract densification
    // FFT packing. k_per_trial also sizes each trial's identity block, so --resume
    // compares it.
    int fft_k_per_trial_ = 8;
    double fft_grid_spacing_ = 0.5;
    double fft_collision_tol_ = 0.1;
    double fft_max_aspect_ = 5.0;
    int rp_iterations_ = 20;
    double rp_pressure_gpa_ = 5.0;
    double rp_max_strain_step_ = 0.05;
    // Running acceptance estimate (valids / seeds tried), EMA, used to size each
    // generation block to ~GEN_MARGIN x one batch of valids.
    double accept_est_ = 0.8;          // optimistic seed: most SGs accept ~0.9
    View1D<int> d_batch_seeds_;        // device-side seed storage for compaction
    // Generation/optimization workspace capacity, a little above batch_size_ so a
    // saturated block's whole valid yield (~1.2x a batch) fits.
    int gen_cap_ = 0;
    std::vector<std::string> atom_elements_;  // element symbol per atom; padding is "X"
    // Flat indices of the non-padding atoms, in order. See klasp/padding.hpp.
    std::vector<int> real_atoms_;
    // Real site count per component, for the symmetry layout handed to PET.
    std::vector<int> basis_mol_n_sites_;

    // PET-MAD final relaxation stage (optional).
    PetEvaluator* pet_eval_ = nullptr;  // shared, orchestrator-owned (not deleted here)
    bool pet_relax_ = false;
    bool pet_symmetry_ = true;
    double pet_reference_ = 0.0;  // isolated-molecule energy (kJ/mol/fu) -> lattice energy
    int pet_last_evals_ = 0;      // batched PET evals (lockstep rounds) of the last stage-3, for --perf
    bool pet_precondition_ = true;
    bool pet_device_ = true;
    int pet_pass_rounds_ = 0;     // 0 -> a single pass to the iteration cap
    int pet_max_iter_ = 0;        // 0 -> PetRelaxOptions default
    // Stage-3 cost of the current work unit, for --perf; accumulated by
    // pet_flush_results and read by both drivers.
    double pet_stage_ms_ = 0;
    int pet_n_struct_ = 0;
    int pet_atoms_ = 0;
    double pet_min_contact_ = 0.7;  // Angstrom; 0 disables the clash screen
    int pet_clashed_ = 0;           // dropped by that screen, reported per flush
    double pet_worst_clash_ = 1e300;
    int pet_worst_seed_ = -1;
    double pet_force_tol_ = 0.0;  // <=0 -> PetRelaxOptions default
    double pet_stress_tol_ = 0.0; // <=0 -> PetRelaxOptions default

    // Stage 3 runs in two halves so one PET pass covers a whole work unit: the
    // geometry is extracted per finalize_stream chunk (the next chunk overwrites
    // the workspace) and relaxed in one flush per unit, which splits it at PET's
    // own atom budget.
    void pet_stage_inputs(const std::vector<StructureResult>& rows,
                          const std::vector<std::vector<double>>& pdds);
    void pet_flush_results(std::vector<StructureResult>& results);
#ifdef KLASP_HAS_PET
    std::vector<CrystalGeometry> pet_pending_structs_;
    std::vector<StructureResult> pet_pending_rows_;      // stage-2 metadata per staged structure
    std::vector<std::vector<double>> pet_pending_pdds_;  // stage-2 PDD per staged structure
    PetSymmetry pet_pending_sym_;
    bool pet_pending_use_sym_ = false;
    bool pet_sym_warned_ = false;   // warn once per worker when symmetry is unusable
#endif
    PowderFormFactors powder_ff_;  // per-atom Cromer-Mann coefficients (device)
    bool fix_angles_stage1_ = true;    // restrict cell to length-only during charge-only pass
    int stage1_iterations_ = 0;        // 0 = use bsettings_->n_iterations
    double stage1_pressure_gpa_ = 0.0; // external pressure applied during stage-1 (charge-only) opt only
    bool store_all_stages_ = false;    // persist stage 0 and stage 1 alongside the primary stage
    bool generate_only_ = false;       // stop after generation; stage 0 is the primary stage
    std::vector<StructureResult> extra_stage_results_pending_;  // stage 0/1 rows to merge at run_batch end

    bool store_pdd_ = true;            // compute + return the PDD with each result
    bool store_diagnostics_ = false;   // record optimizer convergence diagnostics per structure
    // --- streaming (pipelined) stage-2 ---
    bool stream_ = false;
    int stream_pool_mult_ = 4;         // pool size P = stream_pool_mult_ * batch_size_
    int refill_interval_ = 10;         // streaming harvest/refill checkpoint cadence (iters)
    int max_iters_per_slot_ = 0;       // per-slot cap (0 = use n_iterations)
    bool stream_stage1_ = false;       // stage 1 through the slot scheduler too
    // --- which stepper stage 2 runs, and what the chain is if it is Metropolis ---
    SamplerKind sampler_ = SamplerKind::TRUST_REGION;
    RealType mc_kT_ = RealType(MC_GAS_CONSTANT_KJ * 300.0);
    int mc_moves_ = 400;
    RealType mc_step_trans_ = RealType(0.03);
    RealType mc_step_rot_ = RealType(5.0 * M_PI / 180.0);
    RealType mc_step_strain_ = RealType(0.01);
    RealType mc_min_contact_ = RealType(1.5);
    // How far past the generator's volume ceiling a relaxing structure may go
    // before it is abandoned. Kept generous: too tight loses real structures.
    static constexpr double VOLUME_ABORT_FACTOR = 20.0;

    // The prior the generator last drew from (reported by generate_one), so a
    // chain can propose inside it.
    SamplingPrior gen_prior_;
    bool gen_prior_valid_ = false;
    StreamPool pool_;                  // pending pool of stage-1-relaxed structures
    StreamPool raw_pool_;              // generated structures awaiting stage 1
    bool pool_allocated_ = false;
    int pdd_rows_ = 0;                 // n_asym (= n_indep * n_sites); fixed per run
    std::vector<std::vector<double>> stage0_pdds_pending_;  // PDDs aligned with stage0 pending
    EmitFn emit_;                      // set -> results stream out per chunk
    int last_batch_structures_ = 0;    // structures (not rows) from the last run_batch()
    double max_vacuum_ = 0.0;
    int primary_stage_ = STAGE_MULTIPOLE;
    const std::atomic<bool>* abort_flag_ = nullptr;
    bool wide_slots_ = true;
    long long n_vacuum_dropped_ = 0;
    std::vector<std::vector<double>> last_batch_pdds_;      // PDDs aligned with last run_batch() result
    // Batch slot each row of the last pack_results() came from.
    std::vector<int> last_pack_slots_;
    // Start of the current work unit, so finalize_stream can price a chunk
    // before emitting it.
    std::chrono::steady_clock::time_point unit_t0_{};
    // PET stage-3 explicit geometry from the last run_batch(), by seed. Written
    // only by pet_flush_results and cleared once per work unit.
    std::unordered_map<int, FlexibleStructure> last_batch_flex_;

    int last_rejected_ = 0;  // rejected seeds from last generation
    int pet_dropped_ = 0;    // PET candidates that produced no stage-3 row
    int pet_skipped_ = 0;    // PET candidates the rank-local dedup gate skipped
    // Skips since the last flush, for the per-unit log line; folded into
    // pet_skipped_ there.
    int pet_skipped_this_unit_ = 0;

    // Rank-local basin index for the stage-3 gate; see set_pet_dedup.
    // Non-assignable (holds a mutex), so it is reconfigured in place.
    StructureDedup pet_dedup_;
    bool pet_dedup_enabled_ = false;

    const BasisConfig* basis_ = nullptr;
    const BatchOptimizeSettings* bsettings_ = nullptr;

    // Per-batch AMD + powder descriptors computed on device, with stage timings.
    // Values are host-resident.
    struct BatchDescriptors {
        AMDResult amd;
        PowderResult powder;
        double amd_ms = 0;
        double powder_ms = 0;
    };
    BatchDescriptors compute_batch_descriptors();

    // Generate + read back one chunk of valid structures into bws_ [0, count).
    // Returns count (0 if none).
    int generate_one(int batch, std::vector<int>& batch_seeds);

    // One entry per relaxation stage; both drivers build from this list and
    // differ only in how they schedule slots.
    struct StageSpec {
        int max_rank = 0;          // multipole rank; 0 = charges only
        int stage_id = STAGE_CHARGES;  // what a row packed after this stage is
        int n_iterations = 0;
        RealType pressure_gpa = 0;
        // Strain mask: length-only (true) or the full lattice-allowed set. The
        // charge-only pass can shear a cell flat, so it runs length-only.
        bool fix_angles = false;
        bool warm_hessian = false; // reuse the previous stage's molecular Hessian
        // Streaming only: per-slot iteration cap, 0 meaning "n_iterations".
        int max_iters_per_slot = 0;
        // Re-anchor the strain reference to the relaxed cell after this stage,
        // rebuild the neighbour/Ewald setup and widen the mask. False for the
        // final stage.
        bool reanchor_after = false;
    };

    // The stage list for this worker's configuration. One or two entries: the
    // charge-only pass, then the multipole pass when effective_max_rank_ > 0.
    std::vector<StageSpec> stage_plan() const;

    // Copy a stage's settings into `rs`, leaving run-level settings alone.
    void apply_stage(const StageSpec& s, BatchOptimizeSettings& rs) const;

    // Stage 0: rigid press. Resolves collisions and minimizes volume, then
    // recomputes cell_params from the deformed cell on device.
    void run_rigid_press(int count);

    // Set the strain DOF mask for a stage and zero the strain state. Sets
    // bws_.dofs.n_cell_dof, so it must run before the stage starts.
    void set_stage_strain_dofs(bool fix_angles);

    // Between stages: take the relaxed cell as the new strain reference, refresh
    // the neighbour/Ewald setup against it, and install the next stage's mask.
    void reanchor_strain(int count, bool next_fix_angles);

    // Energy-only evaluation of the raw generated state, for stage-0 rows
    // (--store-all-stages, --generate-only).
    struct HostDofs {
        Kokkos::View<RealType*, Kokkos::HostSpace> energies;
        Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> translations;
        Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace> rotations;
    };
    HostDofs evaluate_generated(int count);

    // Zero-filled descriptors for intermediate-stage rows, which are never
    // deduplicated.
    BatchDescriptors null_descriptors() const;

    // Streaming (pipelined) stage-2 path: build a pool of stage-1-relaxed
    // structures, keep every GPU slot busy via batch_optimize_stream_tr.
    std::vector<StructureResult> run_batch_stream(int batch);

    // Reload streaming harvest into bws_ in chunks, compute descriptors, pack.
    std::vector<StructureResult> finalize_stream(StreamResult& sr, int n_mols);

    // Pack a finished optimizer batch into host-side StructureResult records
    // (energy, cell params, DOFs, density, descriptors). Drops structures with
    // non-finite energy.
    std::vector<StructureResult> pack_results(
        int stage, int count,
        const std::vector<int>& batch_seeds,
        int n_mols,
        const Kokkos::View<RealType*, Kokkos::HostSpace>& h_E,
        const Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>& h_rt,
        const Kokkos::View<RealType***, Kokkos::LayoutRight, Kokkos::HostSpace>& h_rr,
        const BatchDescriptors& desc,
        std::vector<std::vector<double>>& pdds_out,
        const BatchDiagnostics& diag = {},
        const BatchDiagnostics& s1_diag = {});

    // Device Views for GPU generation kernel
    View2D<RealType> d_body_pos_ang_;  // (total_sites, 3) Angstrom
    View1D<RealType> d_cov_radii_;     // (total_sites)
    View1D<RealType> d_vdw_radii_;     // (total_sites)
    View1D<int> d_mol_n_sites_;        // (n_mols)  real sites per component
    View1D<int> d_mol_site_offsets_;   // (n_mols)  offset into d_body_pos_ang_ per component
    View1D<RealType> d_mol_R_;         // (n_mols)  bounding-sphere radius per component (Angstrom)
};

}  // namespace klasp
