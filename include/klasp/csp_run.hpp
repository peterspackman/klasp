#pragma once

// A CSP run as a state struct and phase functions over it. What varies between
// an MPI job and a single process is the WorkSource, ResultSink and Comm.
//
// The phases run in this order:
//
//   csp_setup          basis, workers, batch geometry, the output directory
//   csp_plan_work      targets and seed cursors, including what a resume owes
//   csp_attach         wire the workers to the sink and the work source
//   csp_start_consumer rank 0 opens the database and starts storing
//   csp_loop           claim, run, report, until DONE
//   csp_shutdown       ordered teardown; see the function
//
// csp_plan_work may throw RunRefused, only after every participant agrees.

#include "klasp/comm.hpp"
#include "klasp/config.hpp"
#include "klasp/csp.hpp"
#include "klasp/dedup.hpp"
#include "klasp/io.hpp"
#include "klasp/perf_recorder.hpp"
#include "klasp/result_consumer.hpp"
#include "klasp/result_queue.hpp"
#include "klasp/sg_worker.hpp"
#include "klasp/types.hpp"
#include "klasp/work_source.hpp"

#ifdef KLASP_HAS_PET
#include "klasp/pet_bridge.hpp"
#endif

#include <atomic>
#include <chrono>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace klasp {

// A run that will not start: bad schema, a refused resume. Carries the exit code.
// Thrown after every participant has agreed, so nothing is left waiting in a
// collective.
class RunRefused : public std::exception {
public:
    explicit RunRefused(int c) : code(c) {}
    const char* what() const noexcept override;

    int code;
};

// Parse space group spec: number, comma-list, or named preset.
// Presets: common5, common10, common25, cocrystal, fine10, fine25,
//          coarse10, coarse25, cocrystal_fine, cocrystal_coarse,
//          efficientz1, efficientz2.
// When a preset with per-SG targets is used, fills control.targets.
// Throws ConfigError, naming the spec, when it is none of those or when a number
// in it is not a space group.
std::vector<int> parse_sg_spec(const std::string& spec,
                               std::map<int, int>* targets = nullptr);

// Checks the configuration and resolves a named model to its files. Throws
// ConfigError listing every problem, including a run with no inputs or no space
// groups.
RunConfig prepare_run_config(const RunConfig& config);

// How a run behaves towards its host process. Nothing here changes an energy or
// belongs in the resume fingerprint.
struct CspRunOptions {
    // When the process began; a wall-clock budget is charged from here.
    // Default-constructed means "start counting now".
    std::chrono::steady_clock::time_point process_start{};

    // Turn SIGTERM and SIGINT into "finish the unit in hand, then write out".
    // Off by default because a library should not replace its host's handlers;
    // embedders can call request_run_stop() instead.
    bool install_signal_handlers = false;
};

// The run's stop flag. Set by the signal handler or directly by an embedder;
// the work loop acts on it at the next unit boundary.
void request_run_stop();
bool run_stop_requested();
// Clears the stop flag. Called at the start of every run.
void clear_run_stop();

// Everything a run carries from one phase to the next. Built by the caller with
// the three references it cannot supply itself, then filled in by the phases.
struct CspRun {
    // SGWorker holds a mutex and is not movable, so the workers are sized here.
    CspRun(const RunConfig& cfg, Comm& c, ResultQueue& q, ResultSink& s)
        : config(cfg), comm(c), queue(q), sink(s),
          n_sgs(static_cast<int>(cfg.control.space_groups.size())),
          workers(cfg.control.space_groups.size()) {}

    CspRun(const CspRun&) = delete;
    CspRun& operator=(const CspRun&) = delete;

    const RunConfig& config;
    // output.max_vacuum resolved against the basis cutoff; see resolve_max_vacuum.
    double max_vacuum = 0.0;
    Comm& comm;
    ResultQueue& queue;
    ResultSink& sink;
    CspRunOptions options;

    // Set by the caller between csp_plan_work (which decides the targets) and
    // csp_attach (which hands the workers their seed cursor).
    WorkSource* work = nullptr;
    // Set by the caller before csp_loop; never null by shutdown.
    ResultDrain* drain = nullptr;

    int rank = 0;
    int nranks = 1;

    // --- csp_setup ---
    Workspace ws;
    BasisConfig basis;
    int effective_max_rank = 0;
    BatchOptimizeSettings bsettings;
    int n_sgs = 0;
    PerfRecorder perf;
    std::vector<int> per_sg_batch;
#ifdef KLASP_HAS_PET
    std::unique_ptr<PetEvaluator> pet_eval;
#endif
    // Isolated-molecule energy (kJ/mol/fu); a stage-3 lattice energy is the
    // crystal minus this.
    double pet_reference = 0.0;
    DedupConfig dcfg;
    std::vector<SGWorker> workers;

    // --- csp_plan_work ---
    std::vector<long long> targets;
    std::vector<long long> seed_start;
    std::vector<int> unit_sizes;
    int primary_stage = 0;
    std::string cfg_json;

    // --- csp_start_consumer (rank 0 only) ---
    std::unique_ptr<ResultConsumer> consumer;

    // --- csp_loop ---
    std::vector<int> local_rej;
    std::vector<int> last_sent_rej;
    // Longest work unit this participant has run; shutdown's patience for a
    // missing participant is measured in these.
    double unit_seconds_max = 0;
};

void csp_setup(CspRun& run);
void csp_plan_work(CspRun& run);
void csp_attach(CspRun& run);
void csp_start_consumer(CspRun& run);
void csp_loop(CspRun& run);
void csp_shutdown(CspRun& run);

// Result of run_local_csp.
struct CspRunResult {
    // Structures stored at the primary stage, summed over space groups.
    long long stored = 0;
    // Of those, the ones the dedup index called representatives.
    long long unique = 0;
    // True where the run stopped early: a time budget, a signal, or a lost
    // participant.
    bool stopped_early = false;
};

// Run a whole CSP in this process with no MPI: a WorkLedger replaces the RMA
// window and a QueueSink the MPI sends.
CspRunResult run_local_csp(const RunConfig& config, CspRunOptions options = {});

}  // namespace klasp
