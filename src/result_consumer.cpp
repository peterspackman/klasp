#include "klasp/log.hpp"
#include "klasp/result_consumer.hpp"
#include "klasp/constants.hpp"
#include "klasp/cell.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/warnings.hpp"
#include "klasp/vacuum.hpp"
#include "klasp/run_state.hpp"
#include <filesystem>
#include <fmt/base.h>
#include <atomic>
#include <chrono>
#include <cstdlib>

namespace klasp {


ResultConsumer::ResultConsumer(const std::string& output_dir, int nranks,
                               const std::vector<int>& space_groups,
                               const std::vector<int>& targets,
                               ResultQueue& queue, int primary_stage,
                               bool resume)
    : queue_(queue),
      n_sgs_(static_cast<int>(space_groups.size())),
      space_groups_(space_groups),
      total_counts_(space_groups.size(), 0),
      unique_counts_(space_groups.size(), 0),
      s3_unique_counts_(space_groups.size(), 0),
      status_(output_dir, nranks, space_groups, targets),
      primary_stage_(primary_stage),
      db_path_(output_dir + "/csp.db"),
      resume_(resume) {

    std::filesystem::create_directories(output_dir);
    const std::string& db_path = db_path_;
    db_.open(db_path);
    db_.set_meta("nranks", std::to_string(nranks));

    // Resume from existing DB if present. The unique count comes from unique
    // rows, not count_at_stage (which counts every row).
    int total_resumed = 0;
    for (int i = 0; i < n_sgs_; ++i) {
        total_counts_[i]  = db_.count_at_stage(space_groups_[i], primary_stage_);
        unique_counts_[i] = db_.count_unique_at_stage(space_groups_[i], primary_stage_);
        // Stage 3 keeps its own basin count across a resume.
        s3_unique_counts_[i] = db_.count_unique_at_stage(space_groups_[i], STAGE_FINAL);
        status_.set_total(space_groups_[i], total_counts_[i]);
        status_.set_unique(space_groups_[i], unique_counts_[i]);
        status_.set_stage3_unique(space_groups_[i], s3_unique_counts_[i]);
        // Rows, not uniques: the finished test counts rows against `total`.
        status_.seed_stage3(space_groups_[i],
                            db_.count_at_stage(space_groups_[i], STAGE_FINAL));
        // Seed the discovery-rate window so the first delta has a baseline.
        status_.seed_window(space_groups_[i], total_counts_[i], unique_counts_[i]);
        total_resumed += total_counts_[i];
    }
    if (total_resumed > 0) {
        logging::info("\n{}: {} already holds {} structures.\n",
                   resume_ ? "Resuming" : "WARNING", db_path, total_resumed);
        for (int i = 0; i < n_sgs_; ++i)
            if (total_counts_[i] > 0)
                logging::info("  SG {:>3d}: {} structures, {} unique\n",
                           space_groups_[i], total_counts_[i], unique_counts_[i]);
        if (resume_) {
            logging::info(
                "\nThe generator continues past the highest seed already used, each\n"
                "target is topped up rather than run again, and the stored\n"
                "representatives are loaded into the dedup index so new structures\n"
                "are compared against the whole landscape.\n\n");
        } else {
            logging::info(
                "\nThis is NOT a resume. New structures are deduplicated only against\n"
                "each other -- the existing ones are not loaded into the index -- and the\n"
                "target is a full target's worth again regardless of what is here. Seeds\n"
                "regenerate identically for the same input, so the rows above will be\n"
                "OVERWRITTEN. Pass --resume to continue this run instead, or move the\n"
                "database aside.\n\n");
        }
        std::fflush(stdout);
    }
}

int ResultConsumer::resume_dedup_index() {
    if (!resume_) return 0;
    // Only representatives at the primary stage: a live run never stores
    // duplicates in the index either.
    LandscapeFilter lf;
    lf.unique_only = true;
    const DedupConfig cfg = dedup_.config();
    lf.want_amd = cfg.amd_tol > 0;
    lf.want_powder = cfg.use_powder;
    lf.want_pdd = cfg.pdd_tol > 0;

    // Commit first: load_landscape opens its own connection.
    db_.flush();

    // Every stage that carries descriptors (classify() bins by stage). Stages
    // below the primary one hold no descriptors.
    std::vector<int> stages;
    for (const auto& [stage, rows] : stage_histogram({db_path_}))
        if (stage >= primary_stage_ && rows > 0) stages.push_back(stage);
    if (stages.empty()) stages.push_back(primary_stage_);

    // Streamed to avoid holding the whole landscape alongside the index; order
    // is ascending energy with the same tie-break, which seeding depends on.
    int n = 0, skipped = 0;
    for (int stage : stages) {
        lf.stage = stage;
        for_each_landscape_row(db_path_, lf, [&](const LandscapeRow& row) {
            // A row missing a descriptor a stage needs was never comparable, so it
            // must not become a representative; a missing PDD would make matches()
            // apply a looser rule than the live run.
            if ((lf.want_amd && !row.has_amd) ||
                (lf.want_powder && !row.has_powder) ||
                (lf.want_pdd && row.pdd.empty())) { ++skipped; return; }
            dedup_.seed_existing(row.r, row.pdd.empty() ? nullptr : row.pdd.data(),
                                 static_cast<int>(row.pdd.size()));
            ++n;
        });
    }
    if (n > 0 || skipped > 0) {
        logging::info("  dedup index seeded with {} existing representative{} across stage{} ",
                   n, n == 1 ? "" : "s", stages.size() == 1 ? "" : "s");
        for (std::size_t i = 0; i < stages.size(); ++i)
            logging::info("{}{}", i ? ", " : "", stages[i]);
        if (skipped > 0)
            logging::info(" ({} skipped: descriptor missing)", skipped);
        logging::info("\n\n");
        std::fflush(stdout);
    }
    return n;
}

ResultConsumer::~ResultConsumer() { stop(); }

void ResultConsumer::start() {
    shutdown_ = false;
    thread_ = std::thread(&ResultConsumer::run, this);
}

void ResultConsumer::stop() {
    // Release producers parked on the queue ceiling before joining, so the join
    // cannot deadlock.
    queue_.release_producers();
    shutdown_ = true;
    if (thread_.joinable()) thread_.join();
    drain_remaining();
    db_.flush();
}

// Drain what is left in the queue, reporting progress (the status thread has
// already been joined).
void ResultConsumer::drain_remaining() {
    using Clock = std::chrono::steady_clock;
    // Small backlogs drain in well under a second and do not need announcing.
    constexpr std::size_t kWorthReporting = 2000;
    const std::size_t start = queue_.queue_size_hint();
    const bool report = start >= kWorthReporting;
    if (report) {
        logging::info("\n  Storing the {} result{} still queued. Every other rank is "
                   "waiting at the exit until this finishes.\n",
                   start, start == 1 ? "" : "s");
        std::fflush(stdout);
    }

    auto last = Clock::now();
    long long n = 0;
    ResultMsg m;
    while (queue_.try_pop(m)) {
        try {
            process(m);
        } catch (const std::exception&) { ++n_db_errors_; }
        ++n;
        if (!report) continue;
        const auto now = Clock::now();
        if (now - last >= std::chrono::seconds(5)) {
            last = now;
            logging::info("    stored {} of {}, {} still queued\n",
                       n, start, queue_.queue_size_hint());
            std::fflush(stdout);
        }
    }
    if (report) {
        logging::info("    stored {}; the queue is empty.\n", n);
        std::fflush(stdout);
    }
}

int ResultConsumer::total_count(int sg) const {
    const int i = sg_index(sg);
    return i >= 0 ? total_counts_[i] : 0;
}

int ResultConsumer::unique_count(int sg) const {
    const int i = sg_index(sg);
    return i >= 0 ? unique_counts_[i] : 0;
}

int ResultConsumer::stage3_unique_count(int sg) const {
    const int i = sg_index(sg);
    return i >= 0 ? s3_unique_counts_[i] : 0;
}

void ResultConsumer::finalize() {
    // Dropped on the worker before being reported as produced, so the run still
    // reaches its target.
    if (n_db_errors_ > 0)
        logging::info("\n  WARNING: {} structure{} could not be stored. The landscape is "
                   "incomplete.\n",
                   n_db_errors_, n_db_errors_ == 1 ? "" : "s");
    // Rank 0 setting the pace is worth knowing before blaming slow workers.
    if (const long long throttled = queue_.throttled_pushes(); throttled > 0)
        logging::info("\n  Rank 0's result queue was full {} time{}; the ranks feeding it "
                   "waited.\n",
                   throttled, throttled == 1 ? "" : "s");
    if (n_dropped_vacuum_ > 0)
        logging::info("\n  --max-vacuum dropped {} structure{} holding a slab of vacuum; "
                   "their seeds were reissued.\n",
                   n_dropped_vacuum_, n_dropped_vacuum_ == 1 ? "" : "s");
    // Representatives the ML stage never reached: the gate keeps the first member
    // of a basin per rank while the index keeps the deepest, so they can differ.
    if (report_stage3_gap_) {
        db_.flush();
        const int gap = db_.count_unique_missing_stage(primary_stage_, STAGE_FINAL);
        if (gap > 0)
            logging::info(
                "\n  {} unique structure{} carr{} no stage-3 energy: the ML stage relaxed "
                "another member of {} basin.\n"
                "  Fill them in with:\n"
                "    klasp rerank {} --top-up --target 0\n",
                gap, gap == 1 ? "" : "s", gap == 1 ? "ies" : "y",
                gap == 1 ? "its" : "their", db_path_);
    }
    // Checkpoint the WAL before closing so the database file is self-contained.
    db_.checkpoint(/*truncate=*/true);
    db_.close();
    status_.finalize();
}

void ResultConsumer::process(const ResultMsg& m) {
    StructureResult r = m.sr;
    const double* pdd = m.pdd.empty() ? nullptr : m.pdd.data();
    const int pdd_len = static_cast<int>(m.pdd.size());

    // The vacuum gap arrives measured by SGWorker::pack_results; stage-3 rows are
    // measured from their explicit coordinates.
    if (m.flex.n_atoms > 0) {
        r.vacuum_gap = max_vacuum_gap(m.flex.cell_params, m.flex.frac.data(),
                                      m.flex.n_atoms).gap;
        if (r.vacuum_gap > WARN_VACUUM_GAP_A) r.warnings |= WARN_VACUUM_SLAB;
    }

    const auto verdict = dedup_.classify(r, pdd, pdd_len);
    const bool unique = verdict.is_unique();
    db_.insert_result(&r, unique, verdict.duplicate_of, verdict.duplicate_of_sg);
    // Only stages at or above the primary one carry real descriptors; lower
    // stages are zero-filled and must not be stored as descriptors.
    if (r.stage >= primary_stage_) {
        std::vector<double> amd_vec(r.amd, r.amd + AMD_K);
        db_.insert_amd(r.seed, r.space_group, r.stage, amd_vec);
        db_.insert_powder(r.seed, r.space_group, r.stage, r.powder, POWDER_N_BINS);
        const DedupConfig dcfg = dedup_.config();
        if (pdd && dcfg.pdd_rows > 0 && dcfg.pdd_k > 0)
            db_.insert_pdd(r.seed, r.space_group, r.stage, pdd,
                           dcfg.pdd_rows, dcfg.pdd_k);
    }
    // Explicit non-rigid geometry (PET stage 3): the rigid-body DOFs on the
    // structure row can't represent it, so it gets its own keyed row.
    if (m.flex.n_atoms > 0)
        db_.insert_flexible(r.seed, r.space_group, r.stage, m.flex);
    // r displaced the representative of an existing basin: demote that row,
    // keyed on its own space group (cross_sg clusters span groups).
    if (verdict.displaced())
        db_.demote_to_duplicate(verdict.demoted_seed, verdict.demoted_sg,
                                verdict.demoted_stage, r.seed, r.space_group);
    const int sgi = sg_index(r.space_group);
    // Stage 0 / stage 3 rows re-describe a structure the primary row already
    // counted, so they must not move the per-SG totals. Stage 3 records its own
    // convergence, cost and basins.
    if (r.stage == STAGE_FINAL && sgi >= 0) {
        status_.record_stage3(r.space_group, r.iters_to_converge, r.grad_max, r.time_ms);
        // A displacement swaps a basin's representative without creating one.
        if (unique && !verdict.displaced()) ++s3_unique_counts_[sgi];
        status_.set_stage3_unique(r.space_group, s3_unique_counts_[sgi]);
    }
    if (r.stage == primary_stage_ && sgi >= 0) {
        ++total_counts_[sgi];
        status_.record_energy(r.space_group, r.energy);
        status_.record_diagnostics(r.space_group, r.iters_to_converge, r.grad_max);
        status_.record_stage1(r.space_group, r.s1_iters_to_converge, r.s1_grad_max);
        if (unique) {
            status_.record(r.space_group, r.time_ms);
            // A swap keeps the basin count; only a new basin bumps it.
            if (!verdict.displaced()) ++unique_counts_[sgi];
        } else {
            status_.record_duplicate(r.space_group, r.time_ms);
        }
        status_.set_total(r.space_group, total_counts_[sgi]);
        status_.set_unique(r.space_group, unique_counts_[sgi]);
    }
}

void ResultConsumer::run() {
    // An exception escaping this thread terminates every rank via MPI_Abort, so
    // database failures are absorbed and counted; a full disk is reported once.
    while (!shutdown_) {
        bool did_work = false;

        // Bounded so a backlog cannot starve the status update below.
        ResultMsg m;
        int drained = 0;
        while (drained < 512 && queue_.try_pop(m)) {
            try {
                process(m);
            } catch (const std::exception& e) {
                if (n_db_errors_++ == 0)
                    logging::warn("\nWARNING: storing a structure failed: {}\n"
                        "The run continues; further failures are counted, not "
                        "reported.\n", e.what());
            }
            did_work = true;
            ++drained;
        }
        try {
            db_.flush();
        } catch (const std::exception&) { ++n_db_errors_; }
        status_.set_backlog(queue_.queue_size_hint());
        status_.maybe_update();
        // Passive WAL checkpoint, so a killed run's results are in csp.db itself.
        const auto now = std::chrono::steady_clock::now();
        if (now - last_checkpoint_ > std::chrono::seconds(60)) {
            db_.checkpoint(/*truncate=*/false);
            last_checkpoint_ = now;
        }

        if (!did_work)
            std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    drain_remaining();
    db_.flush();
}

}  // namespace klasp
