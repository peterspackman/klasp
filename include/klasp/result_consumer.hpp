#pragma once

#include "klasp/dedup.hpp"
#include "klasp/database.hpp"
#include "klasp/io.hpp"
#include "klasp/status.hpp"
#include "klasp/result_queue.hpp"

#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <string>

namespace klasp {

// Rank 0 consumer: receives results from all ranks, deduplicates,
// writes to single database, updates status. Runs on its own thread.
class ResultConsumer {
public:
    // `primary_stage` is the stage whose rows map one-to-one onto generated
    // structures (2 with multipoles, else 1); only it counts toward an SG's target.
    ResultConsumer(const std::string& output_dir, int nranks,
                   const std::vector<int>& space_groups,
                   const std::vector<int>& targets,
                   ResultQueue& queue, int primary_stage,
                   bool resume = false);
    ~ResultConsumer();

    // Show the stage-3 columns in the status table.
    void show_stage3(bool on) { status_.show_stage3(on); }

    // Rebuild the dedup index from structures already in the database. Call
    // between configure_dedup() and start(). Returns representatives loaded.
    int resume_dedup_index();

    // Basis used to expand rigid-body results for the vacuum-gap check. Without
    // it the gap is not measured.
    void set_basis(const BasisConfig* basis) { basis_ = basis; }

    // Count of structures workers dropped for a vacuum slab, for the summary.
    void add_vacuum_dropped(long long n) { n_dropped_vacuum_ += n; }

    // Report how many representatives have no stage-3 row at the end of the run.
    void report_stage3_gap(bool v) { report_stage3_gap_ = v; }

    // Set database metadata (call before start).
    void set_meta(const std::string& key, const std::string& value) { db_.set_meta(key, value); }

    // Configure online deduplication (call before start). See DedupConfig.
    void configure_dedup(const DedupConfig& cfg) { dedup_.configure(cfg); }

    void start();

    void stop();

    // All valid structures (unique + duplicates) for an SG; thread-safe.
    int total_count(int sg) const;

    // Unique count for an SG at the primary stage.
    int unique_count(int sg) const;

    // Basins among this SG's stage-3 rows. 0 without an ML stage.
    int stage3_unique_count(int sg) const;

    // Record rejected seeds for status display (thread-safe, called from main thread).
    void record_rejected(int sg, int count) { status_.record_rejected(sg, count); }
    void record_pet_dropped(int sg, int count) { status_.record_pet_dropped(sg, count); }
    void record_pet_skipped(int sg, int count) { status_.record_pet_skipped(sg, count); }
    void set_work_unit(int structures_per_rank) { status_.set_work_unit(structures_per_rank); }
    void set_winding_down(int ranks_stopped, int nranks) {
        status_.set_winding_down(ranks_stopped, nranks);
    }
    void set_pet_stage(bool v) { status_.set_pet_stage(v); }

    // Discovery rate for an SG: fraction of recent structures that were unique.
    double discovery_rate(int sg) const { return status_.discovery_rate(sg); }

    void finalize();

private:
    void run();
    void process(const ResultMsg& m);
    // Drain what is queued, reporting progress.
    void drain_remaining();

    // Position of `sg` in space_groups_, or -1.
    int sg_index(int sg) const {
        for (int i = 0; i < n_sgs_; ++i)
            if (space_groups_[i] == sg) return i;
        return -1;
    }

    ResultQueue& queue_;   // where results land until the consumer stores them
    int n_sgs_;
    std::vector<int> space_groups_;
    std::vector<int> total_counts_;   // per-SG total valid (unique + dup)
    std::vector<int> unique_counts_;  // per-SG unique only, at the primary stage
    // Per-SG basins among stage-3 rows (dedup bins by stage).
    std::vector<int> s3_unique_counts_;

    StructureDedup dedup_;
    CSPDatabase db_;
    StatusReporter status_;
    int primary_stage_;
    const BasisConfig* basis_ = nullptr;
    bool report_stage3_gap_ = false;
    long long n_dropped_vacuum_ = 0;
    long long n_db_errors_ = 0;
    std::chrono::steady_clock::time_point last_checkpoint_{};
    std::string db_path_;
    bool resume_;

    std::thread thread_;
    std::atomic<bool> shutdown_{false};
};

// Fold one rejection delta (kRejFields per-SG counters) into the consumer.
inline void record_rejection_delta(ResultConsumer& consumer,
                                   const std::vector<int>& space_groups,
                                   const std::vector<int>& delta) {
    const int n_sgs = static_cast<int>(space_groups.size());
    for (int i = 0; i < n_sgs; ++i) {
        if (delta[i] > 0)
            consumer.record_rejected(space_groups[i], delta[i]);
        if (delta[n_sgs + i] > 0)
            consumer.record_pet_dropped(space_groups[i], delta[n_sgs + i]);
        if (delta[2 * n_sgs + i] > 0)
            consumer.record_pet_skipped(space_groups[i], delta[2 * n_sgs + i]);
        if (delta[3 * n_sgs + i] > 0)
            consumer.add_vacuum_dropped(delta[3 * n_sgs + i]);
    }
}

}  // namespace klasp
