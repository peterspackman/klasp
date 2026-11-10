#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <array>

namespace klasp {

// Consumer-driven status reporter for rank 0, called inline from the consumer
// loop. The per-SG table is described once as COLUMNS in status.cpp, shared by
// the status file, final summary and non-TTY progress view.
class StatusReporter {
public:
    // Per-SG tracking data. Public because the column formatters are free
    // functions over it.
    struct SGStats {
        int space_group = 0;
        int target = 0;
        int total = 0;          // valid structures (unique + dup)
        int unique = 0;
        int duplicates = 0;
        int invalid = 0;        // non-finite energy after optimization
        int rejected = 0;       // failed collision resolution during generation
        double time_ms = 0;     // cumulative processing time

        // Optimiser diagnostics, unpopulated under --no-store-diagnostics.
        int n_diag = 0;         // structures that carried diagnostics
        int n_converged = 0;    // of those, reached the gradient tolerance
        long long iters_sum = 0;  // sum of iters_to_converge over converged ones

        // Final gradients as a log-spaced histogram, for a median that one blown-up
        // structure cannot drag. Resolution is a fifth of a decade.
        static constexpr int GRAD_BINS = 100;
        static constexpr double GRAD_LOG_LO = -12.0;   // log10 of the lowest edge
        static constexpr double GRAD_PER_DECADE = 5.0;
        std::array<int, GRAD_BINS> grad_hist{};

        int n_s1 = 0, n_s1_converged = 0;   // stage-1 (charge-only) convergence
        long long s1_iters_sum = 0;         // sum of stage-1 iters over converged ones

        // Stage-3 (PET / ML) relaxation, from the stage-3 record. Not counted in
        // the per-SG totals, which stage 2 already counted.
        int n_s3 = 0, n_s3_converged = 0;
        // Basins among the stage-3 rows; differs from the stage-2 `unique` count.
        int s3_unique = 0;
        // Candidates the ML stage never emitted a row for (clash screen, or a
        // non-finite energy).
        int n_s3_dropped = 0;
        // Candidates the ML stage skipped because another structure in the same
        // basin was already relaxed on this rank. Drops and skips both count
        // toward the stage being finished.
        int n_s3_skipped = 0;
        long long s3_iters_sum = 0;         // sum of stage-3 iters over converged ones
        double s3_grad_sum = 0;             // sum of final max force (Ha/Bohr)
        double s3_time_ms = 0;              // cumulative stage-3 wall time

        // Energy range over the space group's stored structures, kJ/mol. A
        // collapsed spread means the vacuum gap has passed the potential cutoff.
        double e_lo = 0, e_hi = 0;
        int n_energy = 0;

        double energy_spread() const {
            return n_energy > 0 ? e_hi - e_lo : -1.0;
        }

        void add_energy(double e) {
            if (!(e > -1e300 && e < 1e300)) return;    // NaN and the sentinels
            if (n_energy == 0) { e_lo = e_hi = e; }
            else if (e < e_lo) e_lo = e;
            else if (e > e_hi) e_hi = e;
            ++n_energy;
        }

        // Discovery rate: new_uniques / total_processed over a sliding window
        int window_total = 0;   // total at last window boundary
        int window_unique = 0;  // unique at last window boundary
        double discovery_rate = 1.0;  // current λ (0-1)

        // Force-field cost per structure; excludes the ML stage.
        double ms_per_struct() const { return total > 0 ? time_ms / total : 0; }
        // End-to-end cost per structure, used for the ETA. s3_time_ms is divided
        // by `total`, not n_s3, because the stage-3 gate relaxes only some
        // structures.
        double total_ms_per_struct() const {
            return total > 0 ? (time_ms + s3_time_ms) / total : 0;
        }
        double converged_frac() const {
            return n_diag > 0 ? double(n_converged) / n_diag : -1.0;
        }
        double mean_iters() const {
            return n_converged > 0 ? double(iters_sum) / n_converged : -1.0;
        }
        void add_grad(double g) {
            if (!(g > 0)) return;      // a log axis has nowhere to put 0
            const double x = (std::log10(g) - GRAD_LOG_LO) * GRAD_PER_DECADE;
            int b = static_cast<int>(std::floor(x));
            // Clamped, not dropped, so outliers still count toward the median.
            b = std::min(std::max(b, 0), GRAD_BINS - 1);
            ++grad_hist[b];
        }

        // Median final gradient, at the centre of the bin the halfway point
        // falls in. -1 when nothing was recorded.
        double median_grad() const {
            int n = 0;
            for (int c : grad_hist) n += c;
            if (n == 0) return -1.0;
            int cum = 0;
            for (int b = 0; b < GRAD_BINS; ++b) {
                cum += grad_hist[b];
                if (2 * cum > n)
                    return std::pow(10.0, GRAD_LOG_LO + (b + 0.5) / GRAD_PER_DECADE);
            }
            return -1.0;
        }
        double s1_converged_frac() const {
            return n_s1 > 0 ? double(n_s1_converged) / n_s1 : -1.0;
        }
        double s3_converged_frac() const {
            return n_s3 > 0 ? double(n_s3_converged) / n_s3 : -1.0;
        }
        // Mean stage-3 iterations over converged structures; compare with
        // --pet-iterations.
        double s3_mean_iters() const {
            return n_s3_converged > 0 ? double(s3_iters_sum) / n_s3_converged : -1.0;
        }
        double s3_mean_grad() const { return n_s3 > 0 ? s3_grad_sum / n_s3 : -1.0; }
        double s3_ms_per_struct() const { return n_s3 > 0 ? s3_time_ms / n_s3 : -1.0; }

        // Mean stage-1 iterations to converge; compare with --iterations-stage1.
        double s1_mean_iters() const {
            return n_s1_converged > 0 ? double(s1_iters_sum) / n_s1_converged : -1.0;
        }

        // P(at least 1 new unique in next batch_size structures)
        double batch_discovery_prob(int batch_size) const {
            if (discovery_rate >= 1.0) return 1.0;
            return 1.0 - std::pow(1.0 - discovery_rate, batch_size);
        }
    };

    // Everything derived from the whole run, computed once per emit.
    struct Snapshot {
        double elapsed_s = 0;
        int n_active = 1;            // SGs still short of target
        int nranks = 1;
        double avg_mps = 0;          // grand mean ms/structure, ETA fallback
        double overall_eta_s = 0;
        long long remaining_structs = 0;
        int total = 0, unique = 0, target = 0, rejected = 0;
        // Whether an ML stage runs, so a space group still receiving stage-3 rows
        // is not reported as finished.
        bool pet_stage = false;
        bool final = false;   // the run has drained; nothing can still be in flight
    };

    StatusReporter(const std::string& output_dir, int nranks,
                   const std::vector<int>& space_groups,
                   const std::vector<int>& targets);

    // Record a new valid (unique) structure for a SG, with processing time in ms.
    void record(int sg, double time_ms = 0);

    // Record a duplicate structure, with processing time in ms.
    void record_duplicate(int sg, double time_ms = 0);

    // Record one structure's energy, in kJ/mol, for the per-space-group spread.
    void record_energy(int sg, double energy);

    // Record one structure's optimiser diagnostics. grad_max < 0 is the
    // "not recorded" sentinel (--no-store-diagnostics) and is ignored;
    // iters < 0 means it never reached the gradient tolerance.
    void record_diagnostics(int sg, int iters_to_converge, double grad_max);

    // Stage-1 convergence for the same structure, carried on its stage-2 record.
    void record_stage1(int sg, int iters_to_converge, double grad_max);

    // Stage-3 (PET) convergence and cost, from the stage-3 record. `max_force`
    // < 0 is the not-recorded sentinel; iters < 0 means it never met the force
    // and stress tolerances.
    void record_stage3(int sg, int iters_to_converge, double max_force, double time_ms);

    // Show the stage-3 columns. Off by default.
    void show_stage3(bool on) { show_stage3_ = on; }

    // Record an invalid structure (non-finite energy).
    void record_invalid(int sg);

    // Record rejected seeds (failed collision resolution during generation).
    void record_rejected(int sg, int count);

    // Candidates the ML stage produced no row for, so the ETA can tell when the
    // stage is finished.
    void record_pet_dropped(int sg, int count);

    // Candidates the ML stage skipped as duplicates of one already relaxed.
    void record_pet_skipped(int sg, int count);

    // Set total valid count (unique + duplicates) for a SG.
    void set_total(int sg, int count);

    // Set unique count for a SG.
    void set_unique(int sg, int count);

    // Set the stage-3 (post-ML) unique count for a SG.
    void set_stage3_unique(int sg, int count);

    // Seed the count of stage-3 rows already on disk, on resume. Needed because
    // the finished test is n_s3 + dropped + skipped >= total, and total is
    // seeded from the database.
    void seed_stage3(int sg, int n_rows);

    // Seed the discovery-rate sliding window from restored counts on resume.
    void seed_window(int sg, int total, int unique);

    double discovery_rate(int sg) const {
        int i = find_sg(sg);
        return (i >= 0) ? sg_[i].discovery_rate : 1.0;
    }

    // Check if it's time to update display (call periodically from consumer loop).
    void maybe_update();

    void finalize();

    // Structures a rank processes before it reports anything; results arrive at
    // work-unit granularity.
    void set_work_unit(int structures_per_rank) { work_unit_ = structures_per_rank; }

    // Tell the reporter an ML stage is configured, so stage-2 completion is not
    // reported as done.
    void set_pet_stage(bool v) { pet_stage_ = v; }

    // Results received but not yet stored. Climbs without bound when rank 0's
    // consumer is falling behind.
    void set_backlog(std::size_t n) { backlog_ = n; }

    // How many ranks have stopped claiming work, and out of how many. Once
    // non-zero the ETA is meaningless.
    void set_winding_down(int ranks_stopped, int nranks) {
        winding_down_ = ranks_stopped;
        winding_down_of_ = nranks;
    }


private:
    std::string winding_down_line(bool compact = false) const;
    void write_status_file();
    void print_progress();
    // Refresh the measured global throughput (structures/sec) used for the
    // overall ETA. Sampled on a coarse interval and EMA-smoothed.
    void update_rate(std::chrono::steady_clock::time_point now);

    Snapshot snapshot() const;
    // Sum of every SG, for the table's total row.
    SGStats totals() const;

    static constexpr int DISCOVERY_WINDOW = 100;

    std::string output_dir_;
    int nranks_;
    int work_unit_ = 0;         // structures per rank per reporting unit (0 = unknown)
    std::size_t backlog_ = 0;
    int winding_down_ = 0;      // ranks that have stopped claiming
    int winding_down_of_ = 0;
    bool pet_stage_ = false;
    bool final_ = false;
    bool show_stage3_ = false;   // emit the stage-3 columns (see show_stage3)
    std::vector<SGStats> sg_;

    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_;
    std::chrono::steady_clock::time_point last_record_;
    bool is_tty_;
    int lines_printed_ = 0;
    double update_interval_ = 0.1;

    // Measured global throughput, smoothed over a long horizon.
    std::chrono::steady_clock::time_point rate_last_time_;
    long long rate_last_total_ = 0;
    double measured_rate_ = 0.0;   // EMA of global structures/sec

    // Skip re-emitting an identical status; a heartbeat still emits occasionally.
    long long last_emit_sig_ = -1;
    std::chrono::steady_clock::time_point last_emit_;
    // TTY-only: repaint the bar a few times a second even when counts are unchanged.
    std::chrono::steady_clock::time_point last_anim_;

    int find_sg(int sg) const {
        for (size_t i = 0; i < sg_.size(); ++i)
            if (sg_[i].space_group == sg) return static_cast<int>(i);
        return -1;
    }
};

}  // namespace klasp
