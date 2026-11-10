#include "klasp/log.hpp"
#include "klasp/status.hpp"
#include "klasp/version.hpp"

#include <fmt/format.h>
#include <fmt/chrono.h>

#include <ctime>
#include <algorithm>
#include <numeric>
#include <cstdlib>

#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace klasp {

using SGStats = StatusReporter::SGStats;
using Snapshot = StatusReporter::Snapshot;

// Thread-safe local time; fmt has no localtime wrapper.
static std::tm local_tm(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

static bool should_use_tty() {
    // --plain, or the deprecated KLASP_PLAIN=1: non-ANSI output for a pty that
    // forwards to a log.
    if (logging::plain()) return false;
    if (const char* p = std::getenv("KLASP_PLAIN"))
        if (p[0] != '\0' && p[0] != '0') return false;
    return isatty(fileno(stdout));
}

static std::string format_duration(double secs) {
    if (secs < 0 || !std::isfinite(secs)) return "---";
    int s = static_cast<int>(secs);
    if (s < 60) return fmt::format("{}s", s);
    if (s < 3600) return fmt::format("{}m{}s", s / 60, s % 60);
    return fmt::format("{}h{}m", s / 3600, (s % 3600) / 60);
}

// Discovery rate as "N%", or "---" before any structure is seen.
static std::string format_disc(double rate, int total) {
    if (total <= 0) return "---";
    return fmt::format("{:.0f}%", rate * 100.0);
}

// Seconds of remaining work for one SG, at its own measured rate.
static double sg_eta_s(const SGStats& s, const Snapshot& sn) {
    const double mps = s.total_ms_per_struct();
    const int remaining = s.target - s.total;
    if (remaining <= 0 || mps <= 0) return 0;
    return remaining * mps / 1000.0 * sn.n_active / std::max(1, sn.nranks);
}

static std::string format_sg_eta(const SGStats& s, const Snapshot& sn) {
    if (s.target - s.total <= 0) {
        // With --pet-relax the stage-3 rows lag the stage-2 rows, so report "pet"
        // until n_s3 + drops + skips reaches the structure count.
        if (sn.pet_stage && !sn.final
            && s.n_s3 + s.n_s3_dropped + s.n_s3_skipped < s.total) return "pet";
        return "done";
    }
    if (s.total_ms_per_struct() <= 0) return "---";
    return format_duration(sg_eta_s(s, sn));
}

// ---------------------------------------------------------------------------
// The table, described once; every renderer walks this list. `total` is false
// for columns whose sum is meaningless (a rate, an ETA); those print blank on
// the total row.
// ---------------------------------------------------------------------------
namespace {

struct Column {
    const char* header;
    int width;
    std::string (*cell)(const SGStats&, const Snapshot&);
    bool total = true;
    bool stage3 = false;      // only emitted when the run has a stage-3 pass
};

std::string c_total(const SGStats& s, const Snapshot&) { return fmt::format("{}", s.total); }
std::string c_unique(const SGStats& s, const Snapshot&) { return fmt::format("{}", s.unique); }
std::string c_target(const SGStats& s, const Snapshot&) { return fmt::format("{}", s.target); }
std::string c_rejected(const SGStats& s, const Snapshot&) { return fmt::format("{}", s.rejected); }
std::string c_disc(const SGStats& s, const Snapshot&) {
    return format_disc(s.discovery_rate, s.total);
}
std::string c_mps(const SGStats& s, const Snapshot&) {
    return fmt::format("{:.1f}", s.total_ms_per_struct());
}
std::string c_conv(const SGStats& s, const Snapshot&) {
    const double f = s.converged_frac();
    return f < 0 ? "---" : fmt::format("{:.1f}%", f * 100.0);
}
std::string c_iters(const SGStats& s, const Snapshot&) {
    const double m = s.mean_iters();
    return m < 0 ? "---" : fmt::format("{:.0f}", m);
}
std::string c_espread(const SGStats& s, const Snapshot&) {
    const double d = s.energy_spread();
    if (d < 0) return "---";
    return fmt::format("{:.2f}", d);
}
std::string c_grad(const SGStats& s, const Snapshot&) {
    const double g = s.median_grad();
    return g < 0 ? "---" : fmt::format("{:.1e}", g);
}
std::string c_s1conv(const SGStats& s, const Snapshot&) {
    const double f = s.s1_converged_frac();
    return f < 0 ? "---" : fmt::format("{:.1f}%", f * 100.0);
}
std::string c_s1iters(const SGStats& s, const Snapshot&) {
    const double m = s.s1_mean_iters();
    return m < 0 ? "---" : fmt::format("{:.0f}", m);
}
std::string c_s3uniq(const SGStats& s, const Snapshot&) {
    return fmt::format("{}", s.s3_unique);
}
std::string c_s3conv(const SGStats& s, const Snapshot&) {
    const double f = s.s3_converged_frac();
    return f < 0 ? "---" : fmt::format("{:.1f}%", f * 100.0);
}
std::string c_s3iters(const SGStats& s, const Snapshot&) {
    const double m = s.s3_mean_iters();
    return m < 0 ? "---" : fmt::format("{:.0f}", m);
}
std::string c_s3force(const SGStats& s, const Snapshot&) {
    const double g = s.s3_mean_grad();
    return g < 0 ? "---" : fmt::format("{:.1e}", g);
}
std::string c_s3ms(const SGStats& s, const Snapshot&) {
    const double m = s.s3_ms_per_struct();
    return m < 0 ? "---" : fmt::format("{:.0f}", m);
}
std::string c_eta(const SGStats& s, const Snapshot& sn) { return format_sg_eta(s, sn); }

// Column order is the display order. Read s1iters against --iterations-stage1
// and iters against --iterations. `unique` and `s3uniq` count basins at stage 2
// and stage 3 and need not agree. `ms/struct` is end-to-end cost per structure
// (amortised ML share included); `s3ms` is per ML relaxation.
constexpr Column COLUMNS[] = {
    {"total",     8, c_total},
    {"unique",    8, c_unique},
    {"target",    8, c_target},
    {"rejected",  8, c_rejected},
    {"disc",      5, c_disc,  false},
    {"dE",        7, c_espread},
    {"s1conv",    7, c_s1conv},
    {"s1iters",   8, c_s1iters},
    {"conv",      6, c_conv},
    {"iters",     5, c_iters},
    {"grad",      8, c_grad},
    {"s3uniq",    8, c_s3uniq,  true, true},
    {"s3conv",    7, c_s3conv,  true, true},
    {"s3iters",   8, c_s3iters, true, true},
    {"s3force",   8, c_s3force, true, true},
    {"s3ms",      7, c_s3ms,    true, true},
    {"ms/struct", 9, c_mps},
    {"ETA",       8, c_eta,   false},
};
constexpr int N_COLUMNS = sizeof(COLUMNS) / sizeof(COLUMNS[0]);

// These build strings because the table goes to status.txt always and to the
// console through the log sink, which --quiet can silence.
std::string header_line(bool show_s3) {
    std::string out = fmt::format("{:>4s}", "SG");
    for (int i = 0; i < N_COLUMNS; ++i) {
        if (COLUMNS[i].stage3 && !show_s3) continue;
        out += fmt::format("  {:>{}s}", COLUMNS[i].header, COLUMNS[i].width);
    }
    return out + "\n";
}

std::string row_line(const std::string& label, const SGStats& s,
                     const Snapshot& sn, bool show_s3, bool is_total = false) {
    std::string out = fmt::format("{:>4s}", label);
    for (int i = 0; i < N_COLUMNS; ++i) {
        if (COLUMNS[i].stage3 && !show_s3) continue;
        const std::string v = (is_total && !COLUMNS[i].total)
            ? std::string{} : COLUMNS[i].cell(s, sn);
        out += fmt::format("  {:>{}s}", v, COLUMNS[i].width);
    }
    return out + "\n";
}

void write_header(FILE* f, bool show_s3) { fmt::print(f, "{}", header_line(show_s3)); }

void write_row(FILE* f, const std::string& label, const SGStats& s,
               const Snapshot& sn, bool show_s3, bool is_total = false) {
    fmt::print(f, "{}", row_line(label, s, sn, show_s3, is_total));
}

}  // namespace

StatusReporter::StatusReporter(const std::string& output_dir, int nranks,
                               const std::vector<int>& space_groups,
                               const std::vector<int>& targets)
    : output_dir_(output_dir), nranks_(nranks),
      is_tty_(should_use_tty()) {
    sg_.resize(space_groups.size());
    for (size_t i = 0; i < space_groups.size(); ++i) {
        sg_[i].space_group = space_groups[i];
        sg_[i].target = targets[i];
    }
    start_time_ = std::chrono::steady_clock::now();
    last_update_ = start_time_;
    last_record_ = start_time_;
    rate_last_time_ = start_time_;
    last_emit_ = start_time_;
}

void StatusReporter::record(int sg, double time_ms) {
    last_record_ = std::chrono::steady_clock::now();
    int i = find_sg(sg);
    if (i >= 0) sg_[i].time_ms += time_ms;
}

void StatusReporter::record_duplicate(int sg, double time_ms) {
    int i = find_sg(sg);
    if (i >= 0) { ++sg_[i].duplicates; sg_[i].time_ms += time_ms; }
}

void StatusReporter::record_energy(int sg, double energy) {
    const int i = find_sg(sg);
    if (i < 0) return;
    sg_[i].add_energy(energy);
}

void StatusReporter::record_diagnostics(int sg, int iters_to_converge, double grad_max) {
    if (grad_max < 0) return;            // --no-store-diagnostics
    int i = find_sg(sg);
    if (i < 0) return;
    auto& s = sg_[i];
    ++s.n_diag;
    s.add_grad(grad_max);
    if (iters_to_converge >= 0) {
        ++s.n_converged;
        s.iters_sum += iters_to_converge;
    }
}

void StatusReporter::record_stage3(int sg, int iters_to_converge, double max_force,
                                   double time_ms) {
    if (max_force < 0) return;           // stage 3 did not run for this structure
    int i = find_sg(sg);
    if (i < 0) return;
    auto& s = sg_[i];
    ++s.n_s3;
    s.s3_grad_sum += max_force;
    s.s3_time_ms += time_ms;
    if (iters_to_converge >= 0) {
        ++s.n_s3_converged;
        s.s3_iters_sum += iters_to_converge;
    }
}

void StatusReporter::record_stage1(int sg, int iters_to_converge, double grad_max) {
    if (grad_max < 0) return;
    int i = find_sg(sg);
    if (i < 0) return;
    ++sg_[i].n_s1;
    if (iters_to_converge >= 0) {
        ++sg_[i].n_s1_converged;
        sg_[i].s1_iters_sum += iters_to_converge;
    }
}

void StatusReporter::record_invalid(int sg) {
    int i = find_sg(sg);
    if (i >= 0) ++sg_[i].invalid;
}

void StatusReporter::record_pet_dropped(int sg, int count) {
    const int i = find_sg(sg);
    if (i >= 0) sg_[i].n_s3_dropped += count;
}

void StatusReporter::record_pet_skipped(int sg, int count) {
    const int i = find_sg(sg);
    if (i >= 0) sg_[i].n_s3_skipped += count;
}

void StatusReporter::record_rejected(int sg, int count) {
    int i = find_sg(sg);
    if (i >= 0) sg_[i].rejected += count;
}

void StatusReporter::set_total(int sg, int count) {
    int i = find_sg(sg);
    if (i < 0) return;
    sg_[i].total = count;
    int delta = sg_[i].total - sg_[i].window_total;
    if (delta >= DISCOVERY_WINDOW) {
        int du = sg_[i].unique - sg_[i].window_unique;
        sg_[i].discovery_rate = static_cast<double>(du) / delta;
        sg_[i].window_total = sg_[i].total;
        sg_[i].window_unique = sg_[i].unique;
    }
}

void StatusReporter::set_unique(int sg, int count) {
    int i = find_sg(sg);
    if (i >= 0) sg_[i].unique = count;
}

void StatusReporter::set_stage3_unique(int sg, int count) {
    int i = find_sg(sg);
    if (i >= 0) sg_[i].s3_unique = count;
}

void StatusReporter::seed_stage3(int sg, int n_rows) {
    int i = find_sg(sg);
    if (i >= 0) sg_[i].n_s3 = n_rows;
}

void StatusReporter::seed_window(int sg, int total, int unique) {
    int i = find_sg(sg);
    if (i < 0) return;
    sg_[i].window_total = total;
    sg_[i].window_unique = unique;
}

SGStats StatusReporter::totals() const {
    SGStats t;
    for (const auto& s : sg_) {
        t.total += s.total; t.unique += s.unique; t.target += s.target;
        t.rejected += s.rejected; t.duplicates += s.duplicates;
        t.invalid += s.invalid; t.time_ms += s.time_ms;
        t.n_diag += s.n_diag; t.n_converged += s.n_converged;
        // The ALL row's spread is the union of the per-SG ranges.
        if (s.n_energy > 0) {
            if (t.n_energy == 0) { t.e_lo = s.e_lo; t.e_hi = s.e_hi; }
            else {
                t.e_lo = std::min(t.e_lo, s.e_lo);
                t.e_hi = std::max(t.e_hi, s.e_hi);
            }
            t.n_energy += s.n_energy;
        }
        t.n_s1 += s.n_s1; t.n_s1_converged += s.n_s1_converged;
        t.s1_iters_sum += s.s1_iters_sum;
        t.iters_sum += s.iters_sum;
        for (int b = 0; b < SGStats::GRAD_BINS; ++b) t.grad_hist[b] += s.grad_hist[b];
        t.n_s3 += s.n_s3; t.n_s3_converged += s.n_s3_converged;
        t.s3_iters_sum += s.s3_iters_sum; t.s3_grad_sum += s.s3_grad_sum;
        t.s3_time_ms += s.s3_time_ms;
        t.n_s3_dropped += s.n_s3_dropped;
        t.n_s3_skipped += s.n_s3_skipped;
        t.s3_unique += s.s3_unique;
    }
    return t;
}

Snapshot StatusReporter::snapshot() const {
    Snapshot sn;
    sn.elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time_).count();
    sn.nranks = nranks_;

    int n_active = 0;
    for (const auto& s : sg_) if (s.total < s.target) ++n_active;
    sn.n_active = std::max(n_active, 1);
    sn.pet_stage = pet_stage_;
    sn.final = final_;

    const SGStats t = totals();
    sn.total = t.total; sn.unique = t.unique;
    sn.target = t.target; sn.rejected = t.rejected;
    // Grand-average ms/structure, the ETA rate for SGs that have not started.
    sn.avg_mps = t.total_ms_per_struct();

    double work_s = 0;
    for (const auto& s : sg_) {
        const int remaining = s.target - s.total;
        if (remaining <= 0) continue;
        sn.remaining_structs += remaining;
        const double mps = s.total_ms_per_struct() > 0 ? s.total_ms_per_struct()
                                                       : sn.avg_mps;
        if (mps > 0) work_s += remaining * mps / 1000.0;
    }
    // Prefer the measured global throughput; fall back to work/nranks until a
    // rate has been sampled.
    sn.overall_eta_s = (measured_rate_ > 1e-9)
        ? sn.remaining_structs / measured_rate_
        : work_s / std::max(1, nranks_);
    return sn;
}

void StatusReporter::update_rate(std::chrono::steady_clock::time_point now) {
    constexpr double RATE_INTERVAL = 5.0;   // min window: absorb within-burst recording
    double dt = std::chrono::duration<double>(now - rate_last_time_).count();
    if (dt < RATE_INTERVAL) return;
    long long grand_total = 0;
    for (auto& s : sg_) grand_total += s.total;
    long long delta = grand_total - rate_last_total_;
    // Results arrive in bursts. Skip an empty window without advancing the
    // baseline, so the next arrival's dt spans the quiet gap and delta/dt is the
    // sustained throughput.
    if (delta <= 0) return;
    double inst = delta / dt;   // structures/sec since the previous batch arrival
    measured_rate_ = (measured_rate_ <= 0.0) ? inst : 0.85 * measured_rate_ + 0.15 * inst;
    rate_last_time_ = now;
    rate_last_total_ = grand_total;
}

void StatusReporter::maybe_update() {
    auto now = std::chrono::steady_clock::now();
    double since = std::chrono::duration<double>(now - last_update_).count();
    if (since < update_interval_) return;
    last_update_ = now;
    update_rate(now);

    // Emit (rewrite status file + reprint) when the per-SG counts change, or
    // every ~20 s as a heartbeat.
    long long sig = 0;
    for (auto& s : sg_) sig = sig * 1000003 + static_cast<long long>(s.total) * 131 + s.unique;
    // A rank stopping also counts as a change.
    sig = sig * 1000003 + winding_down_;
    double since_emit = std::chrono::duration<double>(now - last_emit_).count();
    constexpr double HEARTBEAT = 20.0;
    if (sig != last_emit_sig_ || since_emit >= HEARTBEAT) {
        last_emit_sig_ = sig;
        last_emit_ = now;
        last_anim_ = now;
        write_status_file();
        print_progress();
        return;
    }

    // Counts unchanged: on a TTY, still repaint the in-place bar a few times a
    // second so the spinner and clock tick. The status file is not touched.
    if (is_tty_) {
        double since_anim = std::chrono::duration<double>(now - last_anim_).count();
        if (since_anim >= 0.25) {
            last_anim_ = now;
            print_progress();
        }
    }
}

void StatusReporter::finalize() {
    // Everything has drained, so nothing is outstanding whatever the counters say.
    final_ = true;
    if (is_tty_ && lines_printed_ > 0) {
        // Single-shot: CR, up N lines, clear to end of screen.
        logging::info("\r\033[{}A\033[J", lines_printed_);
    }

    const Snapshot sn = snapshot();
    logging::info("\n=== CSP complete ({}) ===\n", format_duration(sn.elapsed_s));
    // Through the log sink, so --quiet silences the console table. The status
    // file is written regardless.
    if (logging::enabled(logging::Level::Info)) {
        std::string table = header_line(show_stage3_);
        for (const auto& s : sg_)
            table += row_line(fmt::format("{}", s.space_group), s, sn, show_stage3_);
        table += row_line("ALL", totals(), sn, show_stage3_, /*is_total=*/true);
        logging::info("{}", table);
    }
    logging::info("\n{} ranks, {} wall time\n", nranks_, format_duration(sn.elapsed_s));
    std::fflush(stdout);
    write_status_file();
}

void StatusReporter::write_status_file() {
    const Snapshot sn = snapshot();

    std::string path = output_dir_ + "/status.txt";
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) return;

    auto t = std::time(nullptr);
    fmt::print(f, "klasp CSP  {}  [{} rank{}]  commit {}{} ({})\n\n",
               fmt::format("{:%Y-%m-%d %H:%M:%S}", local_tm(t)),
               nranks_, nranks_ > 1 ? "s" : "",
               GIT_HASH_SHORT, GIT_DIRTY ? "-dirty" : "", GIT_BRANCH);
    write_header(f, show_stage3_);
    for (const auto& s : sg_) write_row(f, fmt::format("{}", s.space_group), s, sn, show_stage3_);
    write_row(f, "ALL", totals(), sn, show_stage3_, /*is_total=*/true);
    fmt::print(f, "\nOverall: {}/{} ({} unique, {} rejected)  ETA {}  Elapsed: {}\n",
               sn.total, sn.target, sn.unique, sn.rejected,
               format_duration(sn.overall_eta_s), format_duration(sn.elapsed_s));
    // A large backlog means rank 0 cannot store as fast as the ranks produce.
    if (backlog_ > 1000)
        fmt::print(f, "Waiting to be stored: {} structures (~{} MB held on rank 0)\n",
                   backlog_, (backlog_ * 5120) / (1024 * 1024));
    if (winding_down_ > 0)
        fmt::print(f, "{}\n", winding_down_line());
    // Results only arrive per work unit, so explain a table of zeros.
    if (sn.total == 0 && work_unit_ > 0)
        fmt::print(f,
                   "(no results yet: each rank reports after a full work unit of up to {} "
                   "structures, so the table stays flat until the first one lands)\n",
                   work_unit_);
    if (totals().n_diag == 0)
        fmt::print(f, "(conv/iters/grad not recorded: --no-store-diagnostics)\n");
    std::fclose(f);
    std::rename(tmp.c_str(), path.c_str());
}

// One sentence, used by both the terminal table and status.txt. The count comes
// from rank 0's last claim, so it lags by up to one work unit ("at least").
std::string StatusReporter::winding_down_line(bool compact) const {
    // The live frame erases exactly lines_printed_ rows, so the TTY form must not
    // wrap.
    if (compact)
        return fmt::format("winding down: {}/{} ranks stopped -- ETA n/a",
                           winding_down_, winding_down_of_);
    return fmt::format(
        "Winding down: at least {} of {} ranks have stopped claiming work and are "
        "finishing the unit they hold. The ETA above no longer applies -- the run "
        "ends with the slowest of them, not at the target.",
        winding_down_, winding_down_of_);
}

void StatusReporter::print_progress() {
    const Snapshot sn = snapshot();

    if (is_tty_) {
        if (lines_printed_ > 0) {
            // Up N, then clear from cursor to end of screen, in case stderr output
            // moved the cursor between frames.
            logging::info("\r\033[{}A\033[J", lines_printed_);
        }
        int n_lines = 0;
        for (auto& s : sg_) {
            double pct = s.target > 0 ? 100.0 * s.total / s.target : 0;
            pct = std::min(pct, 100.0);

            int bar_width = 20;
            int filled = static_cast<int>(pct / 100.0 * bar_width);
            std::string bar(filled, '=');
            if (filled < bar_width) bar += '>';
            bar.resize(bar_width, ' ');

            // Compact live view; the full table goes to status.txt.
            const double cf = s.converged_frac();
            const std::string conv = cf < 0 ? "" : fmt::format(" {:.0f}%c", cf * 100.0);
            logging::info("\033[2KSG{:>3d} [{}] {:>5d}/{:<5d} ({:>4d}u {:>4s}{}) {:>5.1f}ms  {}\n",
                       s.space_group, bar, s.total, s.target,
                       s.unique, format_disc(s.discovery_rate, s.total), conv,
                       s.total_ms_per_struct(), format_sg_eta(s, sn));
            ++n_lines;
        }
        // Spinner driven by elapsed time, independent of render cadence.
        static const char SPIN[] = {'|', '/', '-', '\\'};
        const char sp = SPIN[static_cast<int>(sn.elapsed_s * 4) & 3];
        logging::info("\033[2K{} {:>5d}/{:<5d} ({:>4d}u)  ETA {}  elapsed {}\n",
                   sp, sn.total, sn.target, sn.unique,
                   format_duration(sn.overall_eta_s), format_duration(sn.elapsed_s));
        ++n_lines;
        if (winding_down_ > 0) {
            logging::info("\033[2K{}\n", winding_down_line(/*compact=*/true));
            ++n_lines;
        }
        std::fflush(stdout);
        lines_printed_ = n_lines;
    } else {
        auto t = std::time(nullptr);
        logging::info("[{:%H:%M:%S}] {} elapsed\n", local_tm(t),
                   format_duration(sn.elapsed_s));
        {
            std::string table = header_line(show_stage3_);
            for (const auto& s : sg_)
                table += row_line(fmt::format("{}", s.space_group), s, sn, show_stage3_);
            table += row_line("ALL", totals(), sn, show_stage3_, /*is_total=*/true);
            logging::info("{}", table);
        }
        logging::info("total: {}/{} ({} unique)  ETA {}  elapsed {}\n",
                   sn.total, sn.target, sn.unique,
                   format_duration(sn.overall_eta_s), format_duration(sn.elapsed_s));
        if (winding_down_ > 0) logging::info("{}\n", winding_down_line());
        logging::info("\n");
        std::fflush(stdout);
    }
}

}  // namespace klasp
