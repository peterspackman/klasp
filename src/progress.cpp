#include "klasp/log.hpp"
#include "klasp/progress.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace klasp {

namespace {

double now_s() {
    using Clock = std::chrono::steady_clock;
    static const Clock::time_point t0 = Clock::now();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Same format as the status view.
std::string format_duration(double secs) {
    if (secs < 0 || !std::isfinite(secs)) return "---";
    const int s = static_cast<int>(secs);
    if (s < 60) return fmt::format("{}s", s);
    if (s < 3600) return fmt::format("{}m{}s", s / 60, s % 60);
    return fmt::format("{}h{}m", s / 3600, (s % 3600) / 60);
}

bool stderr_is_tty() {
    // --plain, or KLASP_PLAIN.
    if (logging::plain()) return false;
    if (const char* p = std::getenv("KLASP_PLAIN"))
        if (p[0] != '\0' && p[0] != '0') return false;
    return isatty(fileno(stderr)) != 0;
}

constexpr double REDRAW_INTERVAL_S = 0.1;
constexpr int BAR_WIDTH = 24;

}  // namespace

ProgressBar::ProgressBar(std::string label, long long total, bool enabled)
    : label_(std::move(label)), total_(total), enabled_(enabled && total > 0) {
    if (!enabled_) return;
    tty_ = stderr_is_tty();
    start_s_ = now_s();
    draw(/*force=*/true);
}

ProgressBar::~ProgressBar() { finish(); }

void ProgressBar::set(long long done) {
    if (!enabled_ || finished_) return;
    done_ = std::min(done, total_);
    draw(/*force=*/false);
}

void ProgressBar::draw(bool force) {
    const double t = now_s();
    if (!force && t - last_draw_s_ < REDRAW_INTERVAL_S) return;
    last_draw_s_ = t;

    const double frac = total_ > 0 ? static_cast<double>(done_) / static_cast<double>(total_) : 0.0;
    const double elapsed = t - start_s_;

    if (!tty_) {
        // No cursor: emit a line every 10% instead of redrawing.
        const long long pct = static_cast<long long>(frac * 10.0) * 10;
        if (!force && pct == last_plain_pct_) return;
        last_plain_pct_ = pct;
        logging::warn("{}: {}/{} ({:.0f}%) after {}\n", label_, done_, total_,
                   frac * 100.0, format_duration(elapsed));
        std::fflush(stderr);
        return;
    }

    // ETA from the average rate so far.
    const double eta = (done_ > 0 && frac > 0) ? elapsed * (1.0 - frac) / frac : -1.0;

    const int filled = static_cast<int>(frac * BAR_WIDTH);
    std::string bar(static_cast<std::size_t>(std::min(filled, BAR_WIDTH)), '=');
    if (filled < BAR_WIDTH) bar += '>';
    bar.resize(BAR_WIDTH, ' ');

    logging::warn("\r\033[2K{} [{}] {}/{} ({:.0f}%)  {}  ETA {}", label_, bar, done_,
               total_, frac * 100.0, format_duration(elapsed), format_duration(eta));
    std::fflush(stderr);
}

void ProgressBar::finish() {
    if (!enabled_ || finished_) return;
    finished_ = true;
    const double elapsed = now_s() - start_s_;
    if (tty_) {
        std::string bar(BAR_WIDTH, '=');
        logging::warn("\r\033[2K{} [{}] {}/{} (100%)  {}\n", label_, bar, done_, total_,
                   format_duration(elapsed));
    } else {
        logging::warn("{}: {}/{} done in {}\n", label_, done_, total_,
                   format_duration(elapsed));
    }
    std::fflush(stderr);
}

}  // namespace klasp
