#pragma once
// A one-line progress bar for long single-process commands, on stderr. With
// --plain, KLASP_PLAIN=1 or a non-terminal stderr it prints occasional lines instead.

#include <string>

namespace klasp {

class ProgressBar {
public:
    // `total` <= 0 disables the bar; `enabled` false makes every method a no-op.
    ProgressBar(std::string label, long long total, bool enabled);
    ~ProgressBar();

    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;

    // Report absolute progress; redraws are rate limited.
    void set(long long done);
    void advance(long long by = 1) { set(done_ + by); }

    // Draw the completed bar and end the line. The destructor calls it if needed.
    void finish();

private:
    void draw(bool force);

    std::string label_;
    long long total_ = 0;
    long long done_ = 0;
    bool enabled_ = false;
    bool tty_ = false;
    double start_s_ = 0.0;
    double last_draw_s_ = -1.0;
    long long last_plain_pct_ = -1;
    bool finished_ = false;
};

}  // namespace klasp
