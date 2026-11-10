#pragma once

// Where klasp's output goes. Library code writes through a replaceable sink;
// the default writes to stdout/stderr unadorned. Rank gating is the sink's job,
// not the call site's.
//
//     logging::info("  {} structures stored\n", n);   // running commentary
//     logging::warn("... could not read {}\n", path); // stderr, shown unless quiet
//     logging::error("refusing: {}\n", why);          // stderr, always shown
//     logging::detail("    slot {} refilled\n", i);   // only under -v

#include <fmt/format.h>

#include <functional>
#include <string>
#include <string_view>

namespace klasp::logging {

enum class Level : int {
    Error = 0,    // refusals and failures; never suppressed
    Warn = 1,     // something looks wrong but the run continues
    Info = 2,     // the normal running commentary (the default ceiling)
    Detail = 3,   // per-batch, per-slot, per-structure detail (-v)
    Debug = 4,    // -vv
};

// `text` is already formatted and carries its own newline.
using Sink = std::function<void(Level level, std::string_view text)>;

// Replace the sink; nullptr restores the default (stdout for Info and below,
// stderr for Warn and Error).
void set_sink(Sink sink);

// Messages above this level are dropped. Info by default.
void set_level(Level level);
Level level();

// Only rank 0 emits Info and Detail; Warn and Error come from every rank.
// Defaults to rank 0.
void set_rank(int rank);
int rank();

void emit(Level level, std::string_view text);

// True when a message at this level would be emitted.
bool enabled(Level level);

namespace detail_ns {
template <typename... Args>
inline void emit_fmt(Level lv, fmt::format_string<Args...> f, Args&&... args) {
    if (!enabled(lv)) return;
    emit(lv, fmt::format(f, std::forward<Args>(args)...));
}
}  // namespace detail_ns

template <typename... Args>
inline void error(fmt::format_string<Args...> f, Args&&... args) {
    detail_ns::emit_fmt(Level::Error, f, std::forward<Args>(args)...);
}
template <typename... Args>
inline void warn(fmt::format_string<Args...> f, Args&&... args) {
    detail_ns::emit_fmt(Level::Warn, f, std::forward<Args>(args)...);
}
template <typename... Args>
inline void info(fmt::format_string<Args...> f, Args&&... args) {
    detail_ns::emit_fmt(Level::Info, f, std::forward<Args>(args)...);
}
template <typename... Args>
inline void detail(fmt::format_string<Args...> f, Args&&... args) {
    detail_ns::emit_fmt(Level::Detail, f, std::forward<Args>(args)...);
}
template <typename... Args>
inline void debug(fmt::format_string<Args...> f, Args&&... args) {
    detail_ns::emit_fmt(Level::Debug, f, std::forward<Args>(args)...);
}

// Plain output: no ANSI escapes, no in-place redraw. Set when stdout is not a
// terminal or by --plain; KLASP_PLAIN is still read but deprecated.
void set_plain(bool plain);
bool plain();

// Collect everything emitted while alive instead of printing it. Restores the
// previous sink and level on destruction.
class Capture {
public:
    explicit Capture(Level level = Level::Detail);
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    const std::string& text() const { return text_; }

private:
    std::string text_;
    Sink prev_sink_;
    Level prev_level_;
};

}  // namespace klasp::logging
