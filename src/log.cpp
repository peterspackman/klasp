#include "klasp/log.hpp"

#include <cstdio>

namespace klasp::logging {

namespace {

// Process-wide and not atomic: level and sink are set during startup, before
// any printing threads exist. A sink that needs synchronisation takes its own lock.
Level g_level = Level::Info;
int g_rank = 0;
Sink g_sink;
bool g_plain = false;

void default_sink(Level level, std::string_view text) {
    // Warnings and errors to stderr, everything else to stdout.
    std::FILE* out = (level <= Level::Warn) ? stderr : stdout;
    std::fwrite(text.data(), 1, text.size(), out);
    // Flush stdout so a run that dies in a kernel keeps its last line.
    if (out == stdout) std::fflush(stdout);
}

}  // namespace

void set_sink(Sink sink) { g_sink = std::move(sink); }
void set_level(Level level) { g_level = level; }
Level level() { return g_level; }
void set_rank(int rank) { g_rank = rank; }
void set_plain(bool plain) { g_plain = plain; }
bool plain() { return g_plain; }
int rank() { return g_rank; }

bool enabled(Level lv) {
    if (lv > g_level) return false;
    // Warnings and errors come from every rank; narration from rank 0 only.
    if (lv >= Level::Info && g_rank != 0) return false;
    return true;
}

void emit(Level lv, std::string_view text) {
    if (!enabled(lv)) return;
    if (g_sink) g_sink(lv, text);
    else default_sink(lv, text);
}

Capture::Capture(Level lv) : prev_sink_(g_sink), prev_level_(g_level) {
    g_level = lv;
    g_sink = [this](Level, std::string_view text) { text_.append(text); };
}

Capture::~Capture() {
    g_sink = std::move(prev_sink_);
    g_level = prev_level_;
}

}  // namespace klasp::logging
