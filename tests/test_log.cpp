// The output sink: levels, rank gating, redirection.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "klasp/log.hpp"

using Catch::Matchers::ContainsSubstring;
namespace klog = klasp::logging;  // not `log`: that is the math function

namespace {
// Every test leaves the global logging state as it found it.
struct ScopedState {
    klog::Level level = klog::level();
    int rank = klog::rank();
    ~ScopedState() { klog::set_level(level); klog::set_rank(rank); klog::set_sink(nullptr); }
};
}  // namespace

TEST_CASE("Capture takes the output instead of printing it", "[log]") {
    ScopedState guard;
    klog::set_rank(0);
    klog::Capture cap;
    klog::info("stored {} structures\n", 42);
    klog::warn("could not read {}\n", "x.json");
    CHECK_THAT(cap.text(), ContainsSubstring("stored 42 structures"));
    CHECK_THAT(cap.text(), ContainsSubstring("could not read x.json"));
}

TEST_CASE("Capture restores the sink and level it replaced", "[log]") {
    ScopedState guard;
    klog::set_level(klog::Level::Warn);
    {
        klog::Capture cap(klog::Level::Debug);
        CHECK(klog::level() == klog::Level::Debug);
    }
    CHECK(klog::level() == klog::Level::Warn);
}

TEST_CASE("The level decides what is emitted", "[log]") {
    ScopedState guard;
    klog::set_rank(0);

    SECTION("quiet keeps errors and drops narration") {
        klog::Capture cap(klog::Level::Error);
        klog::error("refused\n");
        klog::warn("odd\n");
        klog::info("running\n");
        CHECK_THAT(cap.text(), ContainsSubstring("refused"));
        CHECK_THAT(cap.text(), !ContainsSubstring("odd"));
        CHECK_THAT(cap.text(), !ContainsSubstring("running"));
    }
    SECTION("the default ceiling is info, so detail is dropped") {
        klog::Capture cap(klog::Level::Info);
        klog::info("running\n");
        klog::detail("slot 3 refilled\n");
        CHECK_THAT(cap.text(), ContainsSubstring("running"));
        CHECK_THAT(cap.text(), !ContainsSubstring("slot 3"));
    }
    SECTION("verbose lets detail through") {
        klog::Capture cap(klog::Level::Detail);
        klog::detail("slot 3 refilled\n");
        CHECK_THAT(cap.text(), ContainsSubstring("slot 3"));
    }
}

TEST_CASE("Only rank 0 narrates, but every rank reports a problem", "[log]") {
    ScopedState guard;
    klog::set_rank(7);
    klog::Capture cap(klog::Level::Debug);
    klog::info("running\n");
    klog::detail("slot 3\n");
    klog::warn("odd\n");
    klog::error("refused\n");
    CHECK_THAT(cap.text(), !ContainsSubstring("running"));
    CHECK_THAT(cap.text(), !ContainsSubstring("slot 3"));
    CHECK_THAT(cap.text(), ContainsSubstring("odd"));
    CHECK_THAT(cap.text(), ContainsSubstring("refused"));
}

TEST_CASE("enabled() agrees with what is emitted", "[log]") {
    ScopedState guard;
    klog::set_rank(0);
    klog::Capture cap(klog::Level::Info);
    CHECK(klog::enabled(klog::Level::Error));
    CHECK(klog::enabled(klog::Level::Info));
    CHECK_FALSE(klog::enabled(klog::Level::Detail));
    if (klog::enabled(klog::Level::Detail)) klog::detail("expensive\n");
    CHECK_THAT(cap.text(), !ContainsSubstring("expensive"));
}

TEST_CASE("A message carries its own newline through the sink", "[log]") {
    ScopedState guard;
    klog::set_rank(0);
    klog::Capture cap;
    klog::info("one\n");
    klog::info("two\n");
    CHECK(cap.text() == "one\ntwo\n");
}
