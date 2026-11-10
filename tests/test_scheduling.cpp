#include <catch2/catch_test_macros.hpp>
#include "klasp/scheduling.hpp"

using namespace klasp::sched;

TEST_CASE("stream_pool_target caps to remaining without rounding up", "[scheduling]") {
    const int P = 8192;   // e.g. stream_pool_mult(8) * batch(1024)

    SECTION("unbounded when remaining <= 0") {
        REQUIRE(stream_pool_target(0, P) == P);
        REQUIRE(stream_pool_target(-5, P) == P);
    }
    SECTION("remaining smaller than pool -> exactly remaining (no batch round-up)") {
        REQUIRE(stream_pool_target(1250, P) == 1250);
        REQUIRE(stream_pool_target(1, P) == 1);
    }
    SECTION("remaining larger than pool -> clamped to pool capacity") {
        REQUIRE(stream_pool_target(20000, P) == P);
    }
    SECTION("remaining equal to pool") {
        REQUIRE(stream_pool_target(P, P) == P);
    }
}
