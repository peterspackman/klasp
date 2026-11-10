#include <catch2/catch_test_macros.hpp>
#include "klasp/result_queue.hpp"

#include <vector>

using namespace klasp;

// The worker -> rank 0 result message: a fixed POD header followed by two
// variable-length tails (PDD and PET stage-3 geometry).

static StructureResult make_result(int seed) {
    StructureResult r{};
    r.seed = seed;
    r.space_group = 14;
    r.stage = 3;
    r.n_mols = 2;
    r.energy = -123.456;
    r.density = 1.42;
    r.cell_params[0] = 10.0;
    r.cell_params[5] = 90.0;
    r.amd[0] = 3.25;
    r.powder[0] = 0.5f;
    return r;
}

static FlexibleStructure make_flex(int n_atoms) {
    FlexibleStructure fs;
    fs.n_atoms = n_atoms;
    for (int k = 0; k < 6; ++k) fs.cell_params[k] = 5.0 + k;
    fs.Z.resize(n_atoms);
    fs.frac.resize(3ull * n_atoms);
    for (int a = 0; a < n_atoms; ++a) {
        fs.Z[a] = 1 + (a % 8);
        for (int d = 0; d < 3; ++d) fs.frac[3 * a + d] = 0.01 * (3 * a + d);
    }
    return fs;
}

static ResultMsg round_trip(const StructureResult& r, const std::vector<double>& pdd,
                            const FlexibleStructure* flex) {
    std::vector<char> buf;
    pack_result_msg(buf, r, pdd.empty() ? nullptr : pdd.data(),
                    static_cast<int>(pdd.size()), flex);
    ResultMsg out;
    unpack_result_msg(buf.data(), static_cast<int>(buf.size()), out);
    return out;
}

TEST_CASE("result message round-trips with no tails", "[result_msg]") {
    auto r = make_result(7);
    ResultMsg out = round_trip(r, {}, nullptr);

    REQUIRE(out.sr.seed == 7);
    REQUIRE(out.sr.space_group == 14);
    REQUIRE(out.sr.energy == r.energy);
    REQUIRE(out.pdd.empty());
    REQUIRE(out.flex.n_atoms == 0);
}

TEST_CASE("result message round-trips the PDD tail", "[result_msg]") {
    auto r = make_result(11);
    std::vector<double> pdd{1.5, 2.5, 3.5, 4.5, 5.5};

    ResultMsg out = round_trip(r, pdd, nullptr);

    REQUIRE(out.sr.seed == 11);
    REQUIRE(out.pdd == pdd);
    REQUIRE(out.flex.n_atoms == 0);
}

TEST_CASE("result message round-trips the flexible geometry", "[result_msg]") {
    auto r = make_result(23);
    // Odd atom counts leave the trailing doubles misaligned, which a
    // reinterpret_cast would break on.
    for (int n_atoms : {1, 7, 16, 33}) {
        auto fs = make_flex(n_atoms);
        ResultMsg out = round_trip(r, {}, &fs);

        REQUIRE(out.flex.n_atoms == n_atoms);
        REQUIRE(out.flex.Z == fs.Z);
        REQUIRE(out.flex.frac == fs.frac);
        for (int k = 0; k < 6; ++k) REQUIRE(out.flex.cell_params[k] == fs.cell_params[k]);
    }
}

TEST_CASE("result message round-trips both tails together", "[result_msg]") {
    auto r = make_result(42);
    std::vector<double> pdd{0.25, 0.5, 0.75};
    auto fs = make_flex(9);

    ResultMsg out = round_trip(r, pdd, &fs);

    REQUIRE(out.sr.seed == 42);
    REQUIRE(out.pdd == pdd);
    REQUIRE(out.flex.n_atoms == 9);
    REQUIRE(out.flex.Z == fs.Z);
    REQUIRE(out.flex.frac == fs.frac);
}

TEST_CASE("an empty flexible payload is not sent", "[result_msg]") {
    auto r = make_result(3);
    FlexibleStructure empty;  // n_atoms == 0

    std::vector<char> with_flex, without_flex;
    pack_result_msg(with_flex, r, nullptr, 0, &empty);
    pack_result_msg(without_flex, r, nullptr, 0, nullptr);

    REQUIRE(with_flex.size() == without_flex.size());
}

TEST_CASE("the shutdown sentinel unpacks without a tail", "[result_msg]") {
    // send_sentinel() sends a bare StructureResult with no counts; unpack must
    // stop at the header.
    StructureResult sentinel{};
    sentinel.seed = -1;

    ResultMsg out;
    unpack_result_msg(reinterpret_cast<const char*>(&sentinel), sizeof(StructureResult), out);

    REQUIRE(out.sr.seed == -1);
    REQUIRE(out.pdd.empty());
    REQUIRE(out.flex.n_atoms == 0);
}

// --------------------------------------------------------------------------
// Rank 0's queue ceiling: backpressure on producers, which must always be woken
// again, by the consumer making room or by release_producers() at shutdown.

#include <atomic>
#include <chrono>
#include <thread>

namespace {

// A message whose payload dominates the fixed header, so a test can fill the
// queue with a handful of them rather than thousands.
ResultMsg big_msg(int seed, std::size_t pdd_doubles) {
    ResultMsg m;
    m.sr = make_result(seed);
    m.pdd.assign(pdd_doubles, 1.5);
    return m;
}

}  // namespace

TEST_CASE("the queue stops growing at its ceiling", "[result_msg]") {
    ResultQueue queue;
    const std::size_t payload = 1u << 20;            // 8 MB of PDD per message
    queue.set_max_queue_bytes(20u << 20);           // room for two, not three

    queue.push(big_msg(0, payload));
    queue.push(big_msg(1, payload));
    CHECK(queue.queue_size_hint() == 2);
    CHECK(queue.queue_bytes_hint() > (16u << 20));

    // The third has nowhere to go until something is popped.
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        queue.push(big_msg(2, payload));
        pushed.store(true);
    });
    // Wait for the producer to reach the ceiling rather than sleeping.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (queue.waiting_producers() == 0
           && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    REQUIRE(queue.waiting_producers() == 1);
    CHECK_FALSE(pushed.load());
    CHECK(queue.queue_size_hint() == 2);

    // Draining one releases it.
    ResultMsg out;
    REQUIRE(queue.try_pop(out));
    producer.join();
    CHECK(pushed.load());
    CHECK(queue.queue_size_hint() == 2);
    CHECK(queue.throttled_pushes() == 1);
}

TEST_CASE("a producer parked on the ceiling is released at shutdown", "[result_msg]") {
    // ResultConsumer::stop() joins the consumer thread, so it must call
    // release_producers() first or parked producers wait forever.
    ResultQueue queue;
    const std::size_t payload = 1u << 20;
    queue.set_max_queue_bytes(12u << 20);
    queue.push(big_msg(0, payload));

    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        queue.push(big_msg(1, payload));
        pushed.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE_FALSE(pushed.load());

    queue.release_producers();      // nothing is draining; let it through anyway
    producer.join();
    CHECK(pushed.load());
    CHECK(queue.queue_size_hint() == 2);
}

TEST_CASE("a message bigger than the whole ceiling still gets through", "[result_msg]") {
    // Otherwise a run whose PDD is larger than the ceiling would park its first
    // producer on room that can never appear.
    ResultQueue queue;
    queue.set_max_queue_bytes(1024);
    queue.push(big_msg(0, 1u << 16));   // 512 kB, far over the ceiling
    CHECK(queue.queue_size_hint() == 1);
    ResultMsg out;
    REQUIRE(queue.try_pop(out));
    CHECK(out.sr.seed == 0);
}

TEST_CASE("nothing is lost when producers are throttled", "[result_msg]") {
    // Backpressure must slow producers down, never drop what they produced.
    ResultQueue queue;
    const int n_producers = 4, per_producer = 250;
    queue.set_max_queue_bytes(256u << 10);      // tight

    std::atomic<bool> done{false};
    std::vector<int> seen;
    std::thread consumer([&] {
        ResultMsg m;
        while (!done.load() || !queue.queue_empty()) {
            if (queue.try_pop(m)) seen.push_back(m.sr.seed);
            else std::this_thread::yield();
        }
    });

    std::vector<std::thread> producers;
    for (int p = 0; p < n_producers; ++p)
        producers.emplace_back([&, p] {
            for (int i = 0; i < per_producer; ++i)
                queue.push(big_msg(p * per_producer + i, 512));
        });
    for (auto& t : producers) t.join();
    done.store(true);
    consumer.join();

    REQUIRE(seen.size() == static_cast<std::size_t>(n_producers * per_producer));
    std::sort(seen.begin(), seen.end());
    for (int i = 0; i < n_producers * per_producer; ++i) CHECK(seen[i] == i);
    // And the queue accounting came back to zero rather than drifting.
    CHECK(queue.queue_bytes_hint() == 0);
}

TEST_CASE("an unbounded queue never throttles", "[result_msg]") {
    ResultQueue queue;
    queue.set_max_queue_bytes(0);
    for (int i = 0; i < 200; ++i) queue.push(big_msg(i, 1024));
    CHECK(queue.queue_size_hint() == 200);
    CHECK(queue.throttled_pushes() == 0);
}
