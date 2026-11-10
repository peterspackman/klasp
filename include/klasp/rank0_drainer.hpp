#pragma once

// Rank-0 MPI drain thread: absorbs TAG_RESULT and TAG_REJ messages from worker
// ranks so their send rings never stall. Work distribution is in WorkWindow.

#include "klasp/result_sender.hpp"
#include "klasp/result_consumer.hpp"

#include <fmt/base.h>
#include <mpi.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace klasp {

class RankZeroDrainer : public ResultDrain {
public:
    RankZeroDrainer(ResultQueue& queue, ResultConsumer& consumer,
                    const std::vector<int>& space_groups, int tag_rej)
        : queue_(queue), consumer_(consumer),
          space_groups_(space_groups), tag_rej_(tag_rej) {}

    // Joins the thread, so an exception between start() and stop() does not
    // destroy a joinable std::thread.
    ~RankZeroDrainer() override { stop(); }

    RankZeroDrainer(const RankZeroDrainer&) = delete;
    RankZeroDrainer& operator=(const RankZeroDrainer&) = delete;

    void start() {
        stop_.store(false, std::memory_order_relaxed);
        thread_ = std::thread(&RankZeroDrainer::run, this);
    }

    // Signal shutdown and join. Safe to call before start() (no-op).
    void stop() override {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) thread_.join();
    }

    // Worker ranks whose final sentinel has arrived; nranks - 1 means every
    // result has been received.
    int sentinels_seen() const { return sentinels_.load(std::memory_order_relaxed); }
    int producers_signed_off() const override { return sentinels_seen(); }

    // Main-thread final drain after the barrier and stop(): absorb straggler
    // TAG_RESULT and TAG_REJ messages.
    void mop_up() override {
        // Both drains must run every pass; `||` would skip the rejection drain.
        for (;;) {
            const bool got_results = drain_results_();
            const bool got_rejections = drain_rejections_();
            if (!got_results && !got_rejections) break;
        }
    }

private:
    void run() {
        while (!stop_.load(std::memory_order_relaxed)) {
            const bool got_results = drain_results_();
            const bool got_rejections = drain_rejections_();   // must also run
            const bool any = got_results || got_rejections;
            if (!any)
                std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    bool drain_results_() {
        bool any = false;
        while (true) {
            int flag;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, ResultSender::TAG_RESULT, MPI_COMM_WORLD, &flag, &st);
            if (!flag) break;
            int bytes = 0;
            MPI_Get_count(&st, MPI_BYTE, &bytes);
            std::vector<char> buf(bytes);
            MPI_Recv(buf.data(), bytes, MPI_BYTE, st.MPI_SOURCE,
                     ResultSender::TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ResultMsg m{};   // unpack can leave it untouched; the sentinel test reads .seed
            unpack_result_msg(buf.data(), bytes, m);
            if (m.sr.seed == -1) {
                // One per worker rank, after its last result. MPI preserves
                // order per source/tag/communicator, so nothing from that rank follows.
                sentinels_.fetch_add(1, std::memory_order_relaxed);
                any = true;
                continue;
            }
            queue_.push(std::move(m));
            any = true;
        }
        return any;
    }

    bool drain_rejections_() {
        const int n_sgs = static_cast<int>(space_groups_.size());
        bool any = false;
        while (true) {
            int flag;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, tag_rej_, MPI_COMM_WORLD, &flag, &st);
            if (!flag) break;
            // kRejFields counters per SG, in blocks of n_sgs: rejected seeds, PET
            // candidates with no stage-3 row, dedup-gate skips, vacuum-slab drops.
            std::vector<int> buf(kRejFields * n_sgs);
            MPI_Recv(buf.data(), kRejFields * n_sgs, MPI_INT, st.MPI_SOURCE, tag_rej_,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            record_rejection_delta(consumer_, space_groups_, buf);
            any = true;
        }
        return any;
    }


    ResultQueue& queue_;
    ResultConsumer& consumer_;
    const std::vector<int>& space_groups_;
    int tag_rej_;
    std::atomic<bool> stop_{false};
    std::atomic<int> sentinels_{0};
    std::thread thread_;
};

}  // namespace klasp
