#pragma once

// Result messages, their byte layout, the bounded queue the consumer drains,
// and the ResultSink interface producers hand results to. No MPI here; the MPI
// sink is in result_sender.hpp, the in-process one is QueueSink below.

#include "klasp/dedup.hpp"
#include "klasp/database.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

namespace klasp {

// A result plus its optional variable-length payloads: the PDD (n_asym*k,
// empty when PDD storage is off) and the explicit non-rigid geometry (PET
// stage 3 only; n_atoms == 0 otherwise).
struct ResultMsg {
    StructureResult sr;
    std::vector<double> pdd;
    FlexibleStructure flex;
};

// Wire layout of one result message (worker rank -> rank 0):
//
//   [StructureResult sr][int32 pdd_len][int32 flex_atoms]
//   [double pdd[pdd_len]]
//   [double cell[6]][int32 Z[flex_atoms]][double frac[3*flex_atoms]]
//
// The shutdown sentinel is a bare StructureResult with seed == -1 and no tail,
// so a reader must check the seed before trusting the counts.
inline void pack_result_msg(std::vector<char>& buf, const StructureResult& r,
                            const double* pdd, int pdd_len,
                            const FlexibleStructure* flex) {
    const int n_flex = (flex && flex->n_atoms > 0) ? flex->n_atoms : 0;
    const size_t pdd_bytes = static_cast<size_t>(pdd_len > 0 ? pdd_len : 0) * sizeof(double);
    const size_t flex_bytes =
        n_flex > 0 ? 6 * sizeof(double) + static_cast<size_t>(n_flex) * sizeof(std::int32_t)
                         + 3ull * n_flex * sizeof(double)
                   : 0;
    buf.resize(sizeof(StructureResult) + 2 * sizeof(std::int32_t) + pdd_bytes + flex_bytes);

    char* p = buf.data();
    std::memcpy(p, &r, sizeof(StructureResult));
    p += sizeof(StructureResult);
    const std::int32_t counts[2] = {static_cast<std::int32_t>(pdd_bytes / sizeof(double)),
                                    static_cast<std::int32_t>(n_flex)};
    std::memcpy(p, counts, sizeof(counts));
    p += sizeof(counts);
    if (pdd_bytes > 0) {
        std::memcpy(p, pdd, pdd_bytes);
        p += pdd_bytes;
    }
    if (n_flex > 0) {
        std::memcpy(p, flex->cell_params, 6 * sizeof(double));
        p += 6 * sizeof(double);
        const std::vector<std::int32_t> z32(flex->Z.begin(), flex->Z.begin() + n_flex);
        std::memcpy(p, z32.data(), z32.size() * sizeof(std::int32_t));
        p += z32.size() * sizeof(std::int32_t);
        std::memcpy(p, flex->frac.data(), 3ull * n_flex * sizeof(double));
    }
}

// Inverse of pack_result_msg. A sentinel unpacks to just its header (check
// out.sr.seed == -1); the tails stay empty.
inline void unpack_result_msg(const char* buf, int bytes, ResultMsg& out) {
    if (bytes < static_cast<int>(sizeof(StructureResult))) return;
    std::memcpy(&out.sr, buf, sizeof(StructureResult));
    const char* p = buf + sizeof(StructureResult);
    const char* end = buf + bytes;
    if (end - p < static_cast<ptrdiff_t>(2 * sizeof(std::int32_t))) return;
    std::int32_t counts[2];
    std::memcpy(counts, p, sizeof(counts));
    p += sizeof(counts);

    if (counts[0] > 0) {
        out.pdd.resize(counts[0]);
        std::memcpy(out.pdd.data(), p, out.pdd.size() * sizeof(double));
        p += out.pdd.size() * sizeof(double);
    }
    if (counts[1] > 0) {
        const int n = counts[1];
        out.flex.n_atoms = n;
        std::memcpy(out.flex.cell_params, p, 6 * sizeof(double));
        p += 6 * sizeof(double);
        std::vector<std::int32_t> z32(n);
        std::memcpy(z32.data(), p, z32.size() * sizeof(std::int32_t));
        p += z32.size() * sizeof(std::int32_t);
        out.flex.Z.assign(z32.begin(), z32.end());
        out.flex.frac.resize(3ull * n);
        std::memcpy(out.flex.frac.data(), p, out.flex.frac.size() * sizeof(double));
    }
}

// Bounded, thread-safe queue between result producers and the ResultConsumer
// that stores them.
class ResultQueue {
public:
    // Default ceiling on pending results, in bytes (messages vary in size with
    // the PDD tail).
    static constexpr std::size_t DEFAULT_MAX_QUEUE_BYTES = 256ull << 20;

    // 0 removes the ceiling. Call before the consumer starts.
    void set_max_queue_bytes(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        max_queue_bytes_ = bytes;
        space_cv_.notify_all();
    }

    // Shutdown: stop blocking producers on the ceiling, since nothing will drain.
    // Called at the top of ResultConsumer::stop(), before the join.
    void release_producers() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        draining_ = true;
        space_cv_.notify_all();
    }

    // Enqueue a result, whether it came from this process or off the wire.
    void push(ResultMsg&& m) { push_blocking(std::move(m)); }

    bool try_pop(ResultMsg& out) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop();
        queue_bytes_ -= std::min(queue_bytes_, msg_bytes(out));
        lock.unlock();
        space_cv_.notify_all();
        return true;
    }

    bool queue_empty() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.empty();
    }

    // Best-effort queue size.
    size_t queue_size_hint() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

    size_t queue_bytes_hint() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_bytes_;
    }

    // Producers currently blocked on the ceiling.
    int waiting_producers() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return n_waiting_;
    }

    // Pushes that had to wait for room; non-zero means the consumer set the pace.
    long long throttled_pushes() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return n_throttled_;
    }

private:
    // Bytes this message holds on the queue.
    static std::size_t msg_bytes(const ResultMsg& m) {
        return sizeof(StructureResult)
             + m.pdd.size() * sizeof(double)
             + static_cast<std::size_t>(std::max(0, m.flex.n_atoms))
                   * (sizeof(int) + 3 * sizeof(double));
    }

    // Wait for room, then enqueue. A single message larger than the whole
    // ceiling goes through when the queue is empty.
    void push_blocking(ResultMsg&& m) {
        const std::size_t bytes = msg_bytes(m);
        std::unique_lock<std::mutex> lock(queue_mutex_);
        bool waited = false;
        while (!draining_ && max_queue_bytes_ > 0 && !queue_.empty() &&
               queue_bytes_ + bytes > max_queue_bytes_) {
            if (!waited) ++n_waiting_;
            waited = true;
            space_cv_.wait(lock);
        }
        if (waited) { --n_waiting_; ++n_throttled_; }
        queue_bytes_ += bytes;
        queue_.push(std::move(m));
    }

    mutable std::mutex queue_mutex_;
    std::condition_variable space_cv_;
    std::queue<ResultMsg> queue_;
    std::size_t queue_bytes_ = 0;
    std::size_t max_queue_bytes_ = DEFAULT_MAX_QUEUE_BYTES;
    bool draining_ = false;
    long long n_throttled_ = 0;
    int n_waiting_ = 0;
};

// Per-space-group counters in one rejection message: rejected seeds, PET
// candidates with no stage-3 row, PET candidates skipped by the dedup gate,
// structures dropped for vacuum. Sender and receiver must agree on this count.
inline constexpr int kRejFields = 4;

// Records per-space-group rejection deltas (kRejFields layout) directly into an
// in-process consumer.
using RejectionRecorder = std::function<void(const std::vector<int>& delta)>;

// Where a work loop hands its results: ResultSender (MPI) or QueueSink
// (in-process).
class ResultSink {
public:
    virtual ~ResultSink() = default;

    virtual void send(const StructureResult& r, const double* pdd = nullptr,
                      int pdd_len = 0, const FlexibleStructure* flex = nullptr) = 0;

    // Every result this sink accepted is now out of this producer's buffers.
    virtual void flush() {}

    // "This producer has sent its last result." No-op in-process.
    virtual void send_sentinel() {}

    // Rejection counters, sent separately from results. `blocking` waits until
    // the delta has left, as shutdown needs.
    virtual void send_rejections(const std::vector<int>& delta, bool blocking) = 0;

    // A deferred rejection message has released its buffer (it dies with the sink).
    virtual void flush_rejections() {}
};

// The receiving side of results as the shutdown sequence sees it: the rank-0
// drain thread under MPI; a no-op in one process.
class ResultDrain {
public:
    virtual ~ResultDrain() = default;

    // Producers that have sent their last result and signed off. Shutdown waits
    // for one per remote producer.
    virtual int producers_signed_off() const { return 0; }

    // Stop absorbing, then sweep up what is in flight; called in that order
    // (see csp_shutdown).
    virtual void stop() {}
    virtual void mop_up() {}
};

// The sink for a consumer in this process: results onto its queue, rejection
// counters straight into it.
class QueueSink : public ResultSink {
public:
    QueueSink(ResultQueue& queue, RejectionRecorder record)
        : queue_(queue), record_(std::move(record)) {}

    void send(const StructureResult& r, const double* pdd = nullptr,
              int pdd_len = 0, const FlexibleStructure* flex = nullptr) override {
        ResultMsg m;
        m.sr = r;
        if (pdd && pdd_len > 0) m.pdd.assign(pdd, pdd + pdd_len);
        if (flex && flex->n_atoms > 0) m.flex = *flex;
        queue_.push(std::move(m));
    }

    void send_rejections(const std::vector<int>& delta, bool /*blocking*/) override {
        if (record_) record_(delta);
    }

    ResultQueue& queue() { return queue_; }

private:
    ResultQueue& queue_;
    RejectionRecorder record_;
};

}  // namespace klasp
