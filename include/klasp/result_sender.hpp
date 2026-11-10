#pragma once

// The MPI half of the result path; the message layout, queue and ResultSink
// interface are in result_queue.hpp.

#include "klasp/result_queue.hpp"

#include <mpi.h>
#include <cstddef>
#include <vector>

namespace klasp {

// Fire-and-forget result dispatch.
// Rank 0: hands straight to the local QueueSink.
// Workers: MPI_Isend to rank 0 through a ring buffer.
class ResultSender : public ResultSink {
public:
    static constexpr int TAG_RESULT = 1;
    // Per-space-group rejection counters, on their own tag.
    static constexpr int TAG_REJ = 8;
    // Ring buffer size; workers block in MPI_Wait when all slots are in flight.
    static constexpr int MAX_INFLIGHT = 1024;

    static constexpr std::size_t DEFAULT_MAX_QUEUE_BYTES =
        ResultQueue::DEFAULT_MAX_QUEUE_BYTES;

    // `local` is the rank-0 queue and rejection recorder; null on workers.
    ResultSender(int rank, int /*nranks*/, QueueSink* local = nullptr)
        : rank_(rank), local_(local),
          send_bufs_(MAX_INFLIGHT), send_reqs_(MAX_INFLIGHT, MPI_REQUEST_NULL) {}

    ~ResultSender() override {
        if (rej_req_ != MPI_REQUEST_NULL) MPI_Wait(&rej_req_, MPI_STATUS_IGNORE);
    }

    ResultSender(const ResultSender&) = delete;
    ResultSender& operator=(const ResultSender&) = delete;

    void send(const StructureResult& r, const double* pdd = nullptr, int pdd_len = 0,
              const FlexibleStructure* flex = nullptr) override {
        if (rank_ == 0) {
            local_->send(r, pdd, pdd_len, flex);
        } else {
            if (send_reqs_[slot_] != MPI_REQUEST_NULL)
                MPI_Wait(&send_reqs_[slot_], MPI_STATUS_IGNORE);
            auto& buf = send_bufs_[slot_];
            pack_result_msg(buf, r, pdd, pdd_len, flex);
            MPI_Isend(buf.data(), static_cast<int>(buf.size()), MPI_BYTE,
                      0, TAG_RESULT, MPI_COMM_WORLD, &send_reqs_[slot_]);
            slot_ = (slot_ + 1) % MAX_INFLIGHT;
        }
    }

    void flush() override {
        for (auto& req : send_reqs_)
            if (req != MPI_REQUEST_NULL) MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    void send_sentinel() override {
        if (rank_ != 0) {
            StructureResult sentinel{};
            sentinel.seed = -1;
            MPI_Send(&sentinel, sizeof(StructureResult), MPI_BYTE,
                     0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    // Rank 0 records directly; a worker sends one message with all kRejFields
    // counters. `blocking` waits for the send to complete.
    void send_rejections(const std::vector<int>& delta, bool blocking) override {
        if (rank_ == 0) {
            local_->send_rejections(delta, blocking);
            return;
        }
        if (rej_req_ != MPI_REQUEST_NULL) MPI_Wait(&rej_req_, MPI_STATUS_IGNORE);
        if (blocking) {
            MPI_Send(delta.data(), static_cast<int>(delta.size()), MPI_INT,
                     0, TAG_REJ, MPI_COMM_WORLD);
            return;
        }
        rej_buf_ = delta;                  // must outlive the Isend
        MPI_Isend(rej_buf_.data(), static_cast<int>(rej_buf_.size()), MPI_INT,
                  0, TAG_REJ, MPI_COMM_WORLD, &rej_req_);
    }

    // Wait for any outstanding rejection Isend; rej_buf_ must outlive it.
    void flush_rejections() override {
        if (rej_req_ != MPI_REQUEST_NULL) MPI_Wait(&rej_req_, MPI_STATUS_IGNORE);
    }

private:
    int rank_;
    QueueSink* local_;
    // Workers: MPI ring buffer of byte buffers (header + per-run PDD tail)
    std::vector<std::vector<char>> send_bufs_;
    std::vector<MPI_Request> send_reqs_;
    int slot_ = 0;
    std::vector<int> rej_buf_;
    MPI_Request rej_req_ = MPI_REQUEST_NULL;
};

}  // namespace klasp
