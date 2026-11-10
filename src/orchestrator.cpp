// The MPI driver for a CSP run: the MPI WorkSource, ResultSink and Comm, plus
// rank 0's drain thread. The run itself is in csp_run.cpp.

#include "klasp/orchestrator.hpp"

#include "klasp/csp_run.hpp"
#include "klasp/exit_code.hpp"
#include "klasp/log.hpp"
#include "klasp/rank0_drainer.hpp"
#include "klasp/result_sender.hpp"
#include "klasp/work_window.hpp"

#include <fmt/base.h>
#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

namespace klasp {

namespace {

class MpiComm : public Comm {
public:
    MpiComm() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks_);
    }

    int rank() const override { return rank_; }
    int size() const override { return nranks_; }

    void barrier() override { MPI_Barrier(MPI_COMM_WORLD); }

    void broadcast(int* v, int n, int root) override {
        MPI_Bcast(v, n, MPI_INT, root, MPI_COMM_WORLD);
    }
    void broadcast(long long* v, int n, int root) override {
        MPI_Bcast(v, n, MPI_LONG_LONG, root, MPI_COMM_WORLD);
    }
    void broadcast(double* v, int n, int root) override {
        MPI_Bcast(v, n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    }

    // The exit rendezvous. Only rank 0 may act on the deadline: it arrives last
    // (after draining and writing the summary), so a rank missing then is
    // wedged, while a worker cannot tell a wedged rank 0 from a busy one.
    // MPI_Abort does not return, so MPI_Win_free never runs on that path.
    void exit_rendezvous(double budget_s) override {
        using Clock = std::chrono::steady_clock;
        MPI_Request done_req = MPI_REQUEST_NULL;
        MPI_Ibarrier(MPI_COMM_WORLD, &done_req);
        const auto t0 = Clock::now();
        for (;;) {
            int done = 0;
            MPI_Test(&done_req, &done, MPI_STATUS_IGNORE);
            if (done) break;
            if (rank_ == 0 &&
                std::chrono::duration<double>(Clock::now() - t0).count() > budget_s) {
                logging::warn("\nA rank did not reach the exit after {:.0f} min. The landscape is\n"
                    "written and closed; ending the job rather than holding the "
                    "allocation.\n", budget_s / 60.0);
                std::fflush(stderr);
                MPI_Abort(MPI_COMM_WORLD, +ExitCode::OutOfTime);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

private:
    int rank_ = 0;
    int nranks_ = 1;
};

}  // namespace

void run_orchestrated_csp(const RunConfig& requested, CspRunOptions options) {
    // Every rank reaches the same verdict, so a refusal here needs no agreement.
    const RunConfig config = prepare_run_config(requested);
    MpiComm comm;
    const int rank = comm.rank();
    const int nranks = comm.size();

    // Rank 0 stores results in this process through a QueueSink.
    ResultQueue queue;
    // Breaks a cycle: the sink needs a recorder that needs the consumer, which is
    // created on the CspRun holding the sink. Nothing calls it before the loop.
    CspRun* runp = nullptr;
    QueueSink local(queue, [&runp](const std::vector<int>& delta) {
        record_rejection_delta(*runp->consumer,
                               runp->config.control.space_groups, delta);
    });
    ResultSender sender(rank, nranks, &local);

    CspRun run(config, comm, queue, sender);
    runp = &run;
    run.options = options;

    csp_setup(run);
    csp_plan_work(run);   // throws RunRefused, on every rank at once

    // The live work counters: a collective allocation initialised from what
    // csp_plan_work decided.
    WorkWindow work(run.targets, run.unit_sizes, rank, nranks,
                    run.seed_start, run.per_sg_batch);
    run.work = &work;

    csp_attach(run);
    csp_start_consumer(run);

    // Rank 0: drain thread for results and rejections (klasp/rank0_drainer.hpp),
    // so worker send rings do not back up while rank 0 computes.
    ResultDrain nothing_in_flight;
    std::unique_ptr<RankZeroDrainer> drainer;
    run.drain = &nothing_in_flight;
    if (rank == 0) {
        drainer = std::make_unique<RankZeroDrainer>(
            queue, *run.consumer, config.control.space_groups,
            ResultSender::TAG_REJ);
        run.drain = drainer.get();
        drainer->start();
    }

    csp_loop(run);
    csp_shutdown(run);
}

}  // namespace klasp
