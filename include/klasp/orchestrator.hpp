#pragma once

// Running a CSP under MPI. The run itself is in csp_run.hpp; run_local_csp
// there is the single-process driver.

#include "klasp/csp_run.hpp"

#include <chrono>

namespace klasp {

// Run a CSP across MPI_COMM_WORLD. MPI_Init_thread must already have been
// called with MPI_THREAD_MULTIPLE (rank 0's drain thread receives concurrently).
// Throws RunRefused, after every rank agrees, when the run will not start.
void run_orchestrated_csp(const RunConfig& config, CspRunOptions options = {});

// Kept for callers that only ever passed the process start time.
inline void run_orchestrated_csp(const RunConfig& config,
                                 std::chrono::steady_clock::time_point process_start) {
    CspRunOptions o;
    o.process_start = process_start;
    run_orchestrated_csp(config, o);
}

}  // namespace klasp
