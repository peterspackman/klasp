#pragma once
// Host-side helper sizing a single streaming call's pool. Free of Kokkos and MPI.

namespace klasp::sched {

// How many pool entries a single streaming call should build: min(P, remaining),
// or P when remaining <= 0 (unbounded). Never rounds up to a batch multiple, so a
// call never overshoots its chunk. P is stream_pool_mult * batch_size.
inline int stream_pool_target(int remaining, int P) {
    if (remaining <= 0) return P;
    return remaining < P ? remaining : P;
}

}  // namespace klasp::sched
