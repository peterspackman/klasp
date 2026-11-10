#pragma once
// Per-kernel profiler backed by Kokkos Tools callbacks: call count and wall time
// per named kernel and per named region (pushRegion / popRegion).
//
// Call instance().enable() after Kokkos::initialize() and write_summary() on
// rank 0 before Kokkos::finalize(). sync_gpu fences after each kernel so times
// reflect device execution; it serializes the GPU, so use it only for profiling.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace klasp {

class KernelProfiler {
public:
    struct Stats {
        uint64_t calls = 0;
        double total_s = 0;
        double min_s = 0;      // only meaningful once calls > 0
        double max_s = 0;
    };

    static KernelProfiler& instance();

    // Install Kokkos Tools callbacks. Safe to call twice.
    void enable(bool sync_gpu = false);

    // Remove the callbacks. Accumulated stats are kept until reset().
    void disable();

    void reset();

    // Write a summary table sorted by total_s, descending. Empty path: stdout.
    void write_summary(const std::string& path) const;

private:
    KernelProfiler() = default;
    KernelProfiler(const KernelProfiler&) = delete;
    KernelProfiler& operator=(const KernelProfiler&) = delete;

    // Kokkos Tools C callbacks. Forward into the singleton.
    static void begin_kernel(const char* name, uint32_t dev_id, uint64_t* handle);
    static void end_kernel(uint64_t handle);
    static void push_region(const char* name);
    static void pop_region();

    void on_begin(const char* name, uint64_t* handle);
    void on_end(uint64_t handle);
    void on_push(const char* name);
    void on_pop();

    bool enabled_ = false;
    bool sync_gpu_ = false;

    std::atomic<uint64_t> next_handle_{1};

    // Separate state for kernels (handle-tracked) vs regions (stack-tracked).
    mutable std::mutex mu_;
    struct ActiveCall {
        std::string name;
        std::chrono::steady_clock::time_point start;
    };
    std::unordered_map<uint64_t, ActiveCall> active_kernels_;
    std::vector<ActiveCall> region_stack_;

    // Kernels and regions share the map; region names carry a prefix.
    std::unordered_map<std::string, Stats> stats_;
};

}  // namespace klasp
