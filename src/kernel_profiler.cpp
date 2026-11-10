#include "klasp/kernel_profiler.hpp"

#include <Kokkos_Core.hpp>
// Tool-registration APIs; Kokkos_Core does not reliably export them.
#include <impl/Kokkos_Profiling.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

namespace klasp {

KernelProfiler& KernelProfiler::instance() {
    static KernelProfiler p;
    return p;
}

void KernelProfiler::enable(bool sync_gpu) {
    if (enabled_) {
        sync_gpu_ = sync_gpu;
        return;
    }
    sync_gpu_ = sync_gpu;
    enabled_ = true;

    namespace kte = Kokkos::Tools::Experimental;
    // The three launch types are the same event for wall-time purposes.
    kte::set_begin_parallel_for_callback   (&KernelProfiler::begin_kernel);
    kte::set_end_parallel_for_callback     (&KernelProfiler::end_kernel);
    kte::set_begin_parallel_reduce_callback(&KernelProfiler::begin_kernel);
    kte::set_end_parallel_reduce_callback  (&KernelProfiler::end_kernel);
    kte::set_begin_parallel_scan_callback  (&KernelProfiler::begin_kernel);
    kte::set_end_parallel_scan_callback    (&KernelProfiler::end_kernel);
    kte::set_push_region_callback          (&KernelProfiler::push_region);
    kte::set_pop_region_callback           (&KernelProfiler::pop_region);
}

void KernelProfiler::disable() {
    if (!enabled_) return;
    namespace kte = Kokkos::Tools::Experimental;
    kte::set_begin_parallel_for_callback   (nullptr);
    kte::set_end_parallel_for_callback     (nullptr);
    kte::set_begin_parallel_reduce_callback(nullptr);
    kte::set_end_parallel_reduce_callback  (nullptr);
    kte::set_begin_parallel_scan_callback  (nullptr);
    kte::set_end_parallel_scan_callback    (nullptr);
    kte::set_push_region_callback          (nullptr);
    kte::set_pop_region_callback           (nullptr);
    enabled_ = false;
}

void KernelProfiler::reset() {
    std::lock_guard<std::mutex> g(mu_);
    active_kernels_.clear();
    region_stack_.clear();
    stats_.clear();
}

// --- C callbacks forwarded to the singleton --------------------------------

void KernelProfiler::begin_kernel(const char* name, uint32_t /*dev_id*/,
                                   uint64_t* handle) {
    instance().on_begin(name, handle);
}
void KernelProfiler::end_kernel(uint64_t handle) {
    instance().on_end(handle);
}
void KernelProfiler::push_region(const char* name) {
    instance().on_push(name);
}
void KernelProfiler::pop_region() {
    instance().on_pop();
}

// --- Begin / end implementation -------------------------------------------

void KernelProfiler::on_begin(const char* name, uint64_t* handle) {
    // The matching end call finds this row by *handle.
    const uint64_t h = next_handle_.fetch_add(1, std::memory_order_relaxed);
    *handle = h;
    ActiveCall call{name ? name : "<unnamed>", std::chrono::steady_clock::now()};
    std::lock_guard<std::mutex> g(mu_);
    active_kernels_.emplace(h, std::move(call));
}

void KernelProfiler::on_end(uint64_t handle) {
    // Fence outside the lock: a fence can trigger further tool callbacks.
    if (sync_gpu_) Kokkos::fence();

    const auto end = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> g(mu_);
    auto it = active_kernels_.find(handle);
    if (it == active_kernels_.end()) return;   // shouldn't happen

    const double dt = std::chrono::duration<double>(end - it->second.start).count();
    auto& s = stats_[it->second.name];
    if (s.calls == 0) { s.min_s = dt; s.max_s = dt; }
    else              { s.min_s = std::min(s.min_s, dt); s.max_s = std::max(s.max_s, dt); }
    s.total_s += dt;
    ++s.calls;
    active_kernels_.erase(it);
}

void KernelProfiler::on_push(const char* name) {
    ActiveCall call{std::string("[region] ") + (name ? name : "<unnamed>"),
                    std::chrono::steady_clock::now()};
    std::lock_guard<std::mutex> g(mu_);
    region_stack_.push_back(std::move(call));
}

void KernelProfiler::on_pop() {
    // Flush all pending GPU work so the interval reflects device time.
    if (sync_gpu_) Kokkos::fence();
    const auto end = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> g(mu_);
    if (region_stack_.empty()) return;
    auto call = std::move(region_stack_.back());
    region_stack_.pop_back();

    const double dt = std::chrono::duration<double>(end - call.start).count();
    auto& s = stats_[call.name];
    if (s.calls == 0) { s.min_s = dt; s.max_s = dt; }
    else              { s.min_s = std::min(s.min_s, dt); s.max_s = std::max(s.max_s, dt); }
    s.total_s += dt;
    ++s.calls;
}

// --- Summary writer --------------------------------------------------------

void KernelProfiler::write_summary(const std::string& path) const {
    // Snapshot the stats under the lock, then write without holding it.
    std::vector<std::pair<std::string, Stats>> rows;
    {
        std::lock_guard<std::mutex> g(mu_);
        rows.reserve(stats_.size());
        for (const auto& kv : stats_) rows.emplace_back(kv.first, kv.second);
    }
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second.total_s > b.second.total_s; });

    auto dump = [&](std::FILE* f) {
        std::fprintf(f, "kernel,calls,total_ms,avg_ms,min_ms,max_ms\n");
        for (const auto& [name, s] : rows) {
            const double avg_ms = s.calls > 0 ? (s.total_s * 1000.0 / s.calls) : 0.0;
            std::fprintf(f, "\"%s\",%llu,%.3f,%.4f,%.4f,%.4f\n",
                         name.c_str(),
                         static_cast<unsigned long long>(s.calls),
                         s.total_s * 1000.0, avg_ms,
                         s.min_s * 1000.0, s.max_s * 1000.0);
        }
    };

    if (path.empty()) {
        dump(stdout);
        return;
    }
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    dump(f);
    std::fclose(f);
}

}  // namespace klasp
