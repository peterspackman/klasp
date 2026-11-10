#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <fmt/format.h>

namespace klasp {

// Per-worker performance recorder. Tracks timing breakdowns per batch.
// Enable with --perf flag. Writes CSV to {output_dir}/perf_rank{R}.csv
class PerfRecorder {
public:
    struct BatchRecord {
        int sg;
        int seed_start;
        int n_seeds;
        int n_valid;           // valid after generation
        int n_kept;            // kept after stage 1 filter
        int n_results;         // final results (finite energy)
        double generate_ms;    // GPU generation time
        double setup_ms;       // batch_setup_cells time
        double stage1_ms;      // charges-only optimization
        double stage2_ms;      // multipole optimization
        double amd_ms;         // AMD descriptor computation
        double powder_ms;      // powder-pattern descriptor computation
        double send_ms;        // result sending time
        double total_ms;       // wall time for entire batch
        // Per-kernel breakdown (stage2 only, populated by asymunit kernel)
        double expand_ms = 0;
        double pairs_ms = 0;
        double recip_ms = 0;
        double post_ms = 0;
        // Stage 3 (optional PET-MAD final relaxation; 0 when --pet-relax is off)
        double stage3_ms = 0;   // wall time of the PET relax + re-score stage
        int pet_n_struct = 0;   // structures relaxed under PET this batch
        int pet_atoms = 0;      // atoms per structure (full cell)
        int pet_evals = 0;      // batched PET energy+force evaluations (lockstep rounds)
    };

    void set_enabled(bool e) { enabled_ = e; }
    bool enabled() const { return enabled_; }

    void record(const BatchRecord& r) {
        if (!enabled_) return;
        records_.push_back(r);
    }

    void write(const std::string& path) const {
        if (!enabled_ || records_.empty()) return;
        std::ofstream f(path);
        f << "sg,seed_start,n_seeds,n_valid,n_kept,n_results,"
             "generate_ms,setup_ms,stage1_ms,stage2_ms,amd_ms,powder_ms,send_ms,total_ms,"
             "gpu_pct,gen_pct,"
             "expand_ms,pairs_ms,recip_ms,post_ms,"
             // Stage 3 (PET) cost, normalised per structure, evaluation and atom.
             "stage3_ms,pet_n_struct,pet_atoms,pet_evals,"
             "pet_ms_per_struct,pet_ms_per_eval,pet_us_per_atom_eval\n";
        for (auto& r : records_) {
            double gpu_ms = r.stage1_ms + r.stage2_ms;
            double gpu_pct = r.total_ms > 0 ? 100.0 * gpu_ms / r.total_ms : 0;
            double gen_pct = r.total_ms > 0 ? 100.0 * r.generate_ms / r.total_ms : 0;
            double pet_per_struct = r.pet_n_struct > 0 ? r.stage3_ms / r.pet_n_struct : 0;
            double pet_per_eval = r.pet_evals > 0 ? r.stage3_ms / r.pet_evals : 0;
            double pet_us_atom = (r.pet_evals > 0 && r.pet_n_struct > 0 && r.pet_atoms > 0)
                ? 1000.0 * r.stage3_ms / (double(r.pet_evals) * r.pet_n_struct * r.pet_atoms) : 0;
            f << fmt::format("{},{},{},{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.1f},{:.1f},"
                             "{:.2f},{:.2f},{:.2f},{:.2f},"
                             "{:.2f},{},{},{},{:.3f},{:.4f},{:.3f}\n",
                             r.sg, r.seed_start, r.n_seeds, r.n_valid, r.n_kept, r.n_results,
                             r.generate_ms, r.setup_ms, r.stage1_ms, r.stage2_ms,
                             r.amd_ms, r.powder_ms, r.send_ms, r.total_ms, gpu_pct, gen_pct,
                             r.expand_ms, r.pairs_ms, r.recip_ms, r.post_ms,
                             r.stage3_ms, r.pet_n_struct, r.pet_atoms, r.pet_evals,
                             pet_per_struct, pet_per_eval, pet_us_atom);
        }
    }

    void print_summary() const {
        if (!enabled_ || records_.empty()) return;
        double total_gen = 0, total_setup = 0, total_s1 = 0, total_s2 = 0;
        double total_amd = 0, total_powder = 0, total_send = 0, total_wall = 0;
        double total_s3 = 0;
        long total_pet_struct = 0, total_pet_evals = 0, total_pet_atomevals = 0;
        int total_seeds = 0, total_valid = 0, total_results = 0;
        for (auto& r : records_) {
            total_gen += r.generate_ms; total_setup += r.setup_ms;
            total_s1 += r.stage1_ms; total_s2 += r.stage2_ms;
            total_amd += r.amd_ms; total_powder += r.powder_ms;
            total_send += r.send_ms;
            total_wall += r.total_ms;
            total_s3 += r.stage3_ms;
            total_pet_struct += r.pet_n_struct;
            total_pet_evals += r.pet_evals;
            total_pet_atomevals += (long)r.pet_evals * r.pet_n_struct * r.pet_atoms;
            total_seeds += r.n_seeds; total_valid += r.n_valid;
            total_results += r.n_results;
        }
        fmt::print("\n--- Performance ({} batches) ---\n", records_.size());
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "generate", total_gen, 100*total_gen/total_wall);
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "setup_cells", total_setup, 100*total_setup/total_wall);
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "stage1", total_s1, 100*total_s1/total_wall);
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "stage2 (FF)", total_s2, 100*total_s2/total_wall);
        if (total_s3 > 0)
            fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "stage3 (PET)", total_s3, 100*total_s3/total_wall);
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "amd", total_amd, 100*total_amd/total_wall);
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "powder", total_powder, 100*total_powder/total_wall);
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "send", total_send, 100*total_send/total_wall);
        double accounted = total_gen + total_setup + total_s1 + total_s2 + total_s3 + total_amd + total_powder + total_send;
        fmt::print("  {:>12s} {:>8.0f} ms  {:>5.1f}%\n", "overhead", total_wall - accounted, 100*(total_wall-accounted)/total_wall);
        fmt::print("  {:>12s} {:>8.0f} ms\n", "TOTAL", total_wall);
        fmt::print("  seeds: {}  valid: {} ({:.0f}%)  results: {}\n",
                   total_seeds, total_valid, 100.0*total_valid/std::max(1,total_seeds), total_results);
        double device_ms = total_gen + total_setup + total_s1 + total_s2 + total_s3 + total_amd + total_powder;
        fmt::print("  GPU utilization: {:.1f}%\n", 100 * device_ms / total_wall);

        // FF stage-2 vs PET stage-3 head-to-head, normalized on the same seeds.
        if (total_s3 > 0 && total_pet_struct > 0) {
            double ff_per_struct = total_results > 0 ? total_s2 / std::max(1L, (long)total_results / 2) : 0;
            // Normalise both stages by the PET structure count.
            double s2_per_struct = total_pet_struct > 0 ? total_s2 / total_pet_struct : 0;
            double s3_per_struct = total_s3 / total_pet_struct;
            double s3_per_eval = total_pet_evals > 0 ? total_s3 / total_pet_evals : 0;
            double s3_us_atom = total_pet_atomevals > 0 ? 1000.0 * total_s3 / total_pet_atomevals : 0;
            fmt::print("\n  --- FF stage-2 vs PET stage-3 (same {} structures) ---\n", total_pet_struct);
            fmt::print("    FF  (stage2): {:8.2f} ms/struct\n", s2_per_struct);
            fmt::print("    PET (stage3): {:8.2f} ms/struct  ({:8.4f} ms/eval, {:.2f} us/atom/eval, {} evals)\n",
                       s3_per_struct, s3_per_eval, s3_us_atom, total_pet_evals);
            if (s2_per_struct > 0)
                fmt::print("    PET / FF cost ratio: {:.1f}x per structure\n", s3_per_struct / s2_per_struct);
            (void)ff_per_struct;
        }
    }

private:
    bool enabled_ = false;
    std::vector<BatchRecord> records_;
};

}  // namespace klasp
