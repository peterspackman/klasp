#pragma once

// Option registration shared between subcommands. Lives beside the commands
// rather than under include/ because it depends on CLI11, which an installed
// klasp does not export.

#include <CLI/CLI.hpp>

#include "klasp/cmd_energy.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/dedup.hpp"
#include "klasp/version.hpp"

#include <cstddef>
#include <exception>
#include <string>
#include <vector>

namespace klasp {

// The build's identity, printed by the startup banner and `--version`.
inline std::string version_banner(const char* program) {
    return std::string(program) + "  " + GIT_HASH_SHORT + (GIT_DIRTY ? "-dirty" : "")
         + "  (" + GIT_BRANCH + ")  [built " + BUILD_DATE + "]";
}

// A paired --x/--no-x flag whose help shows the bound value's current default.
inline CLI::Option* add_switch(CLI::App* sub, const std::string& name,
                               bool& value, const std::string& description) {
    const std::string flags = "--" + name + ",!--no-" + name;
    return sub->add_flag(flags, value,
        description + (value ? "  [on by default]" : "  [off by default]"));
}

// The three ways to name a PET model, registered once.
inline void add_pet_model_options(CLI::App* sub, std::string& model,
                                  std::string& model_json, std::string& model_weights) {
    sub->add_option("--model", model,
        "PET model: a short alias (xs, mols), a model's name in the models directory "
        "(pet-mols-s, pet-mad-s, ...), or a path prefix")
        ->capture_default_str();
    sub->add_option("--model-json", model_json,
        "PET model metadata JSON (overrides --model)");
    sub->add_option("--model-weights", model_weights,
        "PET model safetensors (overrides --model)");
}

// Bind the dedup tolerances on `sub` to `cfg`. Defaults are DedupConfig's member
// initialisers; capture_default_str() reads them for --help.
inline void add_dedup_options(CLI::App* sub, DedupConfig& cfg) {
    sub->add_option("--amd-tol", cfg.amd_tol,
        "AMD L-infinity tolerance (Angstrom). A coarse prefilter ahead of the "
        "powder stage, which does the deciding. Tighten (e.g. 0.1) only with "
        "--no-powder-dedup. <=0 disables the AMD stage.")
        ->capture_default_str();
    sub->add_option("--energy-bin", cfg.energy_bin,
        "Energy bin width (kJ/mol)")
        ->capture_default_str();
    sub->add_option("--density-bin", cfg.density_bin,
        "Density bin width (g/cm^3)")
        ->capture_default_str();
    add_switch(sub, "powder-dedup", cfg.use_powder,
        "The powder stage (mol-cspy's cdtw_cos method), which is what decides "
        "duplicates; AMD alone does not separate a saturated landscape. Off leaves "
        "energy, density and AMD only. Needs the powder descriptor, which prune "
        "reports as missing on load.");
    sub->add_option("--powder-cos-thresh", cfg.powder_cos_thresh,
        "Powder cosine prefilter (candidate if cos_sim >= thresh). mol-cspy's "
        "value, validated against their labels: passes 99.9% of real duplicates "
        "while admitting 0.1% of distinct pairs.")
        ->capture_default_str();
    sub->add_option("--powder-cdtw-thresh", cfg.powder_cdtw_thresh,
        "Powder CDTW L1 threshold: duplicate if min warp distance <= thresh, "
        "<=0 disables the CDTW step and uses cosine only. This is the deciding "
        "stage; mol-cspy's value, validated against their labels on their own "
        "structures: 100% recall, 0.4% false merges. Anything from 2 to 20 works "
        "IF structures are converged -- see csp --grad-tol.")
        ->capture_default_str();
    sub->add_option("--dedup-pdd", cfg.pdd_tol,
        "Aligned-PDD RMS tolerance (Angstrom), a stage after AMD. 5e-3 separates "
        "the same minimum from distinct ones on acetic acid, but only between "
        "converged structures (max gradient <= 1e-7 Ha/Bohr). 0 disables. Needs "
        "the PDD stored (csp --store-pdd).")
        ->capture_default_str();
    sub->add_option("--powder-cdtw-band", cfg.powder_cdtw_band,
        "CDTW Sakoe-Chiba band range (iterative r=0..band-1)")
        ->capture_default_str();
    // Defaults to whatever the caller left in cfg (`csp` off, `prune` on).
    add_switch(sub, "cross-sg", cfg.cross_sg,
        "Compare structures across space groups as well as within them. A "
        "structure found in a subgroup setting, or in a doubled cell, is the "
        "same physical crystal -- the descriptors are symmetry-invariant, so it "
        "is detectable, and prune does it by default. Leaving it off during a "
        "run means the unique counts a run reports are per-space-group and will "
        "be higher than what prune reports for the same landscape.");
}

// Parse a wall-clock duration into seconds: a bare number of minutes ("450"),
// or a value with a unit suffix (30s, 90m, 7.5h). Returns < 0 if the text is
// not a duration.
inline double parse_duration_seconds(const std::string& text) {
    if (text.empty()) return -1;
    double mult = 60.0;                       // bare number = minutes
    std::string num = text;
    switch (text.back()) {
        case 's': case 'S': mult = 1.0;    num.pop_back(); break;
        case 'm': case 'M': mult = 60.0;   num.pop_back(); break;
        case 'h': case 'H': mult = 3600.0; num.pop_back(); break;
        default: break;
    }
    if (num.empty()) return -1;
    try {
        std::size_t used = 0;
        const double v = std::stod(num, &used);
        if (used != num.size() || v < 0) return -1;
        return v * mult;
    } catch (const std::exception&) {
        return -1;
    }
}

// --ewald-accuracy, registered by every subcommand that evaluates an energy.
inline CLI::Option* add_ewald_accuracy_option(CLI::App* sub, double& accuracy) {
    return sub->add_option("--ewald-accuracy", accuracy,
        "Relative truncation target for both Ewald sums, overriding "
        "settings.ewald_accuracy in the input. Puts the real-space cutoff at "
        "sqrt(-ln(a))/kappa and the reciprocal one at 2*kappa*sqrt(-ln(a)). "
        "Must be in (0, 1); negative (the default) means use the input's value. "
        "1e-6 is what the examples ask for and is converged to well under "
        "0.01 kJ/mol; 1e-8 costs roughly 10% more time for roughly 0.001.")
        // Rejected at parse time, so a bad tolerance is a usage error.
        ->check([](const std::string& v) -> std::string {
            double x = 0;
            try { x = std::stod(v); } catch (...) { return "not a number"; }
            if (x < 0 || (x > 0 && x < 1)) return {};
            return "must be a relative tolerance in (0, 1), or negative to use "
                   "the input's value";
        });
}

// Which suspect structures a read-back drops. Shared by `extract` and `prune`;
// defaults come from RowFilter.
inline void add_structure_filter_options(CLI::App* sub, RowFilter& filter) {
    sub->add_flag("--keep-catastrophes", filter.keep_catastrophes,
        "Keep structures whose energy is below --catastrophe-threshold "
        "(default: drop them)");
    sub->add_flag("--keep-warned", filter.keep_warned,
        "Keep structures flagged with warnings, such as a collapsed cell or an "
        "impossible density (default: drop them)");
    sub->add_option("--catastrophe-threshold", filter.catastrophe_threshold,
        "Energy below this is catastrophic (kJ/mol)")
        ->capture_default_str();
}

// Bind the `energy` options on `sub`. Shared by `klasp energy` and the
// klasp-energy binary, which must accept exactly the same arguments.
inline void add_energy_options(CLI::App* sub, EnergyOptions& o,
                               std::vector<std::string>& inputs) {
    sub->add_option("inputs", inputs,
#ifdef KLASP_HAS_PET
        "Structure JSON file(s); with --isolated, basis JSON file(s)")
#else
        "Structure JSON file(s)")
#endif
        ->required();
    sub->add_flag("--relax", o.relax,
        "Optimise each structure instead of a single-point evaluation");
    // Only backends this build can run are offered.
    sub->add_option("--backend", o.backend,
#ifdef KLASP_HAS_PET
        "Energy/force backend: ff (multipole force field) or pet (ML potential)")
        ->capture_default_str()
        ->check(CLI::IsMember({"ff", "pet"}));
#else
        "Energy/force backend: ff (multipole force field). This build has no ML "
        "backend; configure with -DKLASP_ENABLE_PET=ON for `pet`.")
        ->capture_default_str()
        ->check(CLI::IsMember({"ff"}));
#endif
    sub->add_option("-o,--output-dir", o.output_dir,
        "Write each relaxed structure here: JSON and CIF for ff, extended XYZ for pet");
    sub->add_flag("--trajectory", o.trajectory,
        "With --relax: also write every optimiser iteration to <output-dir>/<name>.traj.xyz, "
        "one extended-XYZ frame per iteration (ff)");
    sub->add_option("--summary", o.summary,
        "Write every structure's energies, energy terms, convergence and output files to "
        "this JSON file");
    sub->add_option("--perturb", o.perturb,
        "With --relax: also relax N randomly perturbed copies of the input, to see "
        "which minimum a basin drains to. Single input only.")
        ->capture_default_str();
    sub->add_flag("--fixed-cell", o.fixed_cell,
        "Hold the cell fixed during --relax (relax molecular DOFs only)");
    sub->add_option("--pressure", o.pressure_gpa,
        "Hydrostatic pressure in GPa applied during --relax; the reported energy "
        "then includes PV")
        ->capture_default_str();
    sub->add_option("--iterations", o.iterations,
        "Optimizer iteration cap (0 = backend default)")
        ->capture_default_str();
    sub->add_option("--grad-tol", o.grad_tol,
        "Convergence tolerance: max |gradient| (Ha/Bohr). 0 = backend default "
        "(1e-8 for ff, 1e-3 for pet, whose fp32 forces plateau near 2e-4).")
        ->capture_default_str();
    sub->add_flag("--no-taper", o.no_taper, "Disable energy tapering (ff)");
    sub->add_option("--max-order", o.max_order,
        "Override multipole rank, 0 to 4 (ff). Interactions are kept to "
        "lA + lB <= rank, so 4 truncates at 1/R^5 rather than summing every "
        "pair of rank-4 moments. -1 uses whatever rank the basis carries.")
        ->check(CLI::Range(-1, 4))
        ->capture_default_str();
    add_ewald_accuracy_option(sub, o.ewald_accuracy);
    sub->add_option("--mp-cutoff", o.mp_cutoff_ang,
        "Shorter cutoff (Angstrom) for the rank>=2 multipole term, tapered to zero "
        "over its last 2 A (ff). The same setting csp takes; here so its energy "
        "cost can be measured on single points. 0 = the pair cutoff. Not for a "
        "ranking stage: 14 A moves paracetamol energies by up to 0.77 kJ/mol and "
        "flips 83 of 1770 pairwise orderings.")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    sub->add_flag("--mp-single", o.mp_single,
        "Run the rank>=2 multipole term in single precision (ff). The same "
        "setting csp takes; here so its energy cost can be measured on single "
        "points. A --relax under it has no precision switch and cannot certify "
        "convergence below about 3e-7.");
    sub->add_option("--write-json", o.write_json,
        "Write occ-compatible JSON with expanded molecules (ff, single input)");
#ifdef KLASP_HAS_PET
    sub->add_flag("--isolated", o.isolated,
        "PET vacuum reference: relax each molecule type with no PBC and report the "
        "sum, which is what a PET lattice energy is measured against. Takes basis JSON.");
    sub->add_flag("--symmetry", o.symmetry,
        "Relax under the space-group operators, moving only the asymmetric unit (pet)");
    sub->add_flag("--expand-p1", o.expand_p1,
        "Treat every molecule copy as independent under P1 (pet, --symmetry testing)");
    sub->add_flag("--rigid", o.rigid,
        "Rigid-molecule pre-relaxation before the full relax (pet, without --symmetry)");
    sub->add_flag("!--host", o.device,
        "Use the host relaxer instead of the device-resident one (pet)");
    add_pet_model_options(sub, o.model, o.model_json, o.model_st);
#endif
}

// The one-line description, so both front ends present the command the same way.
inline const char* energy_description() {
    return "Evaluate or relax structures (multipole force field or PET ML potential)";
}

}  // namespace klasp
