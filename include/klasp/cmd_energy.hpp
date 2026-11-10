#pragma once

#include <string>
#include <vector>

namespace klasp {

// Options for the `energy` subcommand: evaluate or relax one or more
// structures under either the multipole force field or the PET ML potential.
struct EnergyOptions {
    bool relax = false;            // optimise rather than single-point
    std::string backend = "ff";    // "ff" (multipole force field) or "pet"
    int perturb = 0;               // additional randomly perturbed copies
    bool isolated = false;         // vacuum molecules, no PBC (lattice-E reference)
    bool fixed_cell = false;       // hold the cell during --relax
    double pressure_gpa = 0;       // hydrostatic pressure during --relax
    bool no_taper = false;
    int max_order = -1;            // <0 = use the structure's max_rank
    // Negative: use settings.ewald_accuracy from the input.
    double ewald_accuracy = -1;
    double mp_cutoff_ang = 0;      // rank>=2 multipole cutoff, Angstrom; 0 = the pair cutoff
    bool mp_single = false;        // rank>=2 multipole term in single precision
    int iterations = 0;            // 0 = backend default
    double grad_tol = 0;           // 0 = backend default
    std::string output_dir;        // relaxed geometry destination ("" = don't write)
    std::string write_json;        // occ-compatible JSON (ff, single structure)
    bool trajectory = false;       // with --relax: every iteration to <output_dir>/<stem>.traj.xyz (ff)
    std::string summary;           // every structure's results as JSON ("" = don't write)

    // PET-only.
    bool symmetry = false;         // symmetry-constrained relaxation
    bool expand_p1 = false;        // treat every molecule copy as independent
    bool rigid = false;            // rigid-body pre-relax before the full relax
    bool device = true;            // device-resident relaxer
    std::string model = "xs";      // named model ("xs", "mols", ...)
    std::string model_json;        // explicit hypers path (overrides --model)
    std::string model_st;          // explicit weights path (overrides --model)
};

// What the evaluation found. Energies are kJ/mol per molecule (E_cell / n_symops),
// matching what SGWorker stores.
struct EnergyResult {
    int structures = 0;
    int failed = 0;
    // One per structure, in evaluation order (input order plus --perturb copies
    // and expanded batch rows), after relaxation if any. A failed structure gives
    // a quiet NaN so indices line up.
    std::vector<double> energy_kj_per_mol;
};

// Throws when any structure failed.
EnergyResult cmd_energy(const std::vector<std::string>& paths,
                        const EnergyOptions& opts);

}  // namespace klasp
