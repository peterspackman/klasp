#pragma once

// One crystal structure under the force field: energy and relaxation, shared by
// `klasp energy`, klasp-energy and the Python module. Input is the JSON document
// (basis plus crystal). Energies are kJ/mol per molecule; lengths are Angstrom.

#include "klasp/types.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace klasp {

// How the model reads a document, beyond what the document itself says.
struct FFModelOptions {
    int max_order = -1;            // multipole rank; <0 is the rank the multipoles carry
    double ewald_accuracy = -1;    // <0 is the document's settings.ewald_accuracy
    bool no_taper = false;
    double mp_cutoff_ang = 0;      // rank>=2 multipole cutoff, Angstrom; 0 = the pair cutoff
    bool mp_single = false;        // rank>=2 multipole term in single precision
};

struct FFRelaxOptions {
    FFModelOptions model;
    bool fixed_cell = false;
    double pressure_gpa = 0.0;     // hydrostatic; the relaxed energy then includes PV
    int iterations = 300;
    double grad_tol = 1e-8;        // on the scaled gradient, as csp measures it; 0 runs every iteration
    int perturb = 0;               // also relax this many randomly displaced copies
    bool trajectory = false;       // keep a frame per optimiser iteration
};

// The lattice energy split into its two families of terms. Evaluated at a
// geometry, without any pressure term.
struct FFEnergyComponents {
    double total = 0;
    double electrostatic = 0;      // every charge and multipole term, Ewald sums included
    double buckingham = 0;         // repulsion-dispersion
};

// Wall-clock seconds. `step` is optimize minus the per-kernel times. A perturbed
// batch shares everything but `final_structure`.
struct FFTimings {
    double total = 0;
    double load = 0;             // read the document, resolve potentials, allocate
    double initial_energy = 0;   // single point at the input geometry (evaluate: its terms)
    double setup = 0;            // batch workspace, starts, cell degrees of freedom
    double optimize = 0;         // the relaxation
    double per_iteration = 0;    // optimize over the iterations it ran
    double pairs = 0;            // real-space pair kernel, within optimize
    double reciprocal = 0;       // Ewald reciprocal sum, within optimize
    double post = 0;             // self, intramolecular and chain-rule terms, within optimize
    double expand = 0;           // placing atoms from the rigid-body state, within optimize
    double step = 0;             // the rest of optimize
    double final_structure = 0;  // the final document, its geometry and its energy terms
    double trajectory = 0;       // rebuilding each frame's positions
};

std::map<std::string, double> timings_by_name(const FFTimings& t);

// One optimiser iteration at the accepted point, in the optimiser's frame: cell
// and positions are consistent but the cell need not be in canonical orientation.
struct FFFrame {
    int iteration = 0;             // 0 is the starting point, before any step
    double energy = 0;             // the accepted point
    double trial_energy = 0;       // the point this iteration evaluated
    bool accepted = true;          // whether that point became the accepted one
    double grad_max = 0;           // the convergence measure, scaled
    double trust_radius = 0;       // Angstrom
    std::array<double, 9> cell{};  // row-major; the columns are the lattice vectors
    std::vector<double> positions; // three per atom, in atomic_numbers order
};

struct FFStructure {
    std::string label;             // empty for the structure as given; "#k" for perturbed start k
    int space_group = 0;           // as the document states it; 0 when it does not
    int n_molecules = 0;
    int n_atoms = 0;
    std::vector<int> atomic_numbers;

    double initial_energy = 0;     // single point at the input geometry
    double energy = 0;             // after relaxation; initial_energy when there was none
    FFEnergyComponents components; // at the final geometry

    bool relaxed = false;
    bool converged = true;
    int iterations = 0;            // to convergence, or the whole budget if it never converged
    double grad_max = -1;          // <0 when not relaxed
    double trust_radius = 0;       // Angstrom, at exit
    int restarts = 0;

    std::array<double, 6> cell_params{};   // a, b, c, alpha, beta, gamma
    std::array<double, 9> cell{};          // as FFFrame::cell
    std::vector<double> positions;         // as FFFrame::positions
    nlohmann::json structure;              // the final geometry as a document klasp reads
    std::vector<FFFrame> trajectory;       // empty unless asked for
    FFTimings timings;
};

// Convert a structure document to the basis-and-crystal layout (the flat layout is
// rearranged; already-nested documents pass through). A basis with no crystal is
// refused.
nlohmann::json normalize_structure_json(const nlohmann::json& doc);

// Single point.
FFStructure ff_evaluate(const nlohmann::json& doc, const FFModelOptions& opts = {});

// Relax; returns the input structure first, then each perturbed start.
std::vector<FFStructure> ff_relax(const nlohmann::json& doc, const FFRelaxOptions& opts = {});

// Extended XYZ, one frame per trajectory entry.
void write_trajectory_xyz(std::ostream& out, const FFStructure& s);

// The final geometry as explicit atoms in P1.
void write_structure_cif(std::ostream& out, const FFStructure& s, const std::string& name);

}  // namespace klasp
