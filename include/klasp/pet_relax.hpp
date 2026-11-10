#pragma once
// PET-MAD (via pet-kokkos) geometry relaxation, the final CSP refinement stage.
//
//   Stage A (rigid): rigid molecules (translation + angle-axis rotation) plus
//     cell strain, driven by a scaled trust-region dogleg (klasp::tr). Skipped
//     when no rigid groups are supplied.
//   Stage B (full): every atomic position plus cell strain, L-BFGS in the ASE
//     UnitCellFilter variables r_i = (I+E) p_i, h = (I+E) h0, whose strain
//     gradient is PET's virial.
//
// Units: Bohr, Hartree, Hartree/Bohr. See pet_bridge.hpp for the bridge to PET.
#include <algorithm>
#include <cmath>
#include <vector>

#include "klasp/pet_bridge.hpp"

namespace klasp {

struct PetRelaxOptions {
    // Stage A (rigid). Runs only if rigid_groups is non-empty.
    int max_iter_rigid = 60;
    // Stage B (full atomic + cell).
    int max_iter_full = 300;

    bool relax_cell = true;

    // Device-resident batched relaxer (PetEvaluator::relax_batch_device). Falls
    // back to the host path for fixed-cell.
    bool device = false;

    // Convergence: max atomic force (Hartree/Bohr) and max |virial|/volume
    // (≈ stress, Hartree/Bohr^3). Tighter than this chases the fp32 force noise.
    double force_tol = 1.0e-3;   // Hartree/Bohr    (~0.05 eV/A)
    double stress_tol = 5.0e-6;  // Hartree/Bohr^3  (~0.15 GPa)

    // Trust-region / max-step caps.
    double max_step = 0.30;      // Bohr per atom coordinate per step
    double max_strain_step = 0.02;

    // Device relaxer: precondition with a stretch-model Hessian built from the
    // initial geometry (mean-diagonal-normalised, gamma supplies the scale). Falls
    // back to scalar L-BFGS scaling above the device DOF cap.
    bool precondition = true;

    // Device relaxer: trust region (see devopt::tr_round) rather than L-BFGS with
    // an Armijo line search. KLASP_PET_LBFGS=1 selects the line search.
    bool trust_region = true;

    // Device relaxer: structures relaxed at once; a finished structure's slot goes
    // to the next waiting one. 0 = all at once (the batch shrinks). Trust region only.
    int max_active = 0;

    // Relative precision of the energies (float32 networks: ~1.2e-7). Energy
    // changes below precision * |E| are ignored by the trust region's accept test.
    double energy_precision = 1.2e-7;

    bool verbose = false;

    // Rigid-body grouping for Stage A: each inner vector lists the atom indices
    // forming one rigid molecule. Empty -> Stage A skipped (full relax only).
    std::vector<std::vector<int>> rigid_groups;
};

// Sum over molecule types of the isolated-molecule PET energy, relaxed in vacuum,
// in kJ/mol per formula unit: E_lattice = E_crystal/fu - reference. The single
// definition of that reference (box padding, iteration cap, fixed cell).
struct BasisConfig;
double pet_isolated_reference(const PetEvaluator& eval, const BasisConfig& basis);

// Shortest interatomic contact under the minimum image, in Bohr. Screens out
// sub-Angstrom contacts before PET, which cannot relax them out. O(N^2), host side.
inline double min_contact_bohr(const CrystalGeometry& s) {
    const int N = static_cast<int>(s.positions.size());
    double hinv[3][3];
    const auto& c = s.cell;
    const double d = c[0][0]*(c[1][1]*c[2][2]-c[1][2]*c[2][1]) -
                     c[0][1]*(c[1][0]*c[2][2]-c[1][2]*c[2][0]) +
                     c[0][2]*(c[1][0]*c[2][1]-c[1][1]*c[2][0]);
    const bool pbc = s.periodic && std::fabs(d) > 1e-12;
    if (pbc) {
        const double id = 1.0/d;
        hinv[0][0]=(c[1][1]*c[2][2]-c[1][2]*c[2][1])*id; hinv[0][1]=(c[0][2]*c[2][1]-c[0][1]*c[2][2])*id;
        hinv[0][2]=(c[0][1]*c[1][2]-c[0][2]*c[1][1])*id; hinv[1][0]=(c[1][2]*c[2][0]-c[1][0]*c[2][2])*id;
        hinv[1][1]=(c[0][0]*c[2][2]-c[0][2]*c[2][0])*id; hinv[1][2]=(c[0][2]*c[1][0]-c[0][0]*c[1][2])*id;
        hinv[2][0]=(c[1][0]*c[2][1]-c[1][1]*c[2][0])*id; hinv[2][1]=(c[0][1]*c[2][0]-c[0][0]*c[2][1])*id;
        hinv[2][2]=(c[0][0]*c[1][1]-c[0][1]*c[1][0])*id;
    }
    double mind = 1e300;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            double dx = s.positions[i][0]-s.positions[j][0];
            double dy = s.positions[i][1]-s.positions[j][1];
            double dz = s.positions[i][2]-s.positions[j][2];
            if (pbc) {  // minimum-image wrap of the fractional difference
                double fx = hinv[0][0]*dx+hinv[0][1]*dy+hinv[0][2]*dz;
                double fy = hinv[1][0]*dx+hinv[1][1]*dy+hinv[1][2]*dz;
                double fz = hinv[2][0]*dx+hinv[2][1]*dy+hinv[2][2]*dz;
                fx -= std::round(fx); fy -= std::round(fy); fz -= std::round(fz);
                dx = c[0][0]*fx+c[0][1]*fy+c[0][2]*fz;
                dy = c[1][0]*fx+c[1][1]*fy+c[1][2]*fz;
                dz = c[2][0]*fx+c[2][1]*fy+c[2][2]*fz;
            }
            mind = std::min(mind, std::sqrt(dx*dx+dy*dy+dz*dz));
        }
    return mind;
}

struct PetRelaxResult {
    CrystalGeometry structure;       // relaxed geometry (Bohr)
    double energy = 0.0;             // Hartree
    double max_force = 0.0;          // Hartree/Bohr (final)
    double max_stress = 0.0;         // Hartree/Bohr^3 (final)
    int rigid_iterations = 0;
    int full_iterations = 0;
    bool converged = false;
    std::vector<double> energy_trace;  // Hartree, one entry per evaluation
};

// Relax `init` under PET. `eval` is a loaded model (reuse across calls).
PetRelaxResult pet_relax(const PetEvaluator& eval, const CrystalGeometry& init,
                         const PetRelaxOptions& opts = {});

// Space-group symmetry for a symmetry-constrained relaxation: only the
// asymmetric-unit atoms carry DOFs and the full cell is regenerated from them
// every step. Operators are fractional: row-major 3x3 rotation + translation.
//
// n_indep independent molecules of sites_of_indep[k] atoms; operator op
// generates one full-cell copy of molecule indep_of_op[op]. Full-cell atoms are
// in (op, site) order at ragged offsets (devopt::SymLayout). Atoms must be
// compacted (no padding slots), and each independent molecule needs an identity
// operator (its reference placement).
struct PetSymmetry {
    int n_indep = 0;               // independent molecules (Z')
    // Real (compacted) atom count of each independent molecule; [n_indep].
    std::vector<int> sites_of_indep;
    int n_op = 0;                  // operators / full-cell molecule copies
    std::vector<double> rot;       // [n_op*9] fractional rotation (row-major 3x3)
    std::vector<double> trans;     // [n_op*3] fractional translation
    std::vector<int> indep_of_op;  // [n_op] which independent molecule each op generates
    int n_asym() const {
        int t = 0;
        for (int ns : sites_of_indep) t += ns;
        return t;
    }
    // Atoms in the full cell: the sum over operators of their molecule's size.
    int n_full() const {
        int t = 0;
        for (int op = 0; op < n_op && op < static_cast<int>(indep_of_op.size()); ++op) {
            const int k = indep_of_op[op];
            if (k >= 0 && k < static_cast<int>(sites_of_indep.size())) t += sites_of_indep[k];
        }
        return t;
    }
    bool valid() const {
        if (n_indep <= 0 || n_op <= 0) return false;
        if (static_cast<int>(indep_of_op.size()) != n_op) return false;
        if (static_cast<int>(sites_of_indep.size()) != n_indep) return false;
        if (rot.size() != static_cast<std::size_t>(n_op) * 9) return false;
        if (trans.size() != static_cast<std::size_t>(n_op) * 3) return false;
        for (int ns : sites_of_indep) if (ns <= 0) return false;
        for (int k : indep_of_op) if (k < 0 || k >= n_indep) return false;
        return has_reference_ops();
    }

    // Every independent molecule has an identity operator among the n_op. Check
    // before make_sym_state, which abandons the state otherwise; P1 inputs with
    // several explicitly placed molecules fail this.
    bool has_reference_ops() const {
        std::vector<bool> seen(static_cast<std::size_t>(n_indep), false);
        for (int op = 0; op < n_op; ++op) {
            bool ident = true;
            for (int i = 0; i < 3 && ident; ++i) {
                if (std::fabs(trans[op * 3 + i]) > 1e-9) ident = false;
                for (int j = 0; j < 3 && ident; ++j)
                    if (std::fabs(rot[op * 9 + i * 3 + j] - (i == j ? 1.0 : 0.0)) > 1e-9)
                        ident = false;
            }
            if (ident) seen[static_cast<std::size_t>(indep_of_op[op])] = true;
        }
        for (bool s : seen) if (!s) return false;
        return true;
    }
};

// Symmetry-constrained relaxation: like pet_relax's Stage B (full atomic + cell)
// but the DOFs are the asymmetric-unit fractional coordinates + a symmetrized
// cell strain. The full cell is rebuilt from the asymmetric unit by the symmetry
// operators each step (symmetry preserved exactly); forces are folded back onto
// the asymmetric unit and the strain gradient is symmetrized over the point
// group. `init` must hold the full cell (all n_op*n_asym atoms, (op,site) order).
PetRelaxResult pet_relax_symmetric(const PetEvaluator& eval, const CrystalGeometry& init,
                                   const PetSymmetry& sym, const PetRelaxOptions& opts = {});

// Batched symmetry-constrained relaxation (one PET pass per iteration over all
// active structures), the symmetry-preserving analogue of pet_relax_batch.
std::vector<PetRelaxResult> pet_relax_sym_batch(const PetEvaluator& eval,
                                                const std::vector<CrystalGeometry>& inits,
                                                const std::vector<PetSymmetry>& syms,
                                                const PetRelaxOptions& opts = {});

// Batched PET relaxation of Stage B: one PET evaluation per iteration over all
// active structures, each advancing its own L-BFGS + cell-filter state.
// Converged structures leave the batch. Structures may be heterogeneous. Steps
// are capped without line search to keep lockstep; a history reset guards
// energy-increasing steps. Returns one result per input, in input order.
std::vector<PetRelaxResult> pet_relax_batch(const PetEvaluator& eval,
                                            const std::vector<CrystalGeometry>& inits,
                                            const PetRelaxOptions& opts = {});

// pet_relax_batch / pet_relax_sym_batch over consecutive chunks of at most
// eval.recommended_batch_atoms() (queried per chunk); an oversized structure
// gets its own chunk. `syms` empty relaxes freely, otherwise one symmetry per
// input. Returns one result per input, in input order.
std::vector<PetRelaxResult> pet_relax_chunked(const PetEvaluator& eval,
                                              const std::vector<CrystalGeometry>& inits,
                                              const std::vector<PetSymmetry>& syms,
                                              const PetRelaxOptions& opts = {});

// Symmetry for a structure rebuilt from its stored rigid-body row, for any Z'.
// Operators are flattened as op*Z' + m, matching expand_rigid_structure and
// build_csp_molecule. `sites_of_indep` are the real per-component atom counts,
// not the padded stride. Returns a symmetry that fails valid() when the pieces
// do not agree.
PetSymmetry expanded_row_symmetry(int space_group, int n_mols,
                                  const std::vector<int>& sites_of_indep);

}  // namespace klasp
