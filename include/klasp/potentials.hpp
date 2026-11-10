#pragma once

#include "klasp/pair_model.hpp"
#include "klasp/types.hpp"

#include <nlohmann/json.hpp>
#include <map>
#include <string>
#include <vector>

namespace klasp {

// The Buckingham potential as the kernels see it: dense symmetric tables over
// atom types, in atomic units. The recorded JSON is generated from this table
// (potential_table_to_json), not copied from the input, so derived pairs are recorded.
struct PotentialTable {
    std::vector<std::string> type_names;   // index == atom type index
    std::vector<std::string> element_of;   // per type; "" when no entry named one
    std::vector<double> A;                 // (n*n) Hartree
    std::vector<double> B;                 // (n*n) 1/Bohr
    std::vector<double> C;                 // (n*n) Hartree * Bohr^6

    // "type_i,type_j" for each pair filled in by a combining rule rather than
    // read from the input. Empty when the input enumerated everything.
    std::vector<std::string> derived;

    // Entries that could not be read (no `types`, fewer than two, no A, rho or
    // C6, or rho <= 0), for a warning. Entries for unused types are not listed.
    std::vector<std::string> skipped;

    // "type_i,type_j" for each pair left at zero between two types that each have
    // some other pair: listed nowhere and not combinable.
    std::vector<std::string> undefined;

    int n_types() const { return static_cast<int>(type_names.size()); }
    double  a(int i, int j) const { return A[i * n_types() + j]; }
    double  b(int i, int j) const { return B[i * n_types() + j]; }
    double  c(int i, int j) const { return C[i * n_types() + j]; }
    void set(int i, int j, double av, double bv, double cv) {
        const int n = n_types();
        A[i * n + j] = A[j * n + i] = av;
        B[i * n + j] = B[j * n + i] = bv;
        C[i * n + j] = C[j * n + i] = cv;
    }
    // A pair nothing has filled in (zero A and B).
    bool unset(int i, int j) const { return a(i, j) == 0.0 && b(i, j) == 0.0; }

    // Atom type of padded co-crystal sites: one past the real types, with an
    // all-zero row and column so padded sites contribute no energy or force.
    // Not in type_names, so it is never recorded.
    int ghost_type() const { return n_types(); }
};

// Read a `potentials` block into the table, indexed by `type_map`, and fill any
// omitted cross pair by Good-Hope combining of the two self-pairs:
//     A_ij = sqrt(A_ii A_jj),  B_ij = (B_ii + B_jj)/2,  C_ij = sqrt(C_ii C_jj)
// Listed pairs are never overwritten; pairs lacking both self-pairs stay unset.
PotentialTable resolve_potentials(const nlohmann::json& potentials,
                                  const std::map<std::string, int>& type_map);

// The inverse: a `buckingham` list that resolves back to an equal table.
nlohmann::json potential_table_to_json(const PotentialTable& table);

// The only writer of ws.buck_A / buck_B / buck_C / buck_Rmin.
void fill_workspace_potentials(const PotentialTable& table, Workspace& ws);

// Read a `potentials.tabulated` block, e.g.
//     "tabulated": { "n_knots": 1500, "r_min": 1.0, "r_max": 0 }
// Absent or not an object gives `enabled = false`. r_min, r_max in Bohr;
// r_max = 0 means the potential's cutoff.
TabulatedSpec parse_tabulated_spec(const nlohmann::json& potentials);

// Read potentials.damping (default on) and potentials.damping_range (default
// 1.5, must be positive). See TangToenniesDamping.
TangToenniesDamping parse_damping(const nlohmann::json& potentials);

// Write both damping keys back into a `potentials` block.
void record_damping(nlohmann::json& potentials, const TangToenniesDamping& damping);

// Tabulate `table` on the `spec` grid, one spline per ordered type pair, and
// upload it. Samples come from buckingham_untapered (including its short-range
// continuation); the taper is not baked in. The ghost type has no tables and
// evaluates to zero. `cutoff_bohr` is used when spec.r_max is 0.
std::shared_ptr<TabulatedPotentials> tabulate_potentials(const PotentialTable& table,
                                                         const TangToenniesDamping& damping,
                                                         const TabulatedSpec& spec,
                                                         double cutoff_bohr);

// Whether two tables agree to within a relative tolerance, ignoring `derived`.
bool tables_agree(const PotentialTable& x, const PotentialTable& y, double rel_tol = 1e-12);

}  // namespace klasp
