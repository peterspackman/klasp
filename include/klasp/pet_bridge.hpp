#pragma once
// PET-MAD energy/force bridge to pet-kokkos.
//
// klasp side: positions and cell in Bohr, lattice vectors as columns, energies
// in Hartree, forces in Hartree/Bohr. PET side: Angstrom, eV, eV/Angstrom,
// lattice vectors as rows.
#include "klasp/geometry.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace klasp {

// Deprecated alias; use CrystalGeometry.
using PetStructure = CrystalGeometry;

// Result of a (possibly batched) PET evaluation, in klasp atomic units.
struct PetResult {
    std::vector<double> energy;      // [B] total energy, Hartree
    std::vector<double> forces;      // [sum N, 3] flattened, Hartree/Bohr, F = -dE/dx
    // [B, 6] Voigt [xx,yy,zz, xy+yx, xz+zx, yz+zy], Hartree: dE/deps_v in the
    // full-tensor convention (off-diagonals carry both halves); see
    // kPetVirialShearFactor and crystal_dof.hpp.
    std::vector<double> virial;
    std::vector<int>    struct_id;   // [sum N] owning structure per atom
    std::vector<int>    n_atoms;     // [B] atom count per structure (force-array offsets)
};

// PET returns the symmetric virial W = V*stress (W[3] = W_xy); klasp's strain
// DOF wants dE/deps_v, whose shear entries are W_xy + W_yx = 2*W_xy
// (crystal_dof.hpp). The shears are doubled here, at the seam.
constexpr double kPetVirialShearFactor = 2.0;

// Voigt component v scaled from PET's convention to klasp's: shears doubled.
constexpr double pet_virial_voigt_scale(int v) { return v < 3 ? 1.0 : kPetVirialShearFactor; }

// Defined in pet_relax.hpp.
struct PetRelaxOptions;
struct PetRelaxResult;

// Per-structure symmetry data for the device-resident symmetry-constrained relax,
// flattened from the host SymState (make_sym_state). DOFs are the asymmetric
// unit (n_indep*n_sites atoms) + strain; the full cell (n_op*n_sites atoms) is
// regenerated from the operators. valid[b]=0: no identity operator, skip.
struct PetSymBatch {
    std::vector<int> valid, n_op, n_indep;              // [B]
    // Ragged layout per structure; see devopt::SymLayout.
    std::vector<std::vector<int>> sites_of_indep;       // [B][n_indep]
    std::vector<std::vector<int>> asym_off;             // [B][n_indep+1]
    std::vector<std::vector<int>> full_off;             // [B][n_op+1]
    std::vector<std::vector<int>> op_of_atom;           // [B][n_full]
    std::vector<std::vector<double>> x_init;            // [B][3*n_asym + 6]
    std::vector<std::vector<double>> M0, tc, R;         // [B][n_op*9], [n_op*3], [n_op*9]
    std::vector<std::vector<int>> indep_of_op;          // [B][n_op]
};

// Resolve a model spec to a metadata JSON + weights path pair. `spec` may be a
// short alias ("xs", "mols"), a model name ("pet-mad-xs", "pet-mols-s"), or a
// path prefix (<spec>.json + <spec>.safetensors); empty means pet-mad-xs. A
// spec containing '/', or naming files that exist, is used verbatim.
//
// Named models are searched in $KLASP_PET_MODEL_DIR (':'-separated), ./models,
// ., the per-user data dir, the install data dir, then the source tree. Throws,
// listing every directory tried, when not found.
void resolve_pet_model(const std::string& spec, std::string& json_out,
                       std::string& st_out);

// Loads the PET model once and evaluates structures against it. Requires
// Kokkos to be initialised by the caller.
class PetEvaluator {
public:
    // Default model: pet-mad-xs, from the search path.
    PetEvaluator();
    // Explicit model: <model>.json metadata + <model>.safetensors weights.
    PetEvaluator(const std::string& model_json, const std::string& model_safetensors);
    ~PetEvaluator();

    PetEvaluator(PetEvaluator&&) noexcept;
    PetEvaluator& operator=(PetEvaluator&&) noexcept;
    PetEvaluator(const PetEvaluator&) = delete;
    PetEvaluator& operator=(const PetEvaluator&) = delete;

    // Evaluate one structure. forces/virial filled when compute_forces is true.
    PetResult evaluate(const CrystalGeometry& s, bool compute_forces = true) const;

    // Evaluate many structures in one batched pass.
    PetResult evaluate(const std::vector<CrystalGeometry>& structures,
                       bool compute_forces = true) const;

    // Device-resident batched relaxation, full atomic + cell L-BFGS; converged
    // structures drop out. Returns one result per input, in input order.
    // Requires B >= 2. symb != nullptr selects the symmetry-constrained
    // parameterization; nullptr relaxes freely in the reference-cell Cartesian
    // (CellState) parameterization.
    std::vector<PetRelaxResult> relax_batch_device(
        const std::vector<CrystalGeometry>& inits, const PetRelaxOptions& opts,
        const PetSymBatch* symb = nullptr) const;

    // Recommended max total atoms per relaxation sub-batch for this model, to fit
    // device memory. Override with KLASP_PET_MAXATOMS.
    int recommended_batch_atoms() const;

    // Default (pet-mad-xs) model file paths. Throw if not on the search path.
    static std::string default_model_json();
    static std::string default_model_safetensors();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace klasp
