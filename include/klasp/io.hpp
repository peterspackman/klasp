#pragma once

#include "klasp/types.hpp"
#include "klasp/csp_cif.hpp"  // CspCifMetadata, on the CIF writers
#include "klasp/ewald.hpp"   // DEFAULT_EWALD_ACCURACY
#include "klasp/potentials.hpp"
#include "klasp/molecule.hpp"
#include <string>
#include <map>
#include <vector>
#include <array>
#include <nlohmann/json.hpp>

namespace klasp {

struct BasisConfig;

// Configuration loaded from structure JSON.
struct CrystalConfig {
    double cutoff;                                        // table cutoff (Bohr)
    double taper_on;                                      // taper start (Bohr)
    double taper_off;                                     // taper end (Bohr)
    std::map<std::string, int> type_map;
    int max_rank;
    int space_group = 0;                                  // 0 = P1 / unknown
    TangToenniesDamping buck_damping;                     // potentials.damping / damping_range
    nlohmann::json basis_json;                            // for JSON output
    // The molecule type of each independent molecule, by the name the file gave it.
    std::vector<int> molecule_type_index;
};

// The parsed `settings` block of an input; the loaders apply it.
struct InputSettings {
    // Relative truncation target for both Ewald sums.
    RealType ewald_accuracy = DEFAULT_EWALD_ACCURACY;
    // Upper bound on the multipole rank, or -1 for none. Only ever lowers the
    // rank the sites carry.
    int max_rank_cap = -1;
    // One message per problem, ready to print.
    std::vector<std::string> warnings;
};

// Maximum |net cell charge| (e) treated as neutral. klasp applies no
// neutralising background, so a charged cell gives a meaningless Ewald energy.
inline constexpr double NEUTRALITY_TOL_E = 1e-3;

inline double net_site_charge(const std::vector<double>& site_charges) {
    double q = 0;
    for (double c : site_charges) q += c;
    return q;
}

// A NaN charge must not read as neutral.
inline bool cell_is_neutral(const std::vector<double>& site_charges) {
    const double q = net_site_charge(site_charges);
    return q >= -NEUTRALITY_TOL_E && q <= NEUTRALITY_TOL_E;
}

// Read a `settings` block. Never throws: anything unusable becomes a warning and
// leaves the corresponding field at its default.
InputSettings parse_settings(const nlohmann::json& settings);

// Apply an --ewald-accuracy override to a loaded workspace. Negative keeps
// settings.ewald_accuracy. Throws on a value that is not a relative tolerance.
void apply_ewald_accuracy_override(double accuracy, Workspace& ws);

// Load a crystal structure from occ-multpair structure JSON. Geometry is
// converted to Bohr; multipoles stay raw e·Bohr^L; Buckingham A (kJ/mol),
// B (1/Bohr), C (kJ/mol·Bohr^6). Fills mol and ws.
CrystalConfig load_structure_json(const std::string& path,
                                  MoleculeData& mol, Workspace& ws);

// Same, from an already-parsed document.
CrystalConfig load_structure_json(const nlohmann::json& data,
                                  MoleculeData& mol, Workspace& ws);

// molecule_types from either layout: flat at the top level, or nested under
// "basis" in the occ form. Throws if the document has none.
const nlohmann::json& structure_molecule_types(const nlohmann::json& doc);

// Data from a basis-format JSON (no crystal cell/symmetry): body-frame molecule
// data and potentials.
//
// For co-crystals n_mols > 1 and per-molecule data is in mol_*. The flat body_*
// vectors concatenate all molecules, padded to n_sites = max sites per type.
struct BasisConfig {
    double cutoff;          // Bohr
    double taper_on;        // Bohr
    double taper_off;       // Bohr
    std::map<std::string, int> type_map;
    int max_rank;
    int n_sites;            // sites per molecule (max across types for co-crystals)
    int n_types;
    int n_mols = 1;         // number of independent molecule types
    TangToenniesDamping buck_damping;   // potentials.damping / damping_range
    // The resolved Buckingham table as the kernels receive it; basis_json's
    // `potentials.buckingham` is generated from this.
    PotentialTable potentials;

    std::vector<int> mol_n_sites;                              // (n_mols,) actual sites per type
    std::vector<std::vector<std::array<double, 3>>> mol_body_positions;  // (n_mols,)(sites,3)
    std::vector<std::vector<double>> mol_body_charges;                   // (n_mols,)(sites,)
    std::vector<std::vector<int>> mol_body_types;                        // (n_mols,)(sites,)
    std::vector<std::vector<std::array<double, 35>>> mol_body_multipoles;// (n_mols,)(sites,35)
    std::vector<std::vector<std::string>> mol_elements;                  // (n_mols,)(sites,)

    std::vector<std::array<double, 3>> body_positions;   // Bohr
    std::vector<double> body_charges;
    std::vector<int> body_types;
    std::vector<std::array<double, 35>> body_multipoles; // Cartesian

    // Per-site element symbols for CSP radii lookup (flat, first molecule for single-mol)
    std::vector<std::string> elements;

    // Original JSON basis subtree, for round-trip writing.
    nlohmann::json basis_json;

    // Real atoms in the asymmetric unit: the number of PDD rows. Not
    // n_mols * n_sites, which counts padding.
    int n_asym_atoms() const {
        if (n_mols <= 1 || mol_n_sites.empty()) return n_sites;
        int total = 0;
        for (int ns : mol_n_sites) total += ns;
        return total;
    }
};

// Move each H along its bond to the nearest heavy atom within 2 Angstrom by
// `extend_bohr` (negative shortens), as settings.h_bond_extend requires.
// Call per molecule, not on a padded co-crystal array: padded slots sit at the
// body origin and can be picked as a heavy neighbour.
void apply_h_bond_extend(std::vector<std::array<double, 3>>& positions,
                         const std::vector<std::string>& elements,
                         double extend_bohr);

// Load basis-format JSON (molecule definition + potentials, no crystal structure).
BasisConfig load_basis_json(const std::string& path, Workspace& ws);

// The same, from a basis document in memory ({"basis": {...}}). `source` names
// it in errors.
BasisConfig load_basis_document(const nlohmann::json& doc, Workspace& ws,
                                const std::string& source = "the basis document");

// Load multiple basis JSONs for co-crystal CSP.
// Each path defines one molecule type. Potentials are taken from the first file.
// Returns merged BasisConfig with n_mols = paths.size() and per-molecule data.
BasisConfig load_basis_json_multi(const std::vector<std::string>& paths, Workspace& ws);

// The co-crystal merge, from documents in memory. `sources` names each in errors.
BasisConfig load_basis_documents(const std::vector<nlohmann::json>& docs, Workspace& ws,
                                 const std::vector<std::string>& sources = {});

// Build a MoleculeData from basis config + space group symops for CSP.
// For co-crystals (basis.n_mols > 1): n_indep = n_mols, body data padded per type.
MoleculeData build_csp_molecule(const BasisConfig& basis, int space_group);

// Build the crystal-format JSON document that write_crystal_json writes.
// Cell params: a,b,c in Bohr, angles in degrees. Translations: fractional COM
// (n_indep * 3). Rotations: angle-axis (n_indep * 3). mol_types maps each
// independent molecule to its molecule type index.
nlohmann::json build_crystal_json(const BasisConfig& basis,
                                  int space_group,
                                  const double cell_params[6],
                                  const double* translations,
                                  const double* rot_vectors,
                                  int n_indep,
                                  double energy = 0.0);

void write_crystal_json(const std::string& path,
                        const BasisConfig& basis,
                        int space_group,
                        const double cell_params[6],
                        const double* translations,
                        const double* rot_vectors,
                        int n_indep,
                        double energy = 0.0);

// The occ-compatible document write_occ_json writes: basis (molecule types,
// potentials, settings) and crystal (every molecule expanded). Without
// `molecule_type_index`, independent molecule k is written as type k.
nlohmann::json build_occ_json(const BasisConfig& basis,
                              const MoleculeData& mol,
                              double energy_per_mol = 0.0,
                              const std::vector<int>& molecule_type_index = {});

// Write occ-compatible JSON with all molecules expanded (resolved symmetry).
// Includes orientation, parity, and translation for each molecule.
void write_occ_json(const std::string& path,
                    const BasisConfig& basis,
                    const MoleculeData& mol,
                    double energy_per_mol = 0.0,
                    const std::vector<int>& molecule_type_index = {});

// Write a CIF file. Cell params: a,b,c in Bohr, angles in degrees.
// niggli reduces the cell (space groups 1 and 2 only; ignored otherwise) by
// mapping the fractional coordinates of the placed structure into the reduced
// basis. Returns true if the cell was reduced.
bool write_cif(const std::string& path,
               const BasisConfig& basis,
               int space_group,
               const double cell_params[6],
               const double* translations,
               const double* rot_vectors,
               int n_indep,
               double energy = 0.0,
               const CspStructureId& id = {},
               bool niggli = false);

// The same, appended to an open stream. `csp` adds the CCDC CSP dictionary items
// (klasp/csp_cif.hpp); null writes the crystallography alone.
bool write_cif(std::ostream& out,
               const BasisConfig& basis,
               int space_group,
               const double cell_params[6],
               const double* translations,
               const double* rot_vectors,
               int n_indep,
               double energy = 0.0,
               const CspStructureId& id = {},
               bool niggli = false,
               const CspCifMetadata* csp = nullptr);

// Write a structure given as explicit atoms in a cell, in P1.
// `predicted_space_group` is recorded under a klasp data name, not as
// `_space_group_IT_number`, since the operations are already applied.
// cell_params are Angstrom and degrees.
void write_cif_explicit(std::ostream& out,
                        const double cell_params[6],
                        int n_atoms,
                        const int* Z,
                        const double* frac,
                        int predicted_space_group,
                        double energy = 0.0,
                        const CspStructureId& id = {},
                        const CspCifMetadata* csp = nullptr);

// Whether Niggli reduction is meaningful for a space group: its operations must
// be basis-independent, which is true only for P1 and P-1.
inline bool niggli_applicable(int space_group) {
    return space_group == 1 || space_group == 2;
}

}  // namespace klasp
