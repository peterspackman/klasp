#pragma once
// Provenance and classification for a predicted structure, in the data names of
// the CCDC CSP CIF dictionary, version 0.8 (see "CSP Data Guidance for
// CSD-Theory"). _ccdc_csp_free_energy_* and _ccdc_csp_exptl_* are not written.

#include <string>
#include <utility>
#include <vector>

namespace klasp {

// Identity of one predicted structure: the CIF data block name, and the same
// fields written inside the block.
//
//     {name}-{generator}-{space_group}-{seed}-{stage}
//     paracetamol-QR-14-6231-3
//
// Seeds restart in every space group, so the space group must be in the key,
// matching the database's primary key (seed, space_group, stage).
struct CspStructureId {
    std::string name;               // compound, e.g. "paracetamol"
    std::string generator = "QR";   // quasi-random, the only generator so far
    int space_group = 0;
    int seed = -1;                  // < 0 when unknown
    int stage = -1;                 // < 0 when unknown

    // Block name, with characters CIF forbids replaced by '_'. Missing fields
    // are omitted rather than left as empty segments.
    std::string block_name() const;
};

// One step of the optimisation. The dictionary flags the step the reported
// energy belongs to as F and every earlier step as P.
struct CspOptimisationStep {
    std::string method;                    // 'Force field' | 'DFT' | 'Machine learning'
    std::string force_field_name;          // empty -> '.' (inapplicable)
    std::string force_field_description;
    std::string software;
    std::string software_version;
    bool final_step = false;               // the step the energy belongs to
};

struct CspCifMetadata {
    std::vector<CspOptimisationStep> steps;

    // 'None' is the dictionary's value for an unclustered landscape.
    std::string clustering_method = "None";

    double density = 0.0;            // g/cm^3
    double energy_absolute = 0.0;    // kJ/mol, lattice energy
    double energy_relative = 0.0;    // kJ/mol above the landscape's global minimum
    bool have_energy = false;
    bool have_density = false;

    // 'static': zero temperature, no thermal correction. Written explicitly.
    std::string simulation_type = "static";
    double temperature_k = 0.0;
};

// The single-valued items, as (data name, raw value) in dictionary order.
// Values are unquoted; the caller quotes when writing the block.
std::vector<std::pair<std::string, std::string>> csp_cif_pairs(const CspCifMetadata& meta);

// The optimisation-history loop: data names, then one row of raw values per
// step. No rows means no loop should be written.
struct CspCifLoop {
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> rows;
};
CspCifLoop csp_cif_optimisation_loop(const CspCifMetadata& meta);

}  // namespace klasp
