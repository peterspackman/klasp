#include "klasp/csp_cif.hpp"

#include <fmt/format.h>

#include <cctype>

namespace klasp {

std::vector<std::pair<std::string, std::string>> csp_cif_pairs(const CspCifMetadata& meta) {
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(6);

    if (meta.have_density)
        out.emplace_back("_ccdc_csp_density_calc", fmt::format("{:.5f}", meta.density));
    out.emplace_back("_ccdc_csp_simulation_type", meta.simulation_type);
    // Written explicitly: an absent temperature is not the same claim as 0 K.
    out.emplace_back("_ccdc_csp_simulation_temperature",
                     fmt::format("{:.1f}", meta.temperature_k));
    out.emplace_back("_ccdc_csp_structure_clustering_method", meta.clustering_method);
    if (meta.have_energy) {
        out.emplace_back("_ccdc_csp_classification_energy_lattice_absolute",
                         fmt::format("{:.4f}", meta.energy_absolute));
        // Relative to the landscape's global minimum; not derivable from one block.
        out.emplace_back("_ccdc_csp_classification_energy_lattice_relative",
                         fmt::format("{:.4f}", meta.energy_relative));
    }
    return out;
}

CspCifLoop csp_cif_optimisation_loop(const CspCifMetadata& meta) {
    CspCifLoop loop;
    if (meta.steps.empty()) return loop;
    // Always a loop, even for a single step, as the dictionary defines it.
    loop.names = {
        "_ccdc_csp_structure_optimisation_method",
        "_ccdc_csp_structure_optimisation_force_field_name",
        "_ccdc_csp_structure_optimisation_force_field_description",
        "_ccdc_csp_structure_optimisation_software",
        "_ccdc_csp_structure_optimisation_software_version",
        "_ccdc_csp_structure_optimisation_stage",
    };
    loop.rows.reserve(meta.steps.size());
    // "." is CIF for inapplicable.
    auto or_dot = [](const std::string& v) { return v.empty() ? std::string(".") : v; };
    for (const auto& st : meta.steps)
        loop.rows.push_back({or_dot(st.method),
                             or_dot(st.force_field_name),
                             or_dot(st.force_field_description),
                             or_dot(st.software),
                             or_dot(st.software_version),
                             st.final_step ? "F" : "P"});
    return loop;
}

std::string CspStructureId::block_name() const {
    // Block names allow no whitespace or quoting, so the name is flattened.
    std::string sys = name.empty() ? "structure" : name;
    for (char& c : sys)
        if (std::isspace(static_cast<unsigned char>(c)) || c == '\'' || c == '"' ||
            c == '#' || c == '[' || c == ']' || c == '/' || c == '\\')
            c = '_';

    std::string out = sys;
    if (!generator.empty()) out += "-" + generator;
    if (space_group > 0)    out += fmt::format("-{}", space_group);
    if (seed >= 0)          out += fmt::format("-{}", seed);
    if (stage >= 0)         out += fmt::format("-{}", stage);
    return out;
}
}  // namespace klasp
