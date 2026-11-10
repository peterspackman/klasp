// CIF output is parsed back with gemmi and checked for meaning, not string form.

#include <catch2/catch_test_macros.hpp>

#include "klasp/csp_cif.hpp"
#include "klasp/io.hpp"

#include <gemmi/cifdoc.hpp>
#include <gemmi/cif.hpp>

#include <sstream>
#include <string>
#include <vector>

using namespace klasp;

namespace {

gemmi::cif::Document parse(const std::string& text) {
    return gemmi::cif::read_memory(text.data(), text.size(), "string");
}

std::string value(const gemmi::cif::Block& b, const std::string& tag) {
    const std::string* v = const_cast<gemmi::cif::Block&>(b).find_value(tag);
    return v ? gemmi::cif::as_string(*v) : std::string();
}

CspCifMetadata two_stage_metadata() {
    CspCifMetadata m;
    CspOptimisationStep ff;
    ff.method = "Force field";
    ff.force_field_name = "klasp";
    ff.force_field_description = "atom-atom exp-6 with distributed multipoles";
    ff.software = "klasp";
    ff.software_version = "abc1234";
    ff.final_step = false;
    m.steps.push_back(ff);
    CspOptimisationStep ml;
    ml.method = "Machine learning potential";
    ml.force_field_name = "PET-MAD";
    ml.force_field_description = "equivariant transformer, symmetry-constrained";
    ml.software = "klasp";
    ml.software_version = "abc1234";
    ml.final_step = true;
    m.steps.push_back(ml);
    m.clustering_method = "klasp online dedup: energy 5.0 kJ/mol bins, powder CDTW 10.0";
    m.density = 1.2345;
    m.have_density = true;
    m.energy_absolute = -83.5;
    m.energy_relative = 2.25;
    m.have_energy = true;
    return m;
}

}  // namespace

TEST_CASE("explicit geometry is written in P1 and says so", "[cif]") {
    const double cell[6] = {7.1, 8.2, 9.3, 90.0, 97.5, 90.0};
    const int Z[3] = {6, 7, 8};
    // Outside [0,1): a molecule is kept whole across a cell face.
    const double frac[9] = {0.1, 0.2, 0.3, 1.15, -0.05, 0.55, 0.4, 0.5, 0.6};

    std::ostringstream out;
    // Predicted in P2_1/c; its four operations are already applied to these coordinates.
    write_cif_explicit(out, cell, 3, Z, frac, /*predicted_space_group=*/14,
                       /*energy=*/-123.456, CspStructureId{"test", "QR", 14, 7, 2});

    auto doc = parse(out.str());
    REQUIRE(doc.blocks.size() == 1);
    const auto& b = doc.blocks.front();

    CHECK(value(b, "_space_group_IT_number") == "1");
    CHECK(value(b, "_space_group_name_H-M_alt") == "P 1");
    CHECK(value(b, "_klasp_predicted_space_group_IT_number") == "14");
    CHECK(value(b, "_klasp_predicted_space_group_name_H-M") == "P 1 21/c 1");
    CHECK(const_cast<gemmi::cif::Block&>(b).find_loop("_space_group_symop_operation_xyz").length() == 0);

    auto labels = const_cast<gemmi::cif::Block&>(b).find_loop("_atom_site_label");
    REQUIRE(labels.length() == 3);
    auto ys = const_cast<gemmi::cif::Block&>(b).find_loop("_atom_site_fract_y");
    REQUIRE(ys.length() == 3);
    CHECK(std::stod(ys[1]) == -0.05);
    auto xs = const_cast<gemmi::cif::Block&>(b).find_loop("_atom_site_fract_x");
    CHECK(std::stod(xs[1]) == 1.15);
}

TEST_CASE("CSP provenance survives a round trip through a CIF parser", "[cif]") {
    const double cell[6] = {7.1, 8.2, 9.3, 90.0, 97.5, 90.0};
    const int Z[2] = {6, 8};
    const double frac[6] = {0.0, 0.0, 0.0, 0.5, 0.5, 0.5};
    const CspCifMetadata m = two_stage_metadata();

    std::ostringstream out;
    write_cif_explicit(out, cell, 2, Z, frac, 14, -83.5,
                       CspStructureId{"test", "QR", 14, 1, 2}, &m);
    auto doc = parse(out.str());
    REQUIRE(doc.blocks.size() == 1);
    auto& b = doc.blocks.front();

    CHECK(value(b, "_ccdc_csp_simulation_type") == "static");
    CHECK(value(b, "_ccdc_csp_simulation_temperature") == "0.0");
    CHECK(std::stod(value(b, "_ccdc_csp_density_calc")) == 1.2345);
    CHECK(std::stod(value(b, "_ccdc_csp_classification_energy_lattice_absolute")) == -83.5);
    CHECK(std::stod(value(b, "_ccdc_csp_classification_energy_lattice_relative")) == 2.25);
    CHECK(value(b, "_ccdc_csp_structure_clustering_method") == m.clustering_method);

    // Both optimisation steps, in order, exactly one flagged final.
    auto methods = b.find_loop("_ccdc_csp_structure_optimisation_method");
    auto stages  = b.find_loop("_ccdc_csp_structure_optimisation_stage");
    auto descs   = b.find_loop("_ccdc_csp_structure_optimisation_force_field_description");
    REQUIRE(methods.length() == 2);
    REQUIRE(stages.length() == 2);
    CHECK(gemmi::cif::as_string(methods[0]) == "Force field");
    CHECK(gemmi::cif::as_string(methods[1]) == "Machine learning potential");
    CHECK(gemmi::cif::as_string(stages[0]) == "P");
    CHECK(gemmi::cif::as_string(stages[1]) == "F");
    CHECK(gemmi::cif::as_string(descs[0]) == "atom-atom exp-6 with distributed multipoles");
}

TEST_CASE("a missing optimisation field is written as inapplicable", "[cif]") {
    CspCifMetadata m;
    CspOptimisationStep step;
    step.method = "DFT";
    step.software = "klasp";
    step.final_step = true;      // force field name and description left empty
    m.steps.push_back(step);

    const CspCifLoop loop = csp_cif_optimisation_loop(m);
    REQUIRE(loop.rows.size() == 1);
    REQUIRE(loop.rows[0].size() == loop.names.size());
    CHECK(loop.rows[0][0] == "DFT");
    CHECK(loop.rows[0][1] == ".");
    CHECK(loop.rows[0][2] == ".");
    CHECK(loop.rows[0][5] == "F");

    const double cell[6] = {5, 5, 5, 90, 90, 90};
    const int Z[1] = {6};
    const double frac[3] = {0, 0, 0};
    std::ostringstream out;
    write_cif_explicit(out, cell, 1, Z, frac, 1, 0.0,
                       CspStructureId{"test", "QR", 1, 0, 2}, &m);
    auto doc = parse(out.str());
    auto& b = doc.blocks.front();
    CHECK(b.find_loop("_ccdc_csp_structure_optimisation_method").length() == 1);
    CHECK(b.find_loop("_ccdc_csp_structure_optimisation_stage").length() == 1);
}

TEST_CASE("energy items are absent when there is no landscape to rank against",
          "[cif]") {
    CspCifMetadata m;                 // have_energy / have_density both false
    const auto pairs = csp_cif_pairs(m);
    for (const auto& [name, v] : pairs) {
        (void)v;
        CHECK(name.find("energy") == std::string::npos);
        CHECK(name != "_ccdc_csp_density_calc");
    }
    // Simulation conditions are still stated without a landscape.
    bool has_type = false;
    for (const auto& [name, v] : pairs) {
        (void)v;
        if (name == "_ccdc_csp_simulation_type") has_type = true;
    }
    CHECK(has_type);
}

TEST_CASE("a structure id keys a CIF block back to its database row", "[cif]") {
    CspStructureId id{"paracetamol", "QR", 14, 6231, 3};
    CHECK(id.block_name() == "paracetamol-QR-14-6231-3");

    SECTION("the space group separates rows the seed does not") {
        // Seeds restart per space group: same seed, different crystals.
        CspStructureId a{"urea", "QR", 2, 17, 2};
        CspStructureId b{"urea", "QR", 14, 17, 2};
        CHECK(a.block_name() != b.block_name());
    }

    SECTION("the stage separates a structure from its own relaxation") {
        CspStructureId ff{"urea", "QR", 14, 17, 2};
        CspStructureId ml{"urea", "QR", 14, 17, 3};
        CHECK(ff.block_name() != ml.block_name());
    }

    SECTION("a name CIF cannot spell is flattened, not rejected") {
        CspStructureId spaced{"acetic acid", "QR", 14, 1, 2};
        CHECK(spaced.block_name() == "acetic_acid-QR-14-1-2");
        CspStructureId quoted{"a'b\"c", "QR", 14, 1, 2};
        CHECK(quoted.block_name() == "a_b_c-QR-14-1-2");
    }

    SECTION("missing fields collapse rather than leaving empty segments") {
        CspStructureId bare{"urea", "QR", 0, -1, -1};
        CHECK(bare.block_name() == "urea-QR");
        CspStructureId anon{"", "QR", 14, 1, 2};
        CHECK(anon.block_name() == "structure-QR-14-1-2");
    }
}
