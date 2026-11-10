// One structure under the force field, through the layer `klasp energy`,
// klasp-energy and the Python module share (klasp/ff_structure.hpp).

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "klasp/cmd_energy.hpp"
#include "klasp/ff_structure.hpp"
#include "klasp/io.hpp"
#include "klasp/log.hpp"
#include "klasp/molecule.hpp"
#include "klasp/types.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace klasp;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using json = nlohmann::json;

namespace {

std::filesystem::path example_path(const char* name) {
    return std::filesystem::path(KLASP_SOURCE_DIR) / "example-inputs" / name;
}

json read_json(const std::filesystem::path& path) {
    std::ifstream f(path);
    REQUIRE(f.is_open());
    return json::parse(f);
}

json example(const char* name) { return read_json(example_path(name)); }

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    REQUIRE(a.size() == b.size());
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

}  // namespace

TEST_CASE("an energy is the sum of its electrostatic and Buckingham terms", "[ff-structure]") {
    const FFStructure s = ff_evaluate(example("ethylcarbamate.json"));
    CHECK(s.n_molecules == 2);
    CHECK(s.n_atoms == 26);
    CHECK(s.atomic_numbers.size() == 26u);
    CHECK(s.positions.size() == 3u * 26u);
    REQUIRE(std::isfinite(s.energy));
    CHECK(s.components.buckingham < 0.0);
    CHECK(s.components.electrostatic < 0.0);
    CHECK_THAT(s.components.electrostatic + s.components.buckingham,
               WithinAbs(s.energy, 1e-9));
}

// A loose Ewald accuracy in basis.settings is the easiest setting to see.
TEST_CASE("settings in a basis-and-crystal document reach the model", "[ff-structure]") {
    json doc = example("ethylcarbamate.json");
    const double as_given = ff_evaluate(doc).energy;
    doc["basis"]["settings"]["ewald_accuracy"] = 1e-2;
    CHECK(std::abs(ff_evaluate(doc).energy - as_given) > 1e-3);
}

TEST_CASE("a CSP basis is refused as a structure, by name", "[ff-structure]") {
    CHECK_THROWS_WITH(ff_evaluate(example("acetic_acid.json")), ContainsSubstring("basis"));
}

TEST_CASE("a relaxation's trajectory runs from the input to the answer",
          "[ff-structure][trajectory]") {
    FFRelaxOptions o;
    o.trajectory = true;
    const auto results = ff_relax(example("ethylcarbamate.json"), o);
    REQUIRE(results.size() == 1);
    const FFStructure& s = results.front();
    REQUIRE(s.converged);
    const auto& t = s.trajectory;
    REQUIRE(t.size() >= 2);

    // Frame 0 is the input, before any step; the last frame is what came back.
    CHECK(t.front().iteration == 0);
    CHECK(t.back().iteration == static_cast<int>(t.size()) - 1);
    CHECK_THAT(t.front().energy, WithinAbs(s.initial_energy, 1e-6));
    CHECK(t.back().energy == s.energy);
    CHECK(max_abs_diff(t.back().positions, s.positions) < 1e-9);

    // Every phase took some time, and none took longer than the whole call.
    CHECK(s.timings.optimize > 0.0);
    CHECK(s.timings.pairs > 0.0);
    CHECK(s.timings.trajectory > 0.0);
    CHECK(s.timings.total >= s.timings.load + s.timings.optimize + s.timings.trajectory);

    // An accepted step does not raise the energy beyond its precision; a
    // rejected one leaves the structure exactly where it was.
    int accepted = 0;
    for (size_t f = 1; f < t.size(); ++f) {
        INFO("frame " << f);
        if (t[f].accepted) {
            ++accepted;
            CHECK(t[f].energy <= t[f - 1].energy + 1e-12 * std::max(1.0, std::abs(t[f - 1].energy)));
        } else {
            CHECK(t[f].energy == t[f - 1].energy);
            CHECK(t[f].positions == t[f - 1].positions);
        }
    }
    CHECK(accepted > 0);
}

TEST_CASE("recording a trajectory does not change the relaxation", "[ff-structure][trajectory]") {
    const json doc = example("ethylcarbamate.json");
    FFRelaxOptions recorded;
    recorded.trajectory = true;
    const FFStructure plain = ff_relax(doc).front();
    const FFStructure with = ff_relax(doc, recorded).front();
    CHECK(plain.trajectory.empty());
    CHECK(plain.energy == with.energy);
    CHECK(plain.iterations == with.iterations);
    CHECK(plain.positions == with.positions);
}

// The optimiser leaves the cell in an arbitrary orientation while a document
// stores only lengths and angles, so the writer must rotate orientations to the
// canonical frame. Tight Ewald accuracy, because relaxed and re-evaluated
// energies agree only to about that accuracy.
TEST_CASE("a relaxed structure reads back at the energy it was relaxed to", "[ff-structure]") {
    FFRelaxOptions o;
    o.model.ewald_accuracy = 1e-10;
    for (const char* name : {"ethylcarbamate.json", "paracetamol.json"}) {
        INFO(name);
        const FFStructure relaxed = ff_relax(example(name), o).front();
        REQUIRE(relaxed.converged);
        CHECK_THAT(ff_evaluate(relaxed.structure, o.model).energy, WithinAbs(relaxed.energy, 1e-6));
        CHECK_THAT(relaxed.components.total, WithinAbs(relaxed.energy, 1e-6));
    }
}

TEST_CASE("perturbed starts come back after the structure as given", "[ff-structure]") {
    FFRelaxOptions o;
    o.perturb = 2;
    const auto r = ff_relax(example("ethylcarbamate.json"), o);
    REQUIRE(r.size() == 3);
    CHECK(r[0].label.empty());
    CHECK(r[1].label == "#1");
    CHECK(r[2].label == "#2");
    for (const auto& s : r) CHECK(s.initial_energy == r[0].initial_energy);
}

TEST_CASE("energy writes the relaxed structure, a CIF, the trajectory and a summary",
          "[ff-structure][energy]") {
    const auto dir = std::filesystem::temp_directory_path() / "klasp_test_ff_structure";
    std::filesystem::remove_all(dir);
    EnergyOptions o;
    o.relax = true;
    o.trajectory = true;
    o.ewald_accuracy = 1e-10;   // see the read-back test above
    o.output_dir = dir.string();
    o.summary = (dir / "summary.json").string();
    const EnergyResult r = cmd_energy({example_path("ethylcarbamate.json").string()}, o);
    REQUIRE(r.structures == 1);

    const json summary = read_json(o.summary);
    const json& s = summary.at("structures").at(0);
    CHECK(s.at("energy").get<double>() == r.energy_kj_per_mol.at(0));
    CHECK(s.at("converged").get<bool>());
    CHECK(s.contains("components"));
    CHECK(s.at("timings").at("total").get<double>() > 0.0);
    for (const char* kind : {"structure", "cif", "trajectory"})
        CHECK(std::filesystem::exists(s.at("outputs").at(kind).get<std::string>()));

    // One XYZ frame per recorded iteration.
    std::ifstream xyz(s.at("outputs").at("trajectory").get<std::string>());
    std::string line;
    int lines = 0;
    while (std::getline(xyz, line)) ++lines;
    const int per_frame = s.at("n_atoms").get<int>() + 2;
    CHECK(lines % per_frame == 0);
    CHECK(lines / per_frame == s.at("trajectory_frames").get<int>());

    // The written structure is an input in its own right, at the same energy.
    FFModelOptions tight;
    tight.ewald_accuracy = o.ewald_accuracy;
    const FFStructure back =
        ff_evaluate(read_json(s.at("outputs").at("structure").get<std::string>()), tight);
    CHECK_THAT(back.energy, WithinAbs(r.energy_kj_per_mol.at(0), 1e-6));

    std::filesystem::remove_all(dir);
}

TEST_CASE("energy refuses a trajectory it could not record", "[ff-structure][energy]") {
    const std::string input = example_path("ethylcarbamate.json").string();
    EnergyOptions o;
    o.trajectory = true;
    o.output_dir = (std::filesystem::temp_directory_path() / "klasp_test_ff_unused").string();
    CHECK_THROWS_WITH(cmd_energy({input}, o), ContainsSubstring("--relax"));
    o.relax = true;
    o.output_dir.clear();
    CHECK_THROWS_WITH(cmd_energy({input}, o), ContainsSubstring("--output-dir"));
}

// The loader's placement reads only `orientation` and `parity`; documents
// written with rotation matrices must still come back rotated.
TEST_CASE("single-point positions come from the rigid-body state", "[ff-structure]") {
    const FFStructure relaxed = ff_relax(example("ethylcarbamate.json")).front();
    REQUIRE(relaxed.structure.at("crystal").at("molecules").at(0).contains("rotation"));
    FFRelaxOptions one;
    one.iterations = 1;
    one.trajectory = true;
    const FFStructure from_state = ff_relax(relaxed.structure, one).front();
    CHECK(max_abs_diff(ff_evaluate(relaxed.structure).positions,
                       from_state.trajectory.front().positions) < 1e-9);
}

TEST_CASE("a relaxation truncates multipoles at the rank its single point uses",
          "[ff-structure]") {
    for (int rank : {1, 3}) {
        INFO("rank " << rank);
        FFRelaxOptions o;
        o.model.max_order = rank;
        o.iterations = 1;
        o.trajectory = true;
        const FFStructure s = ff_relax(example("ethylcarbamate.json"), o).front();
        CHECK_THAT(s.trajectory.front().energy, WithinAbs(s.initial_energy, 1e-6));
    }
}

// Co-crystal molecules not listed in type order keep their types.
TEST_CASE("written molecules keep the types they were given", "[ff-structure]") {
    json doc = example("ethylcarbamate.json");
    json second = doc["basis"]["molecule_types"][0];
    second["name"] = "second";
    doc["basis"]["molecule_types"].push_back(second);
    doc["crystal"]["molecules"][0]["type"] = "second";
    const FFStructure s = ff_evaluate(doc);
    const json& written = s.structure.at("crystal").at("molecules");
    REQUIRE(written.size() == 2u);
    CHECK(written[0].at("type") == "second");
    CHECK(written[1].at("type") == doc["crystal"]["molecules"][1]["type"]);
}

TEST_CASE("a structure the loader cannot represent is refused, not guessed at",
          "[ff-structure]") {
    json doc = example("ethylcarbamate.json");
    SECTION("a molecule of a type the basis does not define") {
        doc["crystal"]["molecules"][0]["type"] = "nope";
        CHECK_THROWS_WITH(ff_evaluate(doc), ContainsSubstring("'nope'"));
    }
    SECTION("an improper rotation beside a symmetry list") {
        doc["crystal"]["molecules"][0]["rotation"] =
            json::array({json::array({-1.0, 0.0, 0.0}), json::array({0.0, -1.0, 0.0}),
                         json::array({0.0, 0.0, -1.0})});
        doc["symmetry"] = json::array({json{{"op", "x,y,z"}, {"molecule", 0}},
                                       json{{"op", "x,y,z"}, {"molecule", 1}}});
        CHECK_THROWS_WITH(ff_evaluate(doc), ContainsSubstring("improper"));
    }
    SECTION("a symmetry entry copying a molecule that is not there") {
        doc["symmetry"] = json::array({json{{"op", "x,y,z"}, {"molecule", 0}},
                                       json{{"op", "x,y,z"}, {"molecule", 5}}});
        CHECK_THROWS_WITH(ff_evaluate(doc), ContainsSubstring("molecule 5"));
    }
    SECTION("more atoms than the workspace holds") {
        auto ws = allocate_workspace(8, 27, 100, 32, 4);
        MoleculeData mol{};
        CHECK_THROWS_WITH(load_structure_json(doc, mol, ws), ContainsSubstring("atoms"));
    }
}

TEST_CASE("a batch document in the basis-and-crystal layout evaluates each entry",
          "[ff-structure][energy]") {
    json doc = example("ethylcarbamate.json");
    json stretched = doc["crystal"]["cell"];
    stretched["a"] = stretched["a"].get<double>() * 1.02;
    doc["structures"] = json::array(
        {json{{"name", "as_given"}, {"cell", doc["crystal"]["cell"]},
              {"molecules", doc["crystal"]["molecules"]}},
         json{{"name", "stretched"}, {"cell", stretched},
              {"molecules", doc["crystal"]["molecules"]}}});
    const auto path = std::filesystem::temp_directory_path() / "klasp_test_batch_layered.json";
    {
        std::ofstream f(path);
        f << doc.dump();
    }
    const EnergyResult r = cmd_energy({path.string()}, EnergyOptions{});
    std::filesystem::remove(path);
    REQUIRE(r.energy_kj_per_mol.size() == 2u);
    CHECK_THAT(r.energy_kj_per_mol[0],
               WithinAbs(ff_evaluate(example("ethylcarbamate.json")).energy, 1e-6));
    CHECK(std::abs(r.energy_kj_per_mol[1] - r.energy_kj_per_mol[0]) > 1e-3);
}

// What a document leaves undefined is said, by name, rather than guessed at.
TEST_CASE("a document's undefined parts are warned about", "[ff-structure]") {
    json doc = example("ethylcarbamate.json");
    json& type = doc["basis"]["molecule_types"][0];
    type["sites"][0].erase("type");
    type["sites"][1]["element"] = "Qq";
    type["sites"][2]["multipoles"]["dipole"] = json::array({0.1, 0.2});
    doc["basis"]["molecule_types"].push_back(type);
    logging::Capture capture;
    ff_evaluate(doc);
    CHECK_THAT(capture.text(), ContainsSubstring("site 0"));
    CHECK_THAT(capture.text(), ContainsSubstring("has no type"));
    CHECK_THAT(capture.text(), ContainsSubstring("'Qq'"));
    CHECK_THAT(capture.text(), ContainsSubstring("dipole of 2 components"));
    CHECK_THAT(capture.text(), ContainsSubstring("more than once"));
}

// An inverted molecule in a file with no symmetry list is placed at -t + w. A
// lattice-equivalent translation must give the same energy and relax the same way.
TEST_CASE("an inverted molecule given outside the cell relaxes like one inside it",
          "[ff-structure]") {
    FFRelaxOptions o;
    o.iterations = 60;
    const FFStructure relaxed = ff_relax(example("ethylcarbamate.json"), o).front();
    REQUIRE(relaxed.converged);

    json shifted = relaxed.structure;
    int inverted_molecule = -1;
    for (int k = 0; k < static_cast<int>(shifted["crystal"]["molecules"].size()); ++k) {
        const json& r = shifted["crystal"]["molecules"][k]["rotation"];
        const double det = r[0][0].get<double>() * (r[1][1].get<double>() * r[2][2].get<double>() - r[1][2].get<double>() * r[2][1].get<double>())
                         - r[0][1].get<double>() * (r[1][0].get<double>() * r[2][2].get<double>() - r[1][2].get<double>() * r[2][0].get<double>())
                         + r[0][2].get<double>() * (r[1][0].get<double>() * r[2][1].get<double>() - r[1][1].get<double>() * r[2][0].get<double>());
        if (det < 0) inverted_molecule = k;
    }
    REQUIRE(inverted_molecule >= 0);
    json& t = shifted["crystal"]["molecules"][inverted_molecule]["translation"];
    for (auto& x : t) x = x.get<double>() - 1.0;

    const double from_document = ff_evaluate(relaxed.structure).energy;
    CHECK_THAT(ff_evaluate(shifted).energy, WithinAbs(from_document, 1e-8));
    const FFStructure again = ff_relax(shifted, o).front();
    CHECK(again.converged);
    // Against the document's own energy: a relaxed energy and a fresh evaluation
    // agree only to about the Ewald accuracy.
    CHECK_THAT(again.energy, WithinAbs(from_document, 1e-6));
}
