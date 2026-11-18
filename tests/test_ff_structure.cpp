// One structure under the force field, through the layer `klasp energy`,
// klasp-energy and the Python module share (klasp/ff_structure.hpp).

#include "kokkos_listener.hpp"
#include "test_support.hpp"
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
#include <array>
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

namespace {

// Paracetamol as one molecule and the P2_1/a operators that generate the other
// three. The copies' fractional centres then spread over more than a cell, as
// they do in a CSP run.
json paracetamol_p21a() {
    json doc = example("paracetamol.json");
    json& mols = doc["crystal"]["molecules"];
    mols = json::array({mols[0]});
    doc["space_group"] = 14;
    doc["symmetry"] = json::array();
    for (const char* op : {"x, y, z", "-x+1/2, y+1/2, -z", "-x, -y, -z", "x+1/2, -y+1/2, z"})
        doc["symmetry"].push_back({{"molecule", 0}, {"op", op}});
    return doc;
}

// One structure in a batch slot, with the translations set directly so the
// loader's own canonicalisation cannot hide where a molecule sits.
klasp::testing::SingleSlot batch_slot(const json& doc) {
    auto ws = allocate_workspace(512, 27 * 27 * 27, 50000, 32, 4);
    MoleculeData mol{};
    load_structure_json(doc, mol, ws);
    return klasp::testing::make_single_slot(mol, ws);
}

std::array<RealType, 3> translation_of(klasp::testing::SingleSlot& s) {
    auto h = Kokkos::create_mirror(Kokkos::HostSpace(), s.bws.dofs.translations);
    Kokkos::deep_copy(h, s.bws.dofs.translations);
    return {h(0, 0, 0), h(0, 0, 1), h(0, 0, 2)};
}

void set_translation(klasp::testing::SingleSlot& s, const std::array<RealType, 3>& t) {
    klasp::testing::set_translation(s, 0, t[0], t[1], t[2]);
}

}  // namespace

TEST_CASE("moving a molecule by a lattice vector leaves the energy unchanged",
          "[ff-structure][images]") {
    auto slot = batch_slot(paracetamol_p21a());
    REQUIRE(slot.n_indep == 1);
    const auto t0 = translation_of(slot);
    const RealType e0 = klasp::testing::single_point<4>(slot);
    for (const auto& n : std::vector<std::array<int, 3>>{
             {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, 0, -1}, {2, -1, 1}}) {
        INFO("shift " << n[0] << " " << n[1] << " " << n[2]);
        auto t = t0;
        for (int d = 0; d < 3; ++d) t[d] += n[d];
        set_translation(slot, t);
        CHECK_THAT(klasp::testing::single_point<4>(slot), WithinAbs(e0, 1e-12));
    }
}

namespace {

// Paracetamol in R-3 given in the hexagonal setting, as CSPy writes it: a = b,
// gamma = 120 and eighteen operators, the six of -3 under the centring
// translations (2/3, 1/3, 1/3) and (1/3, 2/3, 2/3).
json paracetamol_r3_hexagonal() {
    json doc = example("paracetamol.json");
    json& mols = doc["crystal"]["molecules"];
    mols = json::array({mols[0]});
    mols[0]["translation"] = {0.13, 0.21, 0.07};
    doc["crystal"]["cell"] = {{"a", 16.0}, {"b", 16.0}, {"c", 30.0},
                              {"alpha", 90.0}, {"beta", 90.0}, {"gamma", 120.0}};
    doc["space_group"] = 148;
    doc["symmetry"] = json::array();
    for (const char* centring : {"", "2/3+", "1/3+"}) {
        const std::string t = centring;
        const std::string u = t == "2/3+" ? "1/3+" : t == "1/3+" ? "2/3+" : "";
        for (const auto& op : std::vector<std::array<std::string, 3>>{
                 {"x", "y", "z"}, {"-y", "x-y", "z"}, {"-x+y", "-x", "z"},
                 {"-x", "-y", "-z"}, {"y", "-x+y", "-z"}, {"x-y", "x", "-z"}})
            doc["symmetry"].push_back(
                {{"molecule", 0}, {"op", t + op[0] + "," + u + op[1] + "," + u + op[2]}});
    }
    return doc;
}

void check_hexagonal(const std::array<double, 6>& cell) {
    CHECK_THAT(cell[0], WithinAbs(cell[1], 1e-6));
    CHECK_THAT(cell[3], WithinAbs(90.0, 1e-6));
    CHECK_THAT(cell[4], WithinAbs(90.0, 1e-6));
    CHECK_THAT(cell[5], WithinAbs(120.0, 1e-6));
}

}  // namespace

// The rhombohedral strain basis would tilt a hexagonal-setting cell out of the
// lattice its operators describe, and the relaxation would stall.
TEST_CASE("an R-centred group in the hexagonal setting relaxes as a hexagonal cell",
          "[ff-structure][trigonal]") {
    FFRelaxOptions o;
    o.iterations = 300;
    const FFStructure s = ff_relax(paracetamol_r3_hexagonal(), o).front();
    CHECK(s.converged);
    check_hexagonal(s.cell_params);
}

TEST_CASE("a batch in the hexagonal setting of an R-centred group relaxes",
          "[ff-structure][trigonal][energy]") {
    json doc = paracetamol_r3_hexagonal();
    doc["structures"] = json::array(
        {json{{"name", "r3"}, {"cell", doc["crystal"]["cell"]},
              {"molecules", doc["crystal"]["molecules"]}}});
    const auto dir = std::filesystem::temp_directory_path() / "klasp_test_r3_hexagonal";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto path = dir / "batch.json";
    {
        std::ofstream f(path);
        f << doc.dump();
    }
    EnergyOptions o;
    o.relax = true;
    o.iterations = 300;
    o.output_dir = (dir / "relaxed").string();
    o.summary = (dir / "summary.json").string();
    cmd_energy({path.string()}, o);
    const json summary = read_json(o.summary);
    CHECK(summary.at("structures").at(0).at("converged").get<bool>());
    const json cell = read_json(dir / "relaxed" / "r3.json").at("crystal").at("cell");
    check_hexagonal({cell["a"].get<double>(), cell["b"].get<double>(), cell["c"].get<double>(),
                     cell["alpha"].get<double>(), cell["beta"].get<double>(),
                     cell["gamma"].get<double>()});
    std::filesystem::remove_all(dir);
}

namespace {

using Mat3 = std::array<std::array<double, 3>, 3>;

Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) C[i][j] += A[i][k] * B[k][j];
    return C;
}

Mat3 inverse(const Mat3& M) {
    double m[3][3], out[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i][j] = M[i][j];
    REQUIRE(mat3_inv(m, out));
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R[i][j] = out[i][j];
    return R;
}

Mat3 cell_matrix(const json& cell) {
    RealType p[6] = {cell["a"].get<double>(), cell["b"].get<double>(), cell["c"].get<double>(),
                     cell["alpha"].get<double>(), cell["beta"].get<double>(),
                     cell["gamma"].get<double>()};
    RealType m[3][3];
    cell_params_to_matrix(p, m);
    Mat3 C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) C[i][j] = m[i][j];
    return C;
}

// The same crystal on new axes: column j of P is new axis j in the old
// fractional coordinates. Molecule centres move to P^-1 t and orientations
// follow the new Cartesian frame (new a along x); `ops` are the operations on
// the new axes.
json change_axes(const json& doc_in, const Mat3& P, const std::vector<std::string>& ops) {
    json doc = doc_in;
    const Mat3 C_old = cell_matrix(doc["crystal"]["cell"]);
    const Mat3 axes = mul(C_old, P);
    auto len = [&](int j) {
        return std::sqrt(axes[0][j]*axes[0][j] + axes[1][j]*axes[1][j] + axes[2][j]*axes[2][j]);
    };
    auto ang = [&](int i, int j) {
        double d = 0;
        for (int r = 0; r < 3; ++r) d += axes[r][i] * axes[r][j];
        return std::acos(d / (len(i) * len(j))) * 180.0 / M_PI;
    };
    json cell = {{"a", len(0)}, {"b", len(1)}, {"c", len(2)},
                 {"alpha", ang(1, 2)}, {"beta", ang(0, 2)}, {"gamma", ang(0, 1)}};
    const Mat3 Q = mul(cell_matrix(cell), inverse(axes));
    const Mat3 Pinv = inverse(P);
    doc["crystal"]["cell"] = cell;
    for (json& m : doc["crystal"]["molecules"]) {
        const auto t = m["translation"].get<std::vector<double>>();
        std::vector<double> t_new(3, 0.0);
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) t_new[i] += Pinv[i][k] * t[k];
        m["translation"] = t_new;
        Mat3 R{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
        if (m.contains("rotation"))
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) R[i][j] = m["rotation"][i][j].get<double>();
        const Mat3 R_new = mul(Q, R);
        m["rotation"] = R_new;
    }
    doc["symmetry"] = json::array();
    for (const auto& op : ops) doc["symmetry"].push_back({{"molecule", 0}, {"op", op}});
    return doc;
}

void check_rhombohedral(const std::array<double, 6>& cell) {
    CHECK_THAT(cell[0], WithinAbs(cell[1], 1e-6));
    CHECK_THAT(cell[1], WithinAbs(cell[2], 1e-6));
    CHECK_THAT(cell[3], WithinAbs(cell[4], 1e-6));
    CHECK_THAT(cell[4], WithinAbs(cell[5], 1e-6));
}

// R-3 on rhombohedral (obverse) axes: a_R = (2a+b+c)/3, b_R = (-a+b+c)/3,
// c_R = (-a-2b+c)/3, with the 3-fold permuting them.
json paracetamol_r3_rhombohedral() {
    const Mat3 P{{{2.0/3, -1.0/3, -1.0/3}, {1.0/3, 1.0/3, -2.0/3}, {1.0/3, 1.0/3, 1.0/3}}};
    return change_axes(paracetamol_r3_hexagonal(), P,
                       {"x,y,z", "z,x,y", "y,z,x", "-x,-y,-z", "-z,-x,-y", "-y,-z,-x"});
}

// P2_1/a with the unique axis moved from b to c: new (a, b, c) = old (c, a, b).
json paracetamol_p21_unique_c() {
    const Mat3 P{{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}};
    return change_axes(paracetamol_p21a(), P,
                       {"x,y,z", "-x,-y+1/2,z+1/2", "-x,-y,-z", "x,y+1/2,-z+1/2"});
}

}  // namespace

// The operations, not a space-group number, set the cell's freedom, so one
// crystal relaxes the same way on either set of axes.
TEST_CASE("R-3 relaxes to the same structure on rhombohedral and hexagonal axes",
          "[ff-structure][trigonal]") {
    const json hex = paracetamol_r3_hexagonal();
    const json rh = paracetamol_r3_rhombohedral();
    // The Ewald sums run over different cells, so the two agree only to the
    // Ewald accuracy (1e-4 kJ/mol at the default); tightened to compare minima.
    FFRelaxOptions o;
    o.iterations = 300;
    o.model.ewald_accuracy = 1e-10;
    REQUIRE_THAT(ff_evaluate(rh, o.model).energy,
                 WithinAbs(ff_evaluate(hex, o.model).energy, 1e-6));

    const FFStructure s_hex = ff_relax(hex, o).front();
    const FFStructure s_rh = ff_relax(rh, o).front();
    std::printf("  R-3: hexagonal axes %.10f, rhombohedral axes %.10f kJ/mol, diff %.2e\n",
                s_hex.energy, s_rh.energy, s_rh.energy - s_hex.energy);
    CHECK(s_hex.converged);
    CHECK(s_rh.converged);
    check_hexagonal(s_hex.cell_params);
    check_rhombohedral(s_rh.cell_params);
    CHECK_THAT(s_rh.energy, WithinAbs(s_hex.energy, 1e-6));
}

TEST_CASE("P2_1/c relaxes the same with unique axis b or c", "[ff-structure][monoclinic]") {
    const json b_unique = paracetamol_p21a();
    const json c_unique = paracetamol_p21_unique_c();
    REQUIRE_THAT(ff_evaluate(c_unique).energy, WithinAbs(ff_evaluate(b_unique).energy, 1e-8));

    FFRelaxOptions o;
    o.iterations = 300;
    const FFStructure s_b = ff_relax(b_unique, o).front();
    const FFStructure s_c = ff_relax(c_unique, o).front();
    std::printf("  P2_1/c: unique b %.10f, unique c %.10f kJ/mol, diff %.2e\n",
                s_b.energy, s_c.energy, s_c.energy - s_b.energy);
    CHECK(s_b.converged);
    CHECK(s_c.converged);
    CHECK_THAT(s_b.cell_params[3], WithinAbs(90.0, 1e-6));
    CHECK_THAT(s_b.cell_params[5], WithinAbs(90.0, 1e-6));
    CHECK_THAT(s_c.cell_params[3], WithinAbs(90.0, 1e-6));
    CHECK_THAT(s_c.cell_params[4], WithinAbs(90.0, 1e-6));
    CHECK_THAT(s_c.energy, WithinAbs(s_b.energy, 1e-6));
}

TEST_CASE("an R-centred group in the hexagonal setting needs no space_group to keep its cell",
          "[ff-structure][trigonal]") {
    json doc = paracetamol_r3_hexagonal();
    doc.erase("space_group");
    FFRelaxOptions o;
    o.iterations = 300;
    const FFStructure s = ff_relax(doc, o).front();
    CHECK(s.converged);
    check_hexagonal(s.cell_params);
}

TEST_CASE("a space_group that contradicts the operations is warned about",
          "[ff-structure][trigonal]") {
    json doc = paracetamol_r3_hexagonal();
    doc["space_group"] = 14;
    FFRelaxOptions o;
    o.iterations = 1;
    logging::Capture capture(logging::Level::Warn);
    ff_relax(doc, o);
    CHECK_THAT(capture.text(), ContainsSubstring("space_group 14"));
}

TEST_CASE("relaxing R-3 on rhombohedral axes is deterministic", "[ff-structure][trigonal]") {
    FFRelaxOptions o;
    o.iterations = 300;
    const FFStructure a = ff_relax(paracetamol_r3_rhombohedral(), o).front();
    const FFStructure b = ff_relax(paracetamol_r3_rhombohedral(), o).front();
    CHECK(a.energy == b.energy);
    for (int i = 0; i < 6; ++i) CHECK(a.cell_params[i] == b.cell_params[i]);
}
