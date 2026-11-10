// Crystal structures from the generator, in memory (klasp/structure_generation.hpp).

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "klasp/constants.hpp"
#include "klasp/csp_run.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/dedup.hpp"
#include "klasp/ff_structure.hpp"
#include "klasp/log.hpp"
#include "klasp/structure_generation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace klasp;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using json = nlohmann::json;

namespace {

std::string acetic_path() {
    return (std::filesystem::path(KLASP_SOURCE_DIR) / "example-inputs" / "acetic_acid.json")
        .string();
}

json acetic() {
    std::ifstream f(acetic_path());
    REQUIRE(f.is_open());
    return json::parse(f);
}

std::vector<int> seeds_of(const std::vector<GeneratedStructure>& structures) {
    std::vector<int> seeds;
    for (const auto& s : structures) seeds.push_back(s.seed);
    return seeds;
}

}  // namespace

TEST_CASE("generated structures are asymmetric units that evaluate", "[generate]") {
    GenerateOptions o;
    o.space_group = 14;
    o.count = 3;
    const auto structures = generate_structures({acetic()}, o);
    REQUIRE(structures.size() == 3u);
    for (size_t i = 0; i < structures.size(); ++i) {
        const auto& g = structures[i];
        INFO("seed " << g.seed);
        if (i > 0) CHECK(g.seed > structures[i - 1].seed);
        CHECK(g.structure.at("space_group") == 14);
        CHECK(g.structure.at("seed") == g.seed);
        CHECK(g.structure.at("crystal").at("molecules").size() == 1u);
        CHECK(g.structure.at("symmetry").size() == 4u);
        CHECK(std::isfinite(ff_evaluate(g.structure).energy));
    }

    // Relaxed, a structure still says which draw it came from.
    FFRelaxOptions relax;
    relax.iterations = 3;
    const auto relaxed = ff_relax(structures[1].structure, relax);
    REQUIRE(relaxed.size() == 1u);
    CHECK(relaxed.front().structure.at("seed") == structures[1].seed);
}

// A seed drawn here must give the same structure a CSP run draws for it.
TEST_CASE("a seed gives the structure a CSP run draws for it", "[generate]") {
    const auto dir = std::filesystem::temp_directory_path() / "klasp_test_generate_csp";
    std::filesystem::remove_all(dir);
    RunConfig cfg;
    cfg.control.basis_paths = {acetic_path()};
    cfg.control.space_groups = {14};
    cfg.control.targets[14] = 6;
    cfg.control.generate_only = true;
    cfg.output.dir = dir.string();
    {
        logging::Capture quiet(logging::Level::Error);
        run_local_csp(cfg);
    }
    const auto rows = read_landscape((dir / "csp.db").string(), 0, 14, 1e30,
                                     /*unique_only=*/false, /*exclude_catastrophes=*/false,
                                     0.0, /*exclude_warned=*/false);
    REQUIRE(rows.size() >= 6u);
    std::map<int, const StructureResult*> by_seed;
    for (const auto& r : rows) by_seed[r.seed] = &r;

    GenerateOptions o;
    o.space_group = 14;
    o.count = 6;
    const auto structures = generate_structures({acetic()}, o);
    REQUIRE(structures.size() == 6u);
    for (const auto& g : structures) {
        INFO("seed " << g.seed);
        const auto it = by_seed.find(g.seed);
        REQUIRE(it != by_seed.end());
        const StructureResult& row = *it->second;
        const json& cell = g.structure.at("crystal").at("cell");
        CHECK_THAT(cell.at("a").get<double>(),
                   WithinAbs(row.cell_params[0] * BOHR_TO_ANGSTROM, 1e-9));
        CHECK_THAT(cell.at("c").get<double>(),
                   WithinAbs(row.cell_params[2] * BOHR_TO_ANGSTROM, 1e-9));
        CHECK_THAT(cell.at("beta").get<double>(), WithinAbs(row.cell_params[4], 1e-9));
        const json& t = g.structure.at("crystal").at("molecules").at(0).at("translation");
        for (int d = 0; d < 3; ++d)
            CHECK_THAT(t[d].get<double>(), WithinAbs(row.translation[d], 1e-12));
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("a seed window picks up where the last one stopped", "[generate]") {
    GenerateOptions o;
    o.space_group = 14;
    o.count = 5;
    const auto all = generate_structures({acetic()}, o);
    REQUIRE(all.size() == 5u);

    o.count = 2;
    const auto first = generate_structures({acetic()}, o);
    o.seed = first.back().seed + 1;
    o.count = 3;
    const auto rest = generate_structures({acetic()}, o);

    std::vector<int> joined = seeds_of(first);
    for (const auto& s : rest) joined.push_back(s.seed);
    CHECK(joined == seeds_of(all));
    CHECK(rest.front().structure == all[2].structure);
}

// Two basis files, each naming its molecule "mol"; the structure must keep them
// apart.
TEST_CASE("a co-crystal places one molecule of each type", "[generate]") {
    std::ifstream f((std::filesystem::path(KLASP_SOURCE_DIR) / "example-inputs" / "urea.json")
                        .string());
    REQUIRE(f.is_open());
    const json urea = json::parse(f);

    GenerateOptions o;
    o.space_group = 14;
    o.count = 2;
    const auto structures = generate_structures({acetic(), urea}, o);
    REQUIRE(structures.size() == 2u);
    for (const auto& g : structures) {
        INFO("seed " << g.seed);
        const json& types = g.structure.at("basis").at("molecule_types");
        const json& molecules = g.structure.at("crystal").at("molecules");
        REQUIRE(types.size() == 2u);
        REQUIRE(molecules.size() == 2u);
        CHECK(types[0].at("name") != types[1].at("name"));
        CHECK(molecules[0].at("type") != molecules[1].at("type"));
        CHECK(g.structure.at("symmetry").size() == 8u);
    }
}

TEST_CASE("fft packing draws structures too", "[generate]") {
    GenerateOptions o;
    o.space_group = 14;
    o.count = 2;
    o.generator.collision = CollisionMethod::FFT_PACK;
    const auto structures = generate_structures({acetic()}, o);
    REQUIRE(structures.size() == 2u);
    for (const auto& g : structures) CHECK(std::isfinite(ff_evaluate(g.structure).energy));
}

TEST_CASE("the generator is refused what it cannot do", "[generate]") {
    GenerateOptions o;
    o.space_group = 14;
    SECTION("the rigid press, which relaxes rather than generates") {
        o.generator.collision = CollisionMethod::RIGID_PRESS;
        CHECK_THROWS_WITH(generate_structures({acetic()}, o), ContainsSubstring("rigid_press"));
    }
    SECTION("a space group that does not exist") {
        o.space_group = 231;
        CHECK_THROWS_WITH(generate_structures({acetic()}, o), ContainsSubstring("231"));
    }
    SECTION("fft packing of a co-crystal") {
        o.generator.collision = CollisionMethod::FFT_PACK;
        CHECK_THROWS_WITH(generate_structures({acetic(), acetic()}, o),
                          ContainsSubstring("co-crystal"));
    }
}
