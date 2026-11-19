#include <catch2/catch_test_macros.hpp>
#include "kokkos_listener.hpp"

#include "klasp/csp_run.hpp"
#include "klasp/config.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/io.hpp"
#include "klasp/log.hpp"

#include <sqlite_modern_cpp.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

using namespace klasp;

// A whole CSP with no MPI, checked against the klasp binary on one rank. The
// landscapes must be identical row for row, since seeds index the same
// quasi-random sequence. The test executable does not link MPI, so the reference
// shells out to klasp (built via tests/CMakeLists.txt).

namespace {

using Row = std::tuple<long long, int, int, double>;

std::vector<Row> landscape(const std::string& db_path) {
    std::vector<Row> rows;
    sqlite::database db(db_path);
    db << "select seed, space_group, stage, energy from structure "
          "order by space_group, stage, seed"
       >> [&](long long seed, int sg, int stage, double e) {
              rows.emplace_back(seed, sg, stage, e);
          };
    return rows;
}

std::filesystem::path scratch(const char* leaf) {
    auto p = std::filesystem::temp_directory_path() / "klasp_local_csp" / leaf;
    std::filesystem::remove_all(p);
    return p;
}

}  // namespace

TEST_CASE("a CSP with no MPI produces the landscape the MPI path does",
          "[csp][local]") {
    const auto input = std::filesystem::path(KLASP_SOURCE_DIR)
                     / "example-inputs" / "acetic_acid.json";
    REQUIRE(std::filesystem::exists(input));

    const auto mpi_dir = scratch("mpi");
    const auto local_dir = scratch("local");

    const std::string cmd =
        std::string(KLASP_BINARY_PATH) + " csp " + input.string() +
        " --sg 14 --target 24 -o " + mpi_dir.string() + " > /dev/null 2>&1";
    REQUIRE(std::system(cmd.c_str()) == 0);

    RunConfig cfg;
    cfg.control.basis_paths = {input.string()};
    cfg.control.space_groups = {14};
    cfg.control.targets[14] = 24;
    cfg.output.dir = local_dir.string();

    CspRunResult result;
    {
        logging::Capture quiet(logging::Level::Error);
        result = run_local_csp(cfg);
    }

    const auto a = landscape((mpi_dir / "csp.db").string());
    const auto b = landscape((local_dir / "csp.db").string());

    REQUIRE_FALSE(a.empty());
    CHECK(a.size() == b.size());
    CHECK(a == b);

    CHECK(result.stored > 0);
    CHECK(result.unique > 0);
    CHECK(result.unique <= result.stored);
    CHECK_FALSE(result.stopped_early);

    std::filesystem::remove_all(mpi_dir.parent_path());
}

TEST_CASE("prune --compack keeps one structure per packing, the lowest", "[csp][local][compack]") {
    const auto input = std::filesystem::path(KLASP_SOURCE_DIR)
                     / "example-inputs" / "acetic_acid.json";
    const auto dir = scratch("compack");
    RunConfig cfg;
    cfg.control.basis_paths = {input.string()};
    cfg.control.space_groups = {14};
    cfg.control.targets[14] = 24;
    cfg.output.dir = dir.string();
    {
        logging::Capture quiet(logging::Level::Error);
        run_local_csp(cfg);
    }
    const std::string db = (dir / "csp.db").string();

    // The lowest structure again under another seed: the same packing, by construction.
    auto rows = read_landscape(db, 2, 0, 1e30, /*unique_only=*/true, true, -1000.0);
    REQUIRE(rows.size() >= 2);
    StructureResult copy = rows[0];
    copy.seed += 1000000;
    rows.insert(rows.begin() + 1, copy);

    PruneOptions po;
    const BasisConfig basis = load_csp_basis("", db);
    std::vector<int> lead;
    {
        logging::Capture quiet(logging::Level::Error);
        lead = compack_lead({db}, basis, rows, po.compack_opts);
    }
    REQUIRE(lead.size() == rows.size());
    CHECK(lead[0] == 0);
    CHECK(lead[1] == 0);
    for (std::size_t i = 0; i < lead.size(); ++i) CHECK(lead[i] <= static_cast<int>(i));

    // Merging can only remove structures from what prune writes.
    po.db_path = db;
    po.output_dir = (dir / "plain").string();
    PruneResult plain, merged;
    {
        logging::Capture quiet(logging::Level::Error);
        plain = cmd_prune(po);
        po.compack = true;
        po.output_dir = (dir / "merged").string();
        merged = cmd_prune(po);
    }
    REQUIRE(plain.ok);
    REQUIRE(merged.ok);
    CHECK(merged.unique <= plain.unique);

    // The CSV names each structure as the CIF does, and gives its stage.
    {
        std::ifstream csv(plain.csv_path);
        std::string header, first;
        std::getline(csv, header);
        std::getline(csv, first);
        CHECK(header.substr(header.size() - 9) == ",stage,id");
        const std::string id = first.substr(first.rfind(',') + 1);
        std::ifstream cif(plain.cif_path);
        std::string block;
        std::getline(cif, block);
        CHECK(block == "data_" + id);
    }

    // --window reads only the structures near the lowest.
    po.compack = false;
    po.window = 1.0;
    po.output_dir = (dir / "window").string();
    PruneResult windowed;
    {
        logging::Capture quiet(logging::Level::Error);
        windowed = cmd_prune(po);
    }
    REQUIRE(windowed.ok);
    CHECK(windowed.considered < plain.considered);
    CHECK(windowed.considered >= 1);
}
